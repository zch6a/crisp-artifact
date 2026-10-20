"""Schema Scaling（纯 CPU 版）：词法 linking 在超大库上退化到什么程度？

动机来自 `laya_pareto.py` 的结论：System-One 排序质量远好于词法（等 C̄ 时
SRR +0.12~0.19），但 linking 成本 1788 tokens/查询；而全 schema 直喂的成本是
**O(全库列数)**。所以"什么时候该路由到 System-One"取决于**全 schema prompt 有多大**。

这条推论目前只是粗算（盈亏平衡点 ≈2100 tokens ≈ 250 列），必须实测。
这里就用 `build_scaling_benchmark` 把目标库塞进合成大库（并入的库是纯干扰项，
gold 不变），量三件事随库规模怎么变：
    1. 全 schema 直喂的 prompt token（分子，随规模线性涨）
    2. 词法 top-k + 结构闭包能不能守住 SRR（分母，随规模退化）
    3. 压缩省下的 token 随规模涨多少（这是 System-One 1788 tokens 的对手）

为什么用 Spider 构造而不是 BIRD：Spider 的 gold 抽取和表名前缀改写是现成的
（`build_scaling_benchmark` 就是按 Spider 写的），而且 Spider 库小、可控。
⚠️ Spider 侧没有 embedding 缓存，所以融合退化成 **bm25-only** —— 这是保守版本。

产物：每个规模下的 (full_tokens, 各 k 的 SRR / tokens)，用来定位交叉点。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.data_loader import (build_scaling_benchmark, load_spider,  # noqa: E402
                                   load_spider_schemas, RAW_DIR)
from jevschema.gold_labels import extract_gold  # noqa: E402
from jevschema.metrics import mean_ci, strict_recall  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import make_stage_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")


def bm25_of(ex):
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + (getattr(ex, "evidence", "") or ""))
    return dict(zip(pairs, bm25_scores(q, docs)))


def _hard_scale(base, target_columns):
    """干扰项 = 目标库自己的重命名副本。

    为什么需要这个：`build_scaling_benchmark` 并的是**别的库**，那些表名/列名
    与问题毫无关系，BM25 一眼就能排掉（实测塞到 532 列，SRR 还是 1.0）。
    真实的企业库里难的是反过来 —— 同一个逻辑名（user_id / amount / status）
    在几十张表里重复出现，词法信号几乎给不出区分度。
    所以这里把 base 复制 N 份，每份 db_id 不同（merge_schemas 会用它当表名前缀），
    于是出现大量"几乎同名"的表。gold SQL 只引用带 base 前缀的那份，不变。
    """
    import copy
    from jevschema.data_loader import _prefix_sql_tables
    from jevschema.schema_graph import merge_schemas
    out = []
    for ex in ex_iter(base):
        b = ex.schema
        parts = [b]
        n = b.n_columns()
        j = 0
        while n < target_columns:
            j += 1
            dup = copy.deepcopy(b)
            dup.db_id = f"{b.db_id}_dup{j}"
            parts.append(dup)
            n += b.n_columns()
        merged = merge_schemas(f"{b.db_id}__hard{target_columns}", parts, prefix=True)
        new_sql = _prefix_sql_tables(ex.sql, b.db_id, b)
        ne = copy.copy(ex)
        ne.ex_id = f"{ex.ex_id}-h{target_columns}"
        ne.db_id = merged.db_id
        ne.schema = merged
        ne.sql = new_sql
        ne.gold = extract_gold(new_sql, merged)
        out.append(ne)
    return out


def ex_iter(seq):
    return list(seq)


def filter_empty(ex, tag=""):
    """剔除 gold 为空的样本。

    ⚠️ `SELECT count(*)` 这类查询抽不出任何 (table, column) 对，`flat_pairs()`
    返回空集，而 `strict_recall` 对空 gold **平凡返回 1.0** —— 留着它们会把 SRR
    系统性抬高。实测 Spider-Dev 有 42/1034 = 4.06% 是空 gold；这里 `_prefix_sql_tables`
    + `extract_gold` 改写后还可能再产生新的空集，所以**构造前后都要过滤**。
    """
    n0 = len(ex)
    ex = [e for e in ex
          if e.gold is not None and e.gold.flat_pairs()]
    if n0 != len(ex):
        print(f"  [{tag}] 剔除空 gold {n0 - len(ex)}/{n0} 条"
              f"（{100 * (n0 - len(ex)) / max(1, n0):.2f}%）："
              f"空集会让 strict_recall 平凡返回 1.0")
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sizes", default="80,200,500,1200,2500")
    ap.add_argument("--ks", default="10,15,20,30,40")
    ap.add_argument("--closure", default="f1p5s15k1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard", action="store_true",
                    help="困难模式：干扰项换成**目标库自己的重命名副本**（表名/列名几乎一样）。"
                         "跨领域的干扰库对 BM25 太好排除了（实测 532 列时 SRR 仍是 1.0），"
                         "测不出词法排序的退化；真正会卡住词法的是同域近似重复。")
    ap.add_argument("--target", type=float, default=0.95,
                    help="汇总表用的 SRR 门槛：只有达到它的 k 才有资格谈'省了多少'")
    ap.add_argument("--s1-cost", type=float, default=1788.0,
                    help="System-One（laya）实测 linking 成本 tokens/查询，作为盈亏基准")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    sizes = [int(x) for x in a.sizes.split(",")]
    cfg = parse_closure(a.closure)

    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    base = load_spider(limit=a.n, split="dev", shuffle=False)
    base = filter_empty(base, "base")
    print(f"Spider-Dev 前 {len(base)} 例（原始平均列数 "
          f"{np.mean([e.schema.n_columns() for e in base]):.0f}）  "
          f"closure={cfg.tag()}  ks={ks}")

    out = {"n": len(base), "closure": cfg.tag(), "ks": ks, "sizes": {}}
    print(f"\n{'规模':>7}{'实际列数':>9}{'full tok':>10}"
          + "".join(f"{'k='+str(k):>16}" for k in ks))
    print(f"{'':>7}{'':>9}{'':>10}" + "".join(f"{'SRR':>8}{'tok':>8}" for k in ks))

    for size in sizes:
        if a.hard:
            ex = _hard_scale(base, size)
        else:
            ex = build_scaling_benchmark(base, schemas, target_columns=size, seed=a.seed)
        # 表名前缀改写后 gold 可能变空（改写失败/聚合查询），构造后必须再过滤一次
        ex = filter_empty(ex, f"size={size}")
        ncols = float(np.mean([e.schema.n_columns() for e in ex]))
        full_tok = float(np.mean([
            make_stage_tokens(e.schema, e.question, getattr(e, "evidence", "") or "")
            .total_tokens for e in ex]))

        row = {"target": size, "actual_cols": round(ncols, 1),
               "full_tokens": round(full_tok, 1), "by_k": {}}
        line = f"{size:>7}{ncols:>9.0f}{full_tok:>10.0f}"
        for k in ks:
            srrs, toks = [], []
            for e in ex:
                pairs = e.schema.columns()
                sc = bm25_of(e)
                v = np.asarray([float(sc.get(p, 0.0)) for p in pairs], dtype=float)
                order = np.argsort(-v, kind="stable")
                kept = closure(e.schema, [pairs[j] for j in order], k, cfg)
                srrs.append(strict_recall(e.gold, kept))
                toks.append(make_stage_tokens(e.schema, e.question,
                                              getattr(e, "evidence", "") or "",
                                              kept_columns=kept).gen_prompt_tokens)
            s = float(np.mean(srrs)); t = float(np.mean(toks))
            row["by_k"][str(k)] = {"srr": round(s, 4),
                                   "tokens": round(t, 1),
                                   "ci": [round(x, 4) for x in mean_ci(srrs)[1:3]],
                                   "saving": round(1 - t / full_tok, 4)}
            line += f"{s:>8.4f}{t:>8.0f}"
        print(line, flush=True)
        out["sizes"][str(size)] = row

    # ⚠️ 上一版用 `saving * (srr>=target)` 当 key，一旦没有任何 k 达标，
    # 所有 key 都是 0，`max` 就退化成取第一个（k 最小的那个），于是打出
    # "最省 k=10，省下 4955 tok" 这种 SRR 只有 0.65 的假好数字。
    # 正确做法：先筛达标的，达标集合为空就明确报**不可达**。
    print(f"\n== 在 SRR≥{a.target:.2f} 约束下，压缩省下的 token 够不够付 System-One ==")
    print(f"{'规模':>7}{'列数':>8}{'full':>9}{'最省k':>7}{'该k SRR':>9}"
          f"{'省下 tok':>10}{'够不够付 1788':>15}")
    for size in sizes:
        r = out["sizes"][str(size)]
        ok = [(k, d) for k, d in r["by_k"].items() if d["srr"] >= a.target]
        if not ok:
            best_k = max(r["by_k"].items(), key=lambda kv: kv[1]["srr"])[0]
            print(f"{size:>7}{r['actual_cols']:>8.0f}{r['full_tokens']:>9.0f}"
                  f"{'—':>7}{r['by_k'][best_k]['srr']:>9.4f}{'不可达':>10}{'—':>15}")
            continue
        k, d = min(ok, key=lambda kv: kv[1]["tokens"])
        saved = r["full_tokens"] - d["tokens"]
        print(f"{size:>7}{r['actual_cols']:>8.0f}{r['full_tokens']:>9.0f}{k:>7}"
              f"{d['srr']:>9.4f}{saved:>10.0f}"
              f"{('够' if saved >= a.s1_cost else '不够'):>15}")

    path = a.out or os.path.join(RESULTS, "scaling_lex.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
