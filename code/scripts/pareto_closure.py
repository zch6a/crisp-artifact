"""帕累托前沿：把"结构补全策略 × 列预算"全部摊到 (token, SRR) 平面上。

前面的实验都是固定 top-k 再比 SRR，但不同补全策略的 C̄ 差别很大
（fk 闭包 C̄=37.6，加表内补齐 p=12 后 C̄=53.6），**固定 k 比 SRR 是不公平的**。
论文要的图是前沿：横轴"每个查询花掉的 LLM token"，纵轴 SRR。

所以这里做的是：
    对每个补全策略 × 每个 top-k，算出 (平均 total tokens, SRR)，
    再在这些点上取帕累托前沿，并把参考点画在同一张平面上：
        - 全 schema 直给（不做 linking）        651 tokens, SRR=1.0（定义上）
        - AutoLink（arXiv 2511.17190）          8.0K linking + gen, SRR=0.974 @ C̄=35.8

补全策略（都是零模型调用的结构性规则）：
    f        外键闭包：补上跨表连接键
    p<N>     入选表内再补 N 列（按融合分取该表内次高的 N 列）
    s<M>     入选表里总列数 ≤ M 的，整表保留
    r<β>     比例补齐：入选表保留 ceil(β·该表列数) 列（与表大小自适应）

token 口径与 eval_bird 完全一致：linking 段只计**真要发给 LLM 的东西**
（纯词法模式 = 0），生成段用 tokens.make_stage_tokens 现算，不估。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import mean_ci, paired_bootstrap, strict_recall  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import make_stage_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


def _load(kind, examples, fn):
    p = os.path.join(CACHE, f"bird_{kind}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    if fn is None:      # 只读模式：缓存缺的留空，绝不用别的信号顶替
        return [store.get(e.ex_id, {}) for e in examples]
    todo = [e for e in examples if e.ex_id not in store]
    if todo:
        os.makedirs(CACHE, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for e in todo:
                sc = fn(e)
                store[e.ex_id] = sc
                f.write(json.dumps({"id": e.ex_id,
                                    "s": {"|".join(k): v for k, v in sc.items()}},
                                   ensure_ascii=False) + "\n")
    return [store[e.ex_id] for e in examples]


def bm25_of(ex):
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def _norm(d, how="max"):
    v = np.asarray(list(d.values()), dtype=float)
    if how == "max":
        hi = v.max()
        return {k: float(x) for k, x in
                zip(d.keys(), v / hi if hi > 0 else np.zeros_like(v))}
    lo, hi = v.min(), v.max()
    return {k: float(x) for k, x in zip(d.keys(), (v - lo) / max(1e-9, hi - lo))}


def parse_cfg(g):
    """直接复用 selection.parse_closure —— 两处各写一份解析规则必然悄悄分叉。"""
    return parse_closure(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ks", default="5,8,10,12,15,18,20,25,30,35,40")
    ap.add_argument("--grid", default="f0,f1,f1p2,f1p3,f1p5,f1p8,f1p12,"
                                      "f1s6,f1s10,f1r0.3,f1r0.5,f1r0.7,"
                                      "f1p3r0.3,f1p5r0.5")
    ap.add_argument("--bench", default="bird",
                    choices=["bird", "spider", "spider_train"],
                    help="论文要求 ≥3 个基准，同一个脚本不能只跑一个数据集。"
                         "spider_train 与 spider(dev) **按库划分、数据库不重叠**，"
                         "所以它是独立基准而不是同一批数据的第二次采样")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    B = a.bench                      # 缓存 tag 前缀：bird/spider 的分数绝不能混

    if B.startswith("spider"):
        from jevschema.data_loader import load_spider
        ex = load_spider(limit=a.n if a.n > 0 else None,
                         split=("dev" if B == "spider" else "train"), shuffle=False)
    else:
        schemas = load_bird_schemas()
        ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)

    # ⚠️ 空 gold 必须剔除：`SELECT count(*)` 这类查询抽不出列，gold 是空集，
    # 而 strict_recall 对空集平凡返回 1.0 —— 留着它们会把 SRR 系统性抬高。
    # BIRD-Dev 恰好 0 条（主战场数字不受影响），Spider-Dev 有 42/1034 = 4.06%。
    n0 = len(ex)
    ex = [e for e in ex if e.gold is not None and e.gold.flat_pairs()]
    if n0 != len(ex):
        print(f"  剔除空 gold {n0 - len(ex)} 条（{100*(n0-len(ex))/max(1,n0):.2f}%）："
              f"空集会让 strict_recall 平凡返回 1.0")
    bm = _load(f"{B}_bm25", ex, bm25_of)
    emb_cache = os.path.join(CACHE, f"{B}_embedding.jsonl")
    eb = _load(f"{B}_embedding", ex, None)  # 只读：缺缓存就退化成纯 bm25，不允许顶替
    have_emb = any(len(d) > 0 for d in eb)
    print(f"{B.upper()}-Dev n={len(ex)}  融合={'bm25+emb' if have_emb else 'bm25 only'}")

    meta = []
    for e in ex:
        meta.append((e, list(e.schema.columns()), e.gold.flat_pairs()))

    orders = []
    for (e, pairs, gold), b, d in zip(meta, bm, eb):
        bn = _norm({p: float(b.get(p, 0.0)) for p in pairs}, "max")
        dn = (_norm({p: float(d.get(p, 0.0)) for p in pairs}, "minmax")
              if have_emb else {p: 0.0 for p in pairs})
        s = np.asarray([0.5 * bn[p] + 0.5 * dn[p] for p in pairs])
        orders.append([pairs[j] for j in np.argsort(-s, kind="stable")])

    # 全 schema 参考（Spider 的 Example 没有 evidence 字段，用 getattr 兜住）
    full_tok = float(np.mean([
        make_stage_tokens(e.schema, e.question, getattr(e, "evidence", "") or "")
        .total_tokens for e, _, _ in meta]))

    print(f"全 schema 直给：{full_tok:.0f} tokens/query（SRR 定义为 1.0）")
    print(f"\n{'config':<12}{'k':>4}{'C̄':>7}{'gen_tok':>9}{'SRR':>8}")
    rows = []
    for g in a.grid.split(","):
        cfg = parse_cfg(g)
        per_srr = {}
        for k in ks:
            srrs, cbars, gtoks = [], [], []
            for (e, pairs, gold), o in zip(meta, orders):
                kept = closure(e.schema, o, k, cfg)
                st = make_stage_tokens(e.schema, e.question,
                                       getattr(e, "evidence", "") or "",
                                       kept_columns=kept)
                srrs.append(strict_recall(e.gold, kept))
                cbars.append(len(kept))
                gtoks.append(st.gen_prompt_tokens)
            per_srr[k] = srrs
            v = float(np.mean(srrs))
            rows.append({"config": g, "k": k, "srr": round(v, 4),
                         "cbar": round(float(np.mean(cbars)), 2),
                         "gen_tokens": round(float(np.mean(gtoks)), 1),
                         "total_tokens": round(float(np.mean(gtoks)), 1),
                         "ci": [round(x, 4) for x in mean_ci(srrs)[1:3]]})
            print(f"{g:<12}{k:>4}{float(np.mean(cbars)):>7.1f}"
                  f"{float(np.mean(gtoks)):>9.0f}{v:>8.4f}")

    # 帕累托前沿：x=tokens（越小越好），y=SRR（越大越好）
    pts = [r for r in rows]
    pts.sort(key=lambda r: (r["total_tokens"], -r["srr"]))
    front, best = [], -1.0
    for r in pts:
        if r["srr"] > best + 1e-9:
            front.append(r)
            best = r["srr"]
    print(f"\n== 帕累托前沿（x=LLM tokens/query 越小越好，y=SRR）共 {len(front)} 点 ==")
    print(f"{'config':<12}{'k':>4}{'tokens':>9}{'C̄':>7}{'SRR':>8}  95%CI")
    prev = None
    for r in front:
        d = "" if prev is None else f"  (+{r['total_tokens']-prev:.0f} tok → +{r['srr']-prev_srr:.4f} SRR)"
        print(f"{r['config']:<12}{r['k']:>4}{r['total_tokens']:>9.0f}{r['cbar']:>7.1f}"
              f"{r['srr']:>8.4f}  [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]{d}")
        prev, prev_srr = r["total_tokens"], r["srr"]

    # AutoLink 只报了 BIRD，Spider 上不硬套
    targets = (0.90, 0.95, 0.974) if B == "bird" else (0.90, 0.95, 0.99)
    print("\n== 与 SOTA 对照 ==")
    if B == "bird":
        print("  AutoLink: SRR 0.974 @ C̄=35.8, linking 8.0K tokens（LLM 迭代探索）")
    for target in targets:
        ok = [r for r in rows if r["srr"] >= target]
        if ok:
            r = min(ok, key=lambda r: r["total_tokens"])
            print(f"  我们达到 SRR≥{target:.3f} 的最省配置: {r['config']} k={r['k']} "
                  f"→ {r['total_tokens']:.0f} tokens (C̄={r['cbar']})  "
                  f"[纯词法，linking LLM tokens = 0]")
        else:
            m = max(rows, key=lambda r: r["srr"])
            print(f"  我们达不到 SRR≥{target:.3f}（全网格最高 {m['srr']:.4f} @ "
                  f"{m['config']} k={m['k']}）")

    # 前沿上最省的那个"够用"的点：SRR ≥ 前沿最高 - 0.01
    top = max(rows, key=lambda r: r["srr"])["srr"]
    near = [r for r in rows if r["srr"] >= top - 0.01]
    r = min(near, key=lambda r: r["total_tokens"])
    print(f"\n前沿最高 SRR={top:.4f}；SRR 损失 ≤0.01 的最省点：{r['config']} k={r['k']} "
          f"SRR={r['srr']:.4f} tokens={r['total_tokens']:.0f} C̄={r['cbar']:.1f}")

    path = a.out or os.path.join(RESULTS, f"pareto_closure_{B}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"bench": B, "n": len(ex), "full_schema_tokens": round(full_tok, 1),
                   "rows": rows, "front": front}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
