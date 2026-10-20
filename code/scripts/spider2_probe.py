"""Spider 2.0-lite（本地 SQLite 子集）当第三基准 + "真实企业库有多大"的实感检查。

为什么值得做：前面那条"省多少是库规模的函数"的结论，所有大库证据都来自**合成**
缩放基准，审稿人一句"合成干扰库不代表真实"就能怼回来。Spider2-lite 的 local 子集是
**真实**的企业/开源 SQLite 库（AdventureWorks、Db-IMDB、northwind、f1…），
正好补这个洞。

⚠️ 样本量必须诚实说明：260 条里只有 **24 条** 同时满足
  (a) 是 local_*（有本地 sqlite）、(b) 有对应的 gold_sql 文件。
  bq*/sf* 那 170 多条虽然有 gold，但 BigQuery / Snowflake 的 schema 不在本地，
  没有 DDL 就抽不出列，**不能硬造**。所以这是"真实但小样本"的第三基准，
  作用是**定性的方向验证 + 真实库规模分布**，不是用来刷分的。

产物两块：
  1. 43 个真实 SQLite 库的列数分布（不需要 gold，纯 PRAGMA，样本可以很大）
     → 回答"真实企业库到底多大"，给合成缩放的规模区间提供现实依据。
  2. 24 条带 gold 的实例跑同一套 融合+结构闭包 流水线 → full tokens / SRR@k / 省了多少。
     ⚠️ 侧没有 embedding 缓存，融合退化成 bm25-only（保守版本）。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import jevschema.paths as _P  # noqa: E402  (env CRISP_DATA / CRISP_MODELS)

import numpy as np  # noqa: E402

from jevschema.data_loader import Example  # noqa: E402
from jevschema.gold_labels import extract_gold  # noqa: E402
from jevschema.metrics import mean_ci, strict_recall  # noqa: E402
from jevschema.schema_graph import Column, ForeignKey, SchemaGraph, Table  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import make_stage_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
RAW = os.path.join(_P.DATA, "raw", "spider2")
SQLITE_DIR = os.path.join(_P.DATA, "spider2_sqlite")


def schema_from_sqlite(path: str, db_id: str) -> SchemaGraph:
    """PRAGMA 直接读真实库结构（含外键），不用 DDL 文本解析。"""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        tables, fks = [], []
        for t in names:
            cols = []
            for r in con.execute(f'PRAGMA table_info("{t}")'):
                cols.append(Column(name=r[1], dtype=(r[2] or "text"), is_pk=bool(r[5])))
            if cols:
                tables.append(Table(name=t, columns=cols))
            for r in con.execute(f'PRAGMA foreign_key_list("{t}")'):
                if r[2] and r[3]:
                    fks.append(ForeignKey(src_table=t, src_col=r[3],
                                          dst_table=r[2], dst_col=r[4] or ""))
        return SchemaGraph(db_id=db_id, tables=tables, fks=fks)
    finally:
        con.close()


def _resolve_sqlite(db_id: str) -> str | None:
    """jsonl 里的 db 名与 sqlite 文件名大小写/下划线可能不完全一致。"""
    if not os.path.isdir(SQLITE_DIR):
        return None
    files = os.listdir(SQLITE_DIR)
    for f in files:
        if f.lower() == f"{db_id.lower()}.sqlite":
            return os.path.join(SQLITE_DIR, f)
    key = db_id.lower().replace("_", "")
    for f in files:
        if f.lower().replace("_", "").replace(".sqlite", "") == key:
            return os.path.join(SQLITE_DIR, f)
    return None


def bm25_of(ex):
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def _norm(d):
    v = np.asarray(list(d.values()), dtype=float)
    hi = v.max()
    return {k: float(x) for k, x in
            zip(d.keys(), v / hi if hi > 0 else np.zeros_like(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closure", default="f1p5s15k1")
    ap.add_argument("--ks", default="10,15,20,30,40,60,100")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    cfg = parse_closure(a.closure)

    rows = [json.loads(l) for l in open(
        os.path.join(RAW, "spider2-lite.jsonl"), encoding="utf-8")]
    gold_dir = os.path.join(RAW, "gold_sql")

    # ---------------------------------------------------------- 1. 真实库规模分布
    print("== 1. 真实 SQLite 企业库的规模分布（不需要 gold，纯 PRAGMA）==")
    sizes = []
    for f in sorted(os.listdir(SQLITE_DIR)) if os.path.isdir(SQLITE_DIR) else []:
        if not f.endswith(".sqlite"):
            continue
        try:
            g = schema_from_sqlite(os.path.join(SQLITE_DIR, f), f[:-7])
        except Exception:
            continue
        if g.n_columns() == 0:
            continue
        sizes.append((f[:-7], g.n_columns(), g.n_tables(), len(g.fks)))
    sizes.sort(key=lambda z: -z[1])
    print(f"{'库':<34}{'列数':>7}{'表数':>6}{'外键数':>7}")
    for n, c, t, k in sizes:
        print(f"{n:<34}{c:>7}{t:>6}{k:>7}")
    if sizes:
        cs = np.asarray([z[1] for z in sizes], dtype=float)
        print(f"\n  库数 {len(cs)}  列数: 中位 {np.median(cs):.0f}  "
              f"均值 {cs.mean():.0f}  最大 {cs.max():.0f}")
        for thr in (100, 200, 400, 800):
            print(f"    >{thr} 列的库: {int((cs > thr).sum())} 个 "
                  f"({100*float((cs>thr).mean()):.1f}%)")
        print("  → 与合成缩放基准的 80/200/500/1200/2500 列对照，"
              "看真实企业库落在哪一段。")

    # ---------------------------------------------------------- 2. 带 gold 的实例
    print("\n== 2. 带 gold 的 local 实例：跑同一套 融合(bm25 only)+结构闭包 ==")
    ex = []
    miss = {"sqlite": 0, "gold": 0, "empty_gold": 0}
    for r in rows:
        if not r["instance_id"].startswith("local"):
            continue
        gp = os.path.join(gold_dir, r["instance_id"] + ".sql")
        if not os.path.exists(gp):
            miss["gold"] += 1
            continue
        sp = _resolve_sqlite(r["db"])
        if sp is None:
            miss["sqlite"] += 1
            continue
        try:
            g = schema_from_sqlite(sp, r["db"])
        except Exception:
            miss["sqlite"] += 1
            continue
        if g.n_columns() == 0:
            miss["sqlite"] += 1
            continue
        sql = open(gp, encoding="utf-8").read()
        gold = extract_gold(sql, g)
        # ⚠️ 空 gold 会让 strict_recall 平凡返回 1.0，必须剔
        if gold is None or not gold.flat_pairs():
            miss["empty_gold"] += 1
            continue
        ex.append(Example(ex_id=r["instance_id"], db_id=r["db"],
                          question=r["question"], sql=sql, schema=g,
                          evidence=r.get("external_knowledge", "") or "",
                          gold=gold))
    print(f"  可用 {len(ex)} 条  （缺 sqlite {miss['sqlite']} / "
          f"缺 gold {miss['gold']} / 空 gold {miss['empty_gold']}）")
    if not ex:
        print("  ⚠️ 没有可用实例，退出")
        return

    cols = [e.schema.n_columns() for e in ex]
    print(f"  这些库的列数: 中位 {np.median(cols):.0f}  均值 {np.mean(cols):.0f}  "
          f"最大 {max(cols)}  （BIRD-Dev 中位 64 / 最大 199）")

    full_tok = float(np.mean([make_stage_tokens(e.schema, e.question,
                                                getattr(e, "evidence", "") or "")
                              .total_tokens for e in ex]))
    print(f"  全 schema 直给: {full_tok:.0f} tokens/query"
          f"（BIRD 650.9 / Spider 261.6）")

    orders = []
    for e in ex:
        pairs = e.schema.columns()
        b = _norm(bm25_of(e))
        s = np.asarray([float(b[p]) for p in pairs])
        orders.append([pairs[j] for j in np.argsort(-s, kind="stable")])

    print(f"\n{'k':>5}{'C̄':>8}{'tok':>9}{'SRR':>9}  95%CI{'':>4}省%")
    out_rows = []
    for k in ks:
        srrs, cb, tk = [], [], []
        for e, o in zip(ex, orders):
            kept = closure(e.schema, o, k, cfg)
            srrs.append(strict_recall(e.gold, kept))
            cb.append(len(kept))
            tk.append(make_stage_tokens(e.schema, e.question,
                                        getattr(e, "evidence", "") or "",
                                        kept_columns=kept).gen_prompt_tokens)
        s, t = float(np.mean(srrs)), float(np.mean(tk))
        ci = mean_ci(srrs)[1:3]
        out_rows.append({"k": k, "srr": round(s, 4), "cbar": round(float(np.mean(cb)), 2),
                         "tokens": round(t, 1), "saving": round(1 - t / full_tok, 4),
                         "ci": [round(x, 4) for x in ci]})
        print(f"{k:>5}{float(np.mean(cb)):>8.1f}{t:>9.0f}{s:>9.4f}  "
              f"[{ci[0]:.3f},{ci[1]:.3f}]{100*(1-t/full_tok):>7.1f}%")

    ok = [r for r in out_rows if r["srr"] >= a.target]
    if ok:
        r = min(ok, key=lambda z: z["tokens"])
        print(f"\n  达到 SRR≥{a.target:.2f} 的最省点: k={r['k']} "
              f"SRR={r['srr']:.4f} tokens={r['tokens']:.0f} "
              f"省 {100*r['saving']:.1f}%（省下 {full_tok - r['tokens']:.0f} tok）")
    else:
        m = max(out_rows, key=lambda z: z["srr"])
        print(f"\n  达不到 SRR≥{a.target:.2f}（最高 {m['srr']:.4f} @ k={m['k']}）")

    print(f"\n  ⚠️ n={len(ex)} 太小，95%CI 会非常宽（见上表）。"
          f"这个基准用来验证**方向和量级**，不能当主结果报。")

    path = a.out or os.path.join(RESULTS, "spider2_probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(ex), "closure": cfg.tag(),
                   "full_schema_tokens": round(full_tok, 1),
                   "db_sizes": [{"db": n, "cols": c, "tables": t, "fks": k}
                                for n, c, t, k in sizes],
                   "rows": out_rows}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
