"""检索单元消融：列级 vs 表级 vs 混合（纯 CPU，用 BM25 打分，全量 BIRD-Dev）。

为什么这个实验要先做：
    SRR 是**逐题全有或全无**的。漏一个 gold 列整题归零，所以最优策略不是"把预算花在
    语义打分上"，而是"把预算花在保险上"。BIRD-Dev 平均 75.6 列 / 8 张表 / 每张表约 9 列，
    gold 只涉及 1.94 张表 —— **选对 4 张表并整表保留也就 36 列**，恰好是 AutoLink 的 C̄=35.8。
    也就是说真正决定成败的是**表级召回**，列级剪枝只对个别大表有意义。

    如果这个假设成立，方法的检索单元就应该改成"表优先"，而不是现在的"全局 top-k 列"。

四种策略：
    col_k          全局 top-k 列（当前实现）
    table_T        top-T 张表整表保留
    hybrid         top-T 张表，表内取 top-m 列
    hybrid_whole   top-T 张表，小表（<=W 列）整表保留，大表内取 top-m 列
所有策略之后都可选叠加 fk 补全 / 主键保留。
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

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.fusion import FusionConfig, lexical_scores  # noqa: E402
from jevschema.metrics import mean_ci, strict_recall  # noqa: E402
from jevschema.selection import expand_kept  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import TOKENIZER, build_gen_prompt, n_tokens, schema_prompt_text  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")


def bm25_all(ex, kind="bm25"):
    return lexical_scores(ex, FusionConfig(lexical=kind))


def table_scores(ex, kind="bm25"):
    docs = [f"{t.name} " + " ".join(c.name for c in t.columns) for t in ex.schema.tables]
    s = bm25_scores(ex.question_block(), docs)
    return {t.name: float(v) for t, v in zip(ex.schema.tables, s)}


def select(ex, sc, policy, k, T, m, W, expand, keep_pk):
    cols = list(ex.schema.columns())
    if policy == "col_k":
        s = np.asarray([float(sc.get(c, 0.0)) for c in cols])
        order = np.argsort(-s, kind="stable")
        kept = [cols[i] for i in order[:k]]
    else:
        ts = table_scores(ex)
        tabs = sorted(ts, key=lambda t: -ts[t])[:T]
        if policy == "table_T":
            kept = [(t, c) for t, c in cols if t in set(tabs)]
        else:
            kept = []
            for t in tabs:
                tcols = [(tt, c) for tt, c in cols if tt == t]
                tbl = ex.schema.table(t)
                if policy == "hybrid_whole" and tbl and len(tbl.columns) <= W:
                    kept.extend(tcols)
                    continue
                s = np.asarray([float(sc.get(c, 0.0)) for c in tcols])
                order = np.argsort(-s, kind="stable")
                kept.extend([tcols[i] for i in order[:m]])
    if keep_pk:
        tabs = {t for t, _ in kept}
        for t in ex.schema.tables:
            if t.name in tabs:
                kept.extend((t.name, c.name) for c in t.columns if c.is_pk)
    kept = list(dict.fromkeys(kept))
    return expand_kept(ex.schema, kept, expand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(ex)}  tokenizer={TOKENIZER}")
    print(f"平均列数 {np.mean([e.schema.n_columns() for e in ex]):.1f} / "
          f"平均表数 {np.mean([e.schema.n_tables() for e in ex]):.1f} / "
          f"平均 gold 表 {np.mean([len(e.gold.tables) for e in ex]):.2f}")
    sc = [bm25_all(e) for e in ex]

    grid = []
    for T in (1, 2, 3, 4, 5, 6, 8):
        grid.append(dict(policy="table_T", T=T, m=0, W=0, expand="fk", keep_pk=True))
    for k in (10, 20, 30, 35, 50):
        grid.append(dict(policy="col_k", k=k, T=0, m=0, W=0, expand="fk", keep_pk=True))
    for T in (3, 4, 5, 6):
        for m in (4, 6, 8, 12):
            grid.append(dict(policy="hybrid", T=T, m=m, k=0, W=0, expand="fk", keep_pk=True))
    for T in (4, 5, 6, 8):
        for W in (8, 12, 20):
            grid.append(dict(policy="hybrid_whole", T=T, m=6, k=0, W=W,
                             expand="fk", keep_pk=True))
    # 关掉 fk / pk 的对照（量化这两步各自的贡献）
    grid.append(dict(policy="table_T", T=4, m=0, k=0, W=0, expand="none", keep_pk=False))
    grid.append(dict(policy="table_T", T=4, m=0, k=0, W=0, expand="fk", keep_pk=False))
    grid.append(dict(policy="table_T", T=4, m=0, k=0, W=0, expand="none", keep_pk=True))

    rows = []
    for g in grid:
        srrs, kepts, trecs, gentoks = [], [], [], []
        for e, s in zip(ex, sc):
            kept = select(e, s, g["policy"], g.get("k", 0), g.get("T", 0),
                          g.get("m", 0), g.get("W", 0), g["expand"], g["keep_pk"])
            srrs.append(strict_recall(e.gold, kept))
            kepts.append(len(kept))
            trecs.append(e.gold.table_recall({t for t, _ in kept}))
            gentoks.append(n_tokens(build_gen_prompt(
                e.db_id, schema_prompt_text(e.schema, columns=kept), e.question, e.evidence)))
        srr = float(np.mean(srrs))
        ci = mean_ci(srrs)
        rows.append({**g, "srr": round(srr, 4), "srr_ci": [round(ci[1], 4), round(ci[2], 4)],
                     "C": round(float(np.mean(kepts)), 1),
                     "table_recall": round(float(np.mean(trecs)), 4),
                     "gen_tokens": round(float(np.mean(gentoks)), 0)})

    print(f"\n{'policy':<14}{'T':>3}{'k':>4}{'m':>4}{'W':>4}{'fk':>4}{'pk':>4}"
          f"{'SRR':>8}{'95%CI':>18}{'C̄':>7}{'表召回':>8}{'gen tok':>9}")
    for r in sorted(rows, key=lambda r: (-r["srr"], r["C"])):
        ci = f"[{r['srr_ci'][0]:.3f},{r['srr_ci'][1]:.3f}]"
        print(f"{r['policy']:<14}{r.get('T',0):>3}{r.get('k',0):>4}{r.get('m',0):>4}"
              f"{r.get('W',0):>4}{str(r['expand'] != 'none'):>4}{str(r['keep_pk']):>4}"
              f"{r['srr']:>8.4f}{ci:>18}{r['C']:>7.1f}{r['table_recall']:>8.4f}"
              f"{r['gen_tokens']:>9.0f}")

    path = a.out or os.path.join(RESULTS, "unit_ablation_bm25.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(ex), "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
