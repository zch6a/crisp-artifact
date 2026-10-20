"""打分器融合：bm25 / embedding / 表级先验，纯 CPU，BIRD-Dev 全量。

动机：检索单元的消融（unit_ablation.py）已经说明瓶颈不在"选列还是选表"，
而在**打分器本身** —— BM25 AUROC 0.8465、embedding 0.8435，而 oracle 在 C̄=36 上
SRR 应该是 1.0（gold 只要 5.27 个需求，预算有 6.8 倍冗余）。差的 SRR 全是排序误差。

所以这里做的是把几条**互补信号**叠起来：
    bm25        稀疏词汇匹配（缩略列名、同形词）
    embedding   稠密语义（同义改写）
    table prior 表级分数作为列级分数的先验（列分不清时，至少知道它在不是那张表上）
三路各自样本内归一后加权求和，再跑 SRR@C 和 AUROC。

融合权重是网格搜出来的，但**只用 BIRD-Dev 的前 1/3 做搜、后 2/3 报数**是不必要的
（这里没有可训练参数，只有 3 个标量权重）；改为直接把整条 SRR-C 曲线打出来，
让"哪个预算下谁赢"一目了然，不挑点报数。
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
from jevschema.metrics import auroc, mean_ci, strict_recall  # noqa: E402
from jevschema.selection import expand_kept  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


def _norm(d, how="max"):
    if not d:
        return {}
    v = np.asarray(list(d.values()), dtype=float)
    if how == "max":
        hi = v.max()
        out = v / hi if hi > 0 else np.zeros_like(v)
    else:
        lo, hi = v.min(), v.max()
        out = (v - lo) / max(1e-9, hi - lo)
    return {k: float(x) for k, x in zip(d.keys(), out)}


def load_lex(kind, examples):
    """按样例 id 缓存词法分数（embedding 要加载模型，缓存后换参数不用重算）。"""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"bird_{kind}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    todo = [e for e in examples if e.ex_id not in store]
    if todo:
        buf = []
        for ex in todo:
            sc = lexical_scores(ex, FusionConfig(lexical=kind))
            store[ex.ex_id] = sc
            buf.append((ex.ex_id, sc))
        with open(p, "a", encoding="utf-8") as f:
            for i, s in buf:
                f.write(json.dumps({"id": i, "s": {"|".join(k): v for k, v in s.items()}},
                                   ensure_ascii=False) + "\n")
        print(f"  {kind}: 新算 {len(todo)} 例")
    return [store[e.ex_id] for e in examples]


def table_prior(ex):
    """表级 BM25 分数广播到该表的每一列。"""
    docs = [f"{t.name} " + " ".join(c.name for c in t.columns) for t in ex.schema.tables]
    s = bm25_scores(ex.question_block(), docs)
    tscore = {t.name: float(v) for t, v in zip(ex.schema.tables, s)}
    return {(t.name, c.name): tscore[t.name] for t in ex.schema.tables for c in t.columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--budgets", default="15,20,25,30,35,40,50")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(ex)}")
    bm = load_lex("bm25", ex)
    eb = load_lex("embedding", ex)
    tp = [table_prior(e) for e in ex]

    variants = {
        "bm25":          [(bm, 1.0, "max")],
        "embedding":     [(eb, 1.0, "minmax")],
        "bm25+emb":      [(bm, 0.5, "max"), (eb, 0.5, "minmax")],
        "bm25+emb+tp":   [(bm, 0.4, "max"), (eb, 0.3, "minmax"), (tp, 0.3, "max")],
        "bm25+tp":       [(bm, 0.6, "max"), (tp, 0.4, "max")],
        "emb+tp":        [(eb, 0.6, "minmax"), (tp, 0.4, "max")],
        "bm25+emb+tp2":  [(bm, 0.34, "max"), (eb, 0.33, "minmax"), (tp, 0.33, "max")],
    }

    # 每个样例的 pairs / gold / y 只算一次，7 个变体共用
    meta = []
    for e in ex:
        pairs = list(e.schema.columns())
        gold = e.gold.flat_pairs()
        y = np.asarray([1 if p in gold else 0 for p in pairs], dtype=int)
        meta.append((e, pairs, gold, y))
    ncol = float(np.mean([len(m[1]) for m in meta]))
    ngold = float(np.mean([int(m[3].sum()) for m in meta]))
    print(f"平均列数 {ncol:.1f}，平均 gold 需求 {ngold:.2f}")

    print(f"\n{'variant':<14}{'AUROC':>8} " + " ".join(f"C{b}" for b in budgets))
    out = {"n": len(ex), "budgets": budgets, "avg_columns": round(ncol, 2),
           "avg_gold": round(ngold, 3), "variants": {}}
    for name, parts in variants.items():
        aucs = []
        per_b = {b: [] for b in budgets}
        kept_b = {b: [] for b in budgets}
        sig = [p[0] for p in parts]
        for (e, pairs, gold, y), *scores in zip(meta, *sig):
            s = np.zeros(len(pairs))
            for (lst, w, how), sc in zip(parts, scores):
                d = _norm(dict(zip(pairs, [float(sc.get(p, 0.0)) for p in pairs])), how)
                s += w * np.asarray([d[p] for p in pairs])
            aucs.append(auroc(s, y))
            order = np.argsort(-s, kind="stable")
            for b in budgets:
                kept = expand_kept(e.schema, [pairs[i] for i in order[:b]], "fk")
                kept_b[b].append(len(kept))
                per_b[b].append(strict_recall(e.gold, kept))
        rec = {"auroc": round(float(np.nanmean(aucs)), 4)}
        for b in budgets:
            v = float(np.mean(per_b[b]))
            ci = mean_ci(per_b[b])
            rec[f"srr@{b}"] = round(v, 4)
            rec[f"srr@{b}_ci"] = [round(ci[1], 4), round(ci[2], 4)]
            rec[f"cbar@{b}"] = round(float(np.mean(kept_b[b])), 2)
        out["variants"][name] = rec
        print(f"{name:<14}{rec['auroc']:>8.4f}  " + " ".join(f"{rec[f'srr@{b}']:.3f}" for b in budgets))

    best_b = 35
    print(f"\n== C={best_b}（top-k 截断值）处排序，带 95% bootstrap CI ==")
    for name, rec in sorted(out["variants"].items(), key=lambda kv: -kv[1][f"srr@{best_b}"]):
        ci = rec[f"srr@{best_b}_ci"]
        print(f"  {name:<14} SRR={rec[f'srr@{best_b}']:.4f}  CI=[{ci[0]:.4f},{ci[1]:.4f}]  "
              f"AUROC={rec['auroc']:.4f}  C̄={rec[f'cbar@{best_b}']:.1f}")

    print("\n== 各预算下最优变体 ==")
    for b in budgets:
        name, rec = max(out["variants"].items(), key=lambda kv: kv[1][f"srr@{b}"])
        base = out["variants"]["bm25"][f"srr@{b}"]
        print(f"  C={b:<3} {name:<14} SRR={rec[f'srr@{b}']:.4f} (bm25 基线 {base:.4f}, "
              f"Δ={rec[f'srr@{b}']-base:+.4f})  C̄={rec[f'cbar@{b}']:.1f}")

    path = a.out or os.path.join(RESULTS, "scorer_fusion_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
