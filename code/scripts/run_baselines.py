"""基线对比（开题报告 实验 3）：同一过滤率下比召回。

关键：不给每个基线固定 top-k，而是扫描过滤率 0.1→0.9，
画 recall-filter 曲线。这样 BM25 / embedding / cross-encoder / Jev-Schema
在同一根横轴上可比，也能直接读出"recall≥98% 时过滤率是多少"。
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

from jevschema.baselines import build_baseline  # noqa: E402
from jevschema.data_loader import RAW_DIR, load_spider_split, load_spider_schemas  # noqa: E402
from jevschema.metrics import auroc, mean_ci, recall_at_filter_rate  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", default="bm25,embedding,random")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross-encoder", default="", help="训练好的 cross-encoder 目录")
    ap.add_argument("--rates", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rates = [float(x) for x in a.rates.split(",")]
    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    # 与 compare_scorers.py 统一取样口径（shuffle=False，取前 N 条）：
    # 之前这里是 shuffle=True，导致基线和方法跑在两组不同的样本上，数字不可比。
    examples = load_spider_split(os.path.join(RAW_DIR, "spider_val.parquet"), schemas,
                                 limit=a.n if a.n > 0 else None, seed=a.seed, shuffle=False)
    print(f"examples: {len(examples)}")

    names = a.baselines.split(",")
    if a.cross_encoder and "cross-encoder" not in names:
        names.append("cross-encoder")

    out: dict = {"n": len(examples), "rates": rates, "curves": {}, "auroc": {}, "recall_ci": {}}
    for name in names:
        kw = {}
        if name == "embedding":
            kw["model_path"] = a.embedding_model
        if name == "cross-encoder":
            kw["model_path"] = a.cross_encoder
        try:
            base = build_baseline(name, **kw)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        per_rate = {r: [] for r in rates}
        aucs = []
        for ex in examples:
            scores = base.score(ex)
            pairs = list(scores.keys())
            s = np.array([scores[p] for p in pairs])
            gold = ex.gold.flat_pairs()
            y = np.array([1 if p in gold else 0 for p in pairs])
            aucs.append(auroc(s, y))
            for r in rates:
                rec, _ = recall_at_filter_rate(s, y, r)
                per_rate[r].append(rec)
        curve = {str(r): round(float(np.mean(v)), 4) for r, v in per_rate.items()}
        ci = {str(r): [round(x, 4) for x in mean_ci(v)[1:]] for r, v in per_rate.items()}
        out["curves"][name] = curve
        out["recall_ci"][name] = ci
        out["auroc"][name] = round(float(np.nanmean(aucs)), 4)
        print(f"{name:14s} auroc={out['auroc'][name]:.4f}")
        print("   filter:", " ".join(f"{r:.1f}:{curve[str(r)]:.3f}" for r in rates))

    os.makedirs(RESULTS, exist_ok=True)
    path = a.out or os.path.join(RESULTS, f"baselines_n{a.n}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
