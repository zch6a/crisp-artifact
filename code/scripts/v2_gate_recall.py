"""Table-gate recall on the scaled benchmark: fraction of questions whose gold tables are all in the
top-T tables, for BM25 max-column gating vs fused(bm25+emb) max-column gating."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, fused_scores, load_examples, read_scores

for n in (250, 500, 1000, 2000):
    exs = load_examples(f"scale{n}")
    bm = read_scores(os.path.join(CACHE, f"scale{n}_bm25.jsonl")); em = read_scores(os.path.join(CACHE, f"scale{n}_emb.jsonl"))
    res = {}
    for name in ("bm25", "fused"):
        for T in (10, 20, 30):
            ok = []
            for e in exs:
                s = bm[e.ex_id] if name == "bm25" else fused_scores(e, bm[e.ex_id], em[e.ex_id])
                best = {}
                for (t, c), v in s.items():
                    best[t] = max(best.get(t, -1e9), v)
                top = {t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:T]}
                gt = {t for req in e.gold.requirements for (t, c) in req}
                ok.append(all(any(t in top for (t, c) in req) for req in e.gold.requirements))
            res[f"{name}@{T}"] = np.mean(ok)
    print(f"N={n}: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
