import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
dev = load_examples("dev"); bm, em = lexical("dev")
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
R = {k: [] for k in (10, 15, 20, 35)}
for f in ("block_t10_dev.jsonl", "block_t10_dev_s1.jsonl", "block_t10_dev_s2.jsonl"):
    S = read_scores(os.path.join(CACHE, f))
    v = View(dev, combine(dev, f0, [S[e.ex_id] for e in dev], 0.02), "f1p5s15k1")
    for k in R: R[k].append(np.mean(v.r <= k))
    print(f, " ".join(f"@{k}={R[k][-1]:.4f}" for k in R))
print("mean+-sd:", " ".join(f"@{k}={np.mean(v):.4f}+-{np.std(v, ddof=1):.4f}" for k, v in R.items()))
