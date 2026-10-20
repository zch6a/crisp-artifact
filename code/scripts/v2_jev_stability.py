"""Jev API determinism / stability: re-score the same 200 dev questions and compare with the first run."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine

A = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl"))
B = read_scores(os.path.join(CACHE, "jev_t10_dev_rep2.jsonl"))
dev = [e for e in load_examples("dev") if e.ex_id in B]
x, y, top, same = [], [], [], 0
for e in dev:
    ks = [k for k in B[e.ex_id] if k in A[e.ex_id]]
    a = np.asarray([A[e.ex_id][k] for k in ks]); b = np.asarray([B[e.ex_id][k] for k in ks])
    x += list(a); y += list(b); same += int(np.allclose(a, b, atol=1e-6))
    top.append(len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10])) / min(10, len(a)))
x, y = np.asarray(x), np.asarray(y)
print(f"n={len(dev)} columns={len(x)}  pearson={np.corrcoef(x, y)[0, 1]:.5f}  mean|diff|={np.abs(x - y).mean():.4f}  "
      f"max|diff|={np.abs(x - y).max():.4f}  identical questions={same}/{len(dev)}  top-10 overlap={np.mean(top):.3f}")
bm, em = lexical("dev")
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
for name, S in (("run1", A), ("run2", B)):
    v = View(dev, combine(dev, f0, [S[e.ex_id] for e in dev], 0.02), "f1k1")
    print(f"  {name}: " + "  ".join(f"SRR@{k}={np.mean(v.r <= k):.4f}" for k in (10, 15, 20, 30)))
