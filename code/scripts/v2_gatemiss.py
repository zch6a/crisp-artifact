"""Can Jev's own confidence detect a table-gate miss?  (signal for a second gate round)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, load_examples, read_scores
from jevschema.metrics import auroc
for n in (500, 2000):
    exs = load_examples(f"scale{n}"); J = read_scores(os.path.join(CACHE, f"jev_fk_scale{n}.jsonl"))
    miss, feat = [], []
    for e in exs:
        if e.ex_id not in J:
            continue
        s = J[e.ex_id]; gated = {t for t, _ in s}
        m = not all(any(t in gated for (t, c) in req) for req in e.gold.requirements)
        p = np.sort(np.asarray(list(s.values())))[::-1]
        # per-requirement heuristic: how many confident columns vs how many tables look relevant
        feat.append([p[0], p[:3].mean(), (p > 0.5).sum(), (p > 0.2).sum()]); miss.append(1 if m else 0)
    F = np.asarray(feat); y = np.asarray(miss)
    print(f"N={n} n={len(y)} gate-miss rate={y.mean():.4f}")
    for j, name in enumerate(("max_p", "top3_mean", "n>0.5", "n>0.2")):
        print(f"   AUROC(low {name} -> miss) = {auroc(-F[:, j], y):.3f}")
    for th in (0.3, 0.5, 0.7):
        route = F[:, 0] < th
        print(f"   route if max_p<{th}: routed={route.mean():.3f} captures {y[route].sum()}/{y.sum()} misses")
