"""Pooled paired EX comparisons for the new benchmarks (SNAILS warehouse, scale curve), question-clustered bootstrap.

Each flags_<gen>.json maps system -> per-question 0/1 list (same question order for every generator). Whole questions
are resampled, carrying all of their generator rows (rows are correlated across generators).
  nb_pooled.py <flags_glob> [--B 10000]
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("flags")
ap.add_argument("--B", type=int, default=10000)
ap.add_argument("--pairs", default="cert_e0.05:full,cert_e0.1:full,cert_e0.05:topk10_uncal,cert_e0.1:topk10_uncal,"
                                    "cert_e0.05:thresh05_uncal,cert_e0.1:thresh05_uncal")
a = ap.parse_args()
F = {os.path.basename(p)[6:-5]: json.load(open(p)) for p in sorted(glob.glob(a.flags))}
print("generators:", ", ".join(F))
rng = np.random.default_rng(0)
for pr in a.pairs.split(","):
    s, b = pr.split(":")
    D = [np.array(f[s], dtype=float) - np.array(f[b], dtype=float) for f in F.values() if f.get(s) and f.get(b) and len(f[s]) == len(f[b])]
    if not D:
        continue
    D = np.stack(D)                                  # generators x questions (None = gold did not execute -> dropped)
    D = D[:, np.isfinite(D).all(0)]
    q = D.mean(0); n = len(q); obs = q.mean() * 100
    bs = np.array([q[rng.integers(0, n, n)].mean() for _ in range(a.B)]) * 100
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    per = " ".join(f"{x*100:+.1f}" for x in D.mean(1))
    print(f"{s:>9} vs {b:<15} {obs:+.2f} pp [{lo:+.2f}, {hi:+.2f}] p={max(p, 1/a.B):.4f}  (G={len(D)}, n={n})  per-gen: {per}")
