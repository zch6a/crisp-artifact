import json, sys, numpy as np
F = json.load(open(sys.argv[1])); ref = sys.argv[2]
rng = np.random.default_rng(0)
for s, fl in F.items():
    if s == ref: continue
    pr = [(a, b) for a, b in zip(F[ref], fl) if a is not None and b is not None]
    d = np.array([b - a for a, b in pr]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    print(f"{s:28s} diff={d.mean():+.4f} 95%CI[{np.quantile(bs,.025):+.4f},{np.quantile(bs,.975):+.4f}] "
          f"p={2*min(np.mean(np.array(bs)<=0), np.mean(np.array(bs)>=0)):.3f} wins/losses={int((d>0).sum())}/{int((d<0).sum())}")
