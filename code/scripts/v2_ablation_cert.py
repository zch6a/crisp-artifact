"""Certified ablation of the final CRISP configuration (BIRD-Dev 1534, protocol A: R random 50/50 splits).

For every variant: realized coverage and certified Cbar at eps in {0.02, 0.05}; delta Cbar vs the full system with a
95% interval over the R shared splits (same calibration/test halves for all variants -> paired).
Full system = Jev tier-1 fused with tier-0 (beta 0.02), closure f1k1, thresh family, n_cal = 767.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, evaluate_budgets, fixed_family, hybrid_family, mass_family, thresh_family

R = int(os.environ.get("R", "200"))
t0 = time.time()
dev = load_examples("dev"); bm, em = lexical("dev")
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
T1 = {k: read_scores(os.path.join(CACHE, f"{src}_dev{sfx}.jsonl")) for k, src, sfx in
      (("jev", "jev_t10", ""), ("laya", "laya_t10", ""), ("block", "block_t10", ""), ("distilled", "block_t10", "_s7"))}
sc = {k: combine(dev, f0, [S.get(e.ex_id, {}) for e in dev], 0.02) for k, S in T1.items()}
sc["jev_beta0"] = combine(dev, f0, [T1["jev"].get(e.ex_id, {}) for e in dev], 0.0)
sc["tier0_only"] = f0

views = {}
def view(reader, cfg):
    if (reader, cfg) not in views:
        views[(reader, cfg)] = View(dev, sc[reader], cfg)
    return views[(reader, cfg)]

def fam(v, name):
    ss = [v.sorted_scores(i) for i in range(len(v.exs))]
    return {"thresh": lambda: thresh_family(v.r, v.n_cols, ss), "fixed": lambda: fixed_family(v.r, v.n_cols),
            "mass": lambda: mass_family(v.r, v.n_cols, [v.cum_mass(i) for i in range(len(v.exs))]),
            "hybrid8": lambda: hybrid_family(v.r, v.n_cols, ss, 8)}[name]()

VARIANTS = [  # (label, reader, closure, family, n_cal)
    ("CRISP (full)", "jev", "f1k1", "thresh", None),
    ("- tier-0 tie-break (beta=0)", "jev_beta0", "f1k1", "thresh", None),
    ("reader: distilled open reader", "distilled", "f1k1", "thresh", None),
    ("reader: block reader", "block", "f1k1", "thresh", None),
    ("reader: pointwise laya", "laya", "f1k1", "thresh", None),
    ("reader: tier-0 lexical only", "tier0_only", "f1k1", "thresh", None),
    ("closure: none (f0)", "jev", "f0", "thresh", None),
    ("closure: FK only (f1)", "jev", "f1", "thresh", None),
    ("closure: + pad/small-table (f1p5s15k1)", "jev", "f1p5s15k1", "thresh", None),
    ("family: fixed-k", "jev", "f1k1", "fixed", None),
    ("family: score mass", "jev", "f1k1", "mass", None),
    ("family: hybrid (k>=8)", "jev", "f1k1", "hybrid8", None),
    ("calibration n=50", "jev", "f1k1", "thresh", 50),
    ("calibration n=100", "jev", "f1k1", "thresh", 100),
    ("calibration n=200", "jev", "f1k1", "thresh", 200),
]
n = len(dev)
rng = np.random.default_rng(0)
splits = [rng.permutation(n) for _ in range(R)]
res = {}
for label, reader, cfg, fname, ncal in VARIANTS:
    v = view(reader, cfg); F = fam(v, fname)
    for eps in (0.02, 0.05):
        cov, cb = [], []
        for p in splits:
            cal, te = p[: n // 2], p[n // 2:]
            if ncal:
                cal = cal[:ncal]
            lam = calibrate(F, cal, eps)
            ks, cv = evaluate_budgets(F, lam, te, v.r)
            cov.append(cv.mean()); cb.append(np.mean([v.size(i, k) for i, k in zip(te, ks)]))
        res[(label, eps)] = (np.asarray(cov), np.asarray(cb))
    print(f"  done {label} ({time.time()-t0:.0f}s)", flush=True)
rows = []
print("\n| variant | eps | coverage | certified Cbar | dCbar vs full [95%] |")
for label, *_ in VARIANTS:
    for eps in (0.02, 0.05):
        cov, cb = res[(label, eps)]; base = res[("CRISP (full)", eps)][1]
        d = cb - base
        rows.append({"variant": label, "eps": eps, "cov": float(cov.mean()), "cbar": float(cb.mean()),
                     "d_cbar": float(d.mean()), "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]})
        print(f"| {label} | {eps} | {cov.mean():.4f} | {cb.mean():.1f} | {d.mean():+.1f} [{np.quantile(d,.025):+.1f},{np.quantile(d,.975):+.1f}] |", flush=True)
json.dump(rows, open(os.path.join(RES, "ablation_cert.json"), "w"), indent=1)
print(f"done ({(time.time()-t0)/60:.1f} min)")
