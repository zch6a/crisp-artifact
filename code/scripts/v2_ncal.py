"""Label efficiency of certification: calibrate on n dev questions (random), test on the rest."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, evaluate_budgets, fixed_family

dev = load_examples("dev"); bm, em = lexical("dev")
f_d = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
L = read_scores(os.path.join(CACHE, "laya_t10_dev.jsonl")); B = read_scores(os.path.join(CACHE, "block_t10_dev.jsonl"))
V = {"pointwise": View(dev, combine(dev, f_d, [L[e.ex_id] for e in dev], 0.02), "f1k1"),
     "block": View(dev, combine(dev, f_d, [B[e.ex_id] for e in dev], 0.02), "f1p5s15k1")}
rng = np.random.default_rng(1); out = []
for name, v in V.items():
    fam = fixed_family(v.r, v.n_cols)
    for eps in (0.02, 0.05):
        for n in (25, 50, 100, 200, 400, 767):
            cov, cb = [], []
            for rep in range(300):
                p = rng.permutation(len(dev)); cal, te = p[:n], p[n:]
                lam = calibrate(fam, cal, eps); ks, cv = evaluate_budgets(fam, lam, te, v.r)
                cov.append(cv.mean()); cb.append(np.mean([v.size(i, k) for i, k in zip(te, ks)]))
            cov = np.asarray(cov)
            row = {"reader": name, "eps": eps, "n_cal": n, "cov": float(cov.mean()), "cov_p05": float(np.quantile(cov, .05)),
                   "p_cov_ge_target": float(np.mean(cov >= 1 - eps)), "cbar": float(np.mean(cb))}
            out.append(row)
            print(f"{name:9s} eps={eps:.2f} n_cal={n:<4} cov={row['cov']:.4f} p05={row['cov_p05']:.4f} Cbar={row['cbar']:5.1f}", flush=True)
json.dump(out, open(os.path.join(RES, "ncal.json"), "w"), indent=1)
