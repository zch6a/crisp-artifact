"""Database-conditional certification (two-level conformal).

Per calibration DB g compute its own split-conformal threshold q_g(eps).  For a new DB use
lam = the m-th largest of {q_1..q_K}.  If DBs are exchangeable, P(q_new <= lam) >= (K-m+1)/(K+1),
and on that event the new DB's within-DB coverage is >= 1-eps (up to its own finite-sample slack).
m=1 (max) -> DB-level guarantee with prob K/(K+1).  Evaluated by leave-one-DB-out on BIRD-Dev.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, conformal_quantile, evaluate_budgets, fixed_family

dev = load_examples("dev")
bm, em = lexical("dev")
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
L = read_scores(os.path.join(CACHE, "laya_t10_dev.jsonl")); B = read_scores(os.path.join(CACHE, "block_t10_dev.jsonl")); J = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl"))
views = {"fused": View(dev, f0, "f1p5s15k1"),
         "pointwise": View(dev, combine(dev, f0, [L[e.ex_id] for e in dev], 0.02), "f1k1"),
         "block": View(dev, combine(dev, f0, [B[e.ex_id] for e in dev], 0.02), "f1p5s15k1")}
if len(J) == len(dev):
    views["jev"] = View(dev, combine(dev, f0, [J[e.ex_id] for e in dev], 0.02), "f1p5s15k1")
dbs = sorted({e.db_id for e in dev}); db_of = np.asarray([dbs.index(e.db_id) for e in dev])
out = []
for name, v in views.items():
    fam = fixed_family(v.r, v.n_cols)
    for eps in (0.01, 0.02, 0.05):
        for method in ("pooled", "max", "2nd", "3rd"):
            covs, sizes, per = [], [], {}
            for d in range(len(dbs)):
                cal_dbs = [g for g in range(len(dbs)) if g != d]
                te = np.where(db_of == d)[0]
                if method == "pooled":
                    lam = calibrate(fam, np.where(db_of != d)[0], eps)
                else:
                    qs = sorted([calibrate(fam, np.where(db_of == g)[0], eps) for g in cal_dbs], reverse=True)
                    lam = qs[{"max": 0, "2nd": 1, "3rd": 2}[method]]
                ks, cv = evaluate_budgets(fam, lam, te, v.r)
                per[dbs[d]] = float(cv.mean())
                covs += list(cv); sizes += [v.size(i, k) for i, k in zip(te, ks)]
            ok = np.mean([c >= 1 - eps for c in per.values()])
            row = {"scorer": name, "eps": eps, "method": method, "cov": float(np.mean(covs)), "cbar": float(np.mean(sizes)),
                   "db_ok_frac": float(ok), "worst": min(per.values()), "per_db": per}
            out.append(row)
            print(f"{name:9s} eps={eps:.2f} {method:6s} cov={row['cov']:.4f} Cbar={row['cbar']:5.1f} "
                  f"DBs>=1-eps: {ok*11:.0f}/11 worst={row['worst']:.3f}", flush=True)
json.dump(out, open(os.path.join(RES, "dbcond_dev.json"), "w"), indent=1)
