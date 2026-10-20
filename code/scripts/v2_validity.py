"""Validity of the certificate (reviewer question "why conformal, not a tuned threshold?").

Scorer: Jev tier-1 fused (jev10f), closure f1k1 (the main configuration).
  E1  calibration curve, protocol A: R random 50/50 dev splits, eps grid, families fixed/thresh.
      Reports mean realized coverage (guarantee: E[cov] >= 1-eps), 5th percentile, share of splits below 1-eps.
      Only the lower bound is claimed: depths are integers, ties make the method conservative.
  E2  small calibration sets: conformal quantile (ceil((n+1)(1-eps))) vs plug-in empirical quantile
      (ceil(n(1-eps))), n_cal in {25, 50, 100, 200}, R draws each.
  E3  shift, zero dev labels: rules tuned on BIRD-train holdout, applied to dev (new databases):
      plug-in threshold / plug-in fixed-k (no finite-sample correction) vs conformal, plus common
      untuned heuristics (top-k columns, probability >= 0.5).
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, evaluate_budgets, fixed_family, thresh_family

R = int(os.environ.get("R", "1000"))
EPS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
t0 = time.time()
dev, hold = load_examples("dev"), load_examples("holdout")
bm, em = lexical("dev"); hbm, hem = lexical("holdout")
f_dev = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
f_hold = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
jd = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl")); jh = read_scores(os.path.join(CACHE, "jev_t10_holdout.jsonl"))
keep_h = [i for i, e in enumerate(hold) if e.ex_id in jh]
hold = [hold[i] for i in keep_h]; f_hold = [f_hold[i] for i in keep_h]
vd = View(dev, combine(dev, f_dev, [jd.get(e.ex_id, {}) for e in dev], 0.02), "f1k1")
vh = View(hold, combine(hold, f_hold, [jh.get(e.ex_id, {}) for e in hold], 0.02), "f1k1")
print(f"loaded dev={len(dev)} holdout={len(hold)} ({time.time()-t0:.0f}s)", flush=True)


def fams(v):
    ss = [v.sorted_scores(i) for i in range(len(v.exs))]
    return {"fixed": fixed_family(v.r, v.n_cols), "thresh": thresh_family(v.r, v.n_cols, ss)}


def plugin_quantile(scores, eps):
    s = np.sort(np.asarray(scores, dtype=float)); n = len(s)
    k = int(math.ceil(n * (1 - eps)))
    return float(s[min(n, max(1, k)) - 1])


out = {"E1": [], "E2": [], "E3": []}
rng = np.random.default_rng(0)
FD = fams(vd)
n = len(dev)
# E1 -----------------------------------------------------------------------------------------------------------
for fname, fam in FD.items():
    for eps in EPS:
        covs, sizes = [], []
        for _ in range(R):
            p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
            lam = calibrate(fam, cal, eps)
            ks, cv = evaluate_budgets(fam, lam, te, vd.r)
            covs.append(cv.mean())
            if len(sizes) < 50:
                sizes.append(np.mean([vd.size(i, k) for i, k in zip(te, ks)]))
        covs = np.asarray(covs)
        row = {"family": fname, "eps": eps, "target": 1 - eps, "mean_cov": float(covs.mean()), "p5": float(np.quantile(covs, .05)),
               "share_below": float((covs < 1 - eps).mean()), "cbar": float(np.mean(sizes))}
        out["E1"].append(row)
        print(f"E1 {fname:6s} eps={eps:<5} target={1-eps:.3f} mean={row['mean_cov']:.4f} p5={row['p5']:.4f} "
              f"share<target={row['share_below']:.3f} Cbar={row['cbar']:.1f}", flush=True)
# E2 -----------------------------------------------------------------------------------------------------------
for fname, fam in FD.items():
    for eps in (0.05, 0.10):
        for ncal in (25, 50, 100, 200):
            cc, cp = [], []
            for _ in range(R):
                p = rng.permutation(n); cal, te = p[:ncal], p[ncal:]
                sc = [fam.nonconf(i) for i in cal]
                for lam, acc in ((calibrate(fam, cal, eps), cc), (plugin_quantile(sc, eps), cp)):
                    acc.append(evaluate_budgets(fam, lam, te, vd.r)[1].mean())
            row = {"family": fname, "eps": eps, "n_cal": ncal, "conformal": float(np.mean(cc)), "plugin": float(np.mean(cp)),
                   "conformal_share_below": float((np.asarray(cc) < 1 - eps).mean()), "plugin_share_below": float((np.asarray(cp) < 1 - eps).mean())}
            out["E2"].append(row)
            print(f"E2 {fname:6s} eps={eps} n_cal={ncal:<4} conformal={row['conformal']:.4f} plug-in={row['plugin']:.4f} "
                  f"(target {1-eps:.2f})", flush=True)
# E3 -----------------------------------------------------------------------------------------------------------
FH = fams(vh)
for fname in FD:
    for eps in (0.02, 0.05, 0.10):
        hs = [FH[fname].nonconf(i) for i in range(len(hold))]
        for rule, lam in (("conformal (holdout->dev)", calibrate(FH[fname], range(len(hold)), eps)),
                          ("plug-in (holdout->dev)", plugin_quantile(hs, eps))):
            ks, cv = evaluate_budgets(FD[fname], lam, range(n), vd.r)
            size = float(np.mean([vd.size(i, k) for i, k in zip(range(n), ks)]))
            out["E3"].append({"family": fname, "eps": eps, "rule": rule, "cov": float(cv.mean()), "cbar": size})
            print(f"E3 {fname:6s} eps={eps} {rule:26s} dev cov={cv.mean():.4f} (target {1-eps:.2f}) Cbar={size:.1f}", flush=True)
for k in (5, 10, 15, 20, 30):
    cv = np.mean([vd.r[i] <= k for i in range(n)]); size = np.mean([vd.size(i, k) for i in range(n)])
    out["E3"].append({"rule": f"top-{k} columns (untuned)", "cov": float(cv), "cbar": float(size)})
    print(f"E3 top-{k:<3} columns + closure (untuned)            dev cov={cv:.4f} Cbar={size:.1f}", flush=True)
for tau in (0.5, 0.3, 0.1):
    ks = [max(1, int(np.sum(vd.sorted_scores(i) >= tau))) for i in range(n)]
    cv = np.mean([vd.r[i] <= k for i, k in zip(range(n), ks)]); size = np.mean([vd.size(i, k) for i, k in zip(range(n), ks)])
    out["E3"].append({"rule": f"score >= {tau} (untuned)", "cov": float(cv), "cbar": float(size)})
    print(f"E3 score>={tau} + closure (untuned)                 dev cov={cv:.4f} Cbar={size:.1f}", flush=True)
json.dump(out, open(os.path.join(RES, "validity.json"), "w"), indent=1)
print(f"done ({(time.time()-t0)/60:.1f} min)")
