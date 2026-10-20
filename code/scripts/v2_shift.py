"""Zero-dev-label certification under covariate shift (protocol C remedies).
  plain    : split conformal on BIRD-train holdout
  weighted : weighted conformal (Tibshirani et al. 2019), w = p(dev|x)/p(hold|x) from a domain
             classifier on unlabeled features (no dev labels); test weight bounded by max weight
  sizebin  : Mondrian by log(#columns) bins (calibrate inside the bin of the test DB)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from v2_cascade import feats
from jevschema.certify import conformal_quantile, fixed_family, evaluate_budgets


def wquant(s, w, w_test, eps):
    o = np.argsort(s); s, w = s[o], w[o]
    c = np.cumsum(w) / (w.sum() + w_test)
    j = np.searchsorted(c, 1 - eps, side="left")
    return float(s[j]) if j < len(s) else float("inf")


dev, hold = load_examples("dev"), load_examples("holdout")
bm, em = lexical("dev"); hbm, hem = lexical("holdout")
f_d = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
f_h = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
cfgs = {"fused": (f_d, f_h, "f1p5s15k1")}
for key, cfg in (("laya_t10", "f1k1"), ("block_t10", "f1p5s15k1")):
    d = read_scores(os.path.join(CACHE, f"{key}_dev.jsonl")); h = read_scores(os.path.join(CACHE, f"{key}_holdout.jsonl"))
    cfgs[key] = (combine(dev, f_d, [d[e.ex_id] for e in dev], 0.02), combine(hold, f_h, [h[e.ex_id] for e in hold], 0.02), cfg)
out = []
for name, (sd, sh, cfg) in cfgs.items():
    vd, vh = View(dev, sd, cfg), View(hold, sh, cfg)
    Xd, Xh = feats(vd), feats(vh)
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(np.vstack([Xh, Xd]), np.r_[np.zeros(len(Xh)), np.ones(len(Xd))])
    ph = clf.predict_proba(Xh)[:, 1]
    w = ph / np.maximum(1e-6, 1 - ph) * (len(Xh) / len(Xd))
    w = np.minimum(w, np.quantile(w, 0.99))
    ess = w.sum() ** 2 / (w ** 2).sum()
    fam_h, fam_d = fixed_family(vh.r, vh.n_cols), fixed_family(vd.r, vd.n_cols)
    sH = np.asarray([fam_h.nonconf(i) for i in range(len(hold))], float)
    nb_h = np.digitize(np.log(vh.n_cols), [3.5, 4.0, 4.5]); nb_d = np.digitize(np.log(vd.n_cols), [3.5, 4.0, 4.5])
    for eps in (0.02, 0.05, 0.10):
        res = {}
        lam = conformal_quantile(sH, eps)
        ks, cv = evaluate_budgets(fam_d, lam, range(len(dev)), vd.r); res["plain"] = (cv.mean(), np.mean([vd.size(i, k) for i, k in enumerate(ks)]))
        lam = wquant(sH, w, w.max(), eps)
        ks, cv = evaluate_budgets(fam_d, lam, range(len(dev)), vd.r); res["weighted"] = (cv.mean(), np.mean([vd.size(i, k) for i, k in enumerate(ks)]))
        covs, sz = [], []
        for b in np.unique(nb_d):
            idx = np.where(nb_d == b)[0]; hb = sH[nb_h == b]
            lam = conformal_quantile(hb, eps) if len(hb) else float("inf")
            ks, cv = evaluate_budgets(fam_d, lam, idx, vd.r); covs += list(cv); sz += [vd.size(i, k) for i, k in zip(idx, ks)]
        res["sizebin"] = (np.mean(covs), np.mean(sz))
        print(f"{name:9s} eps={eps:.2f} target={1-eps:.2f} | " + " | ".join(f"{k}: cov={c:.4f} Cbar={s:.1f}" for k, (c, s) in res.items()) + f" | ESS={ess:.0f}", flush=True)
        out.append({"scorer": name, "eps": eps, **{k: {"cov": float(c), "cbar": float(s)} for k, (c, s) in res.items()}})
json.dump(out, open(os.path.join(RES, "shift_calib.json"), "w"), indent=1)
