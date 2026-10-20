"""Certified cascade v2: per-tier family/closure + benefit-aware routers (tier-0 features only).

Routers (all fitted on BIRD-train holdout):
  budget  : route if tier-0 predicted log-depth is large (difficulty model)
  gain    : ridge-predicted log r0 - log r1 (how much System-One reading helps)
  gaincost: predicted gain per unit tier-1 encoder cost (~ gated columns)
Every route is calibrated separately (Mondrian), so coverage >= 1-eps holds for any router.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, full_tokens, fused_scores, lexical, load_examples, read_meta, read_scores
from v2_cascade import tier1_scores, feats
from jevschema.certify import calibrate, evaluate_budgets, fixed_family, mass_family, thresh_family
from jevschema.difficulty import Difficulty


def family(view, name):
    ss = [view.sorted_scores(i) for i in range(len(view.exs))]
    if name == "thresh":
        return thresh_family(view.r, view.n_cols, ss)
    if name == "mass":
        return mass_family(view.r, view.n_cols, [view.cum_mass(i) for i in range(len(view.exs))])
    return fixed_family(view.r, view.n_cols)


def ridge(X, y, lam=1.0):
    m, s = X.mean(0), X.std(0); s = np.where(s < 1e-6, 1, s)
    Z = np.hstack([(X - m) / s, np.ones((len(X), 1))])
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return lambda Xn: np.hstack([(Xn - m) / s, np.ones((len(Xn), 1))]) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", default="laya_t10")
    ap.add_argument("--cfg0", default="f1p5s15k1")
    ap.add_argument("--cfg1", default="f1k1")
    ap.add_argument("--fam0", default="fixed")
    ap.add_argument("--fam1", default="thresh")
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--routers", default="budget,gain,gaincost")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--eps", default="0.01,0.02,0.03,0.05")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    dev, hold = load_examples("dev"), load_examples("holdout")
    bm, em = lexical("dev"); hbm, hem = lexical("holdout")
    s0d = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
    s0h = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
    ld = read_scores(os.path.join(CACHE, f"{a.tier1}_dev.jsonl")); lh = read_scores(os.path.join(CACHE, f"{a.tier1}_holdout.jsonl"))
    md = read_meta(os.path.join(CACHE, f"{a.tier1}_dev.jsonl")); mh = read_meta(os.path.join(CACHE, f"{a.tier1}_holdout.jsonl"))
    s1d = tier1_scores(dev, s0d, [ld[e.ex_id] for e in dev], a.beta)
    s1h = tier1_scores(hold, s0h, [lh[e.ex_id] for e in hold], a.beta)
    v0d, v1d = View(dev, s0d, a.cfg0), View(dev, s1d, a.cfg1)
    v0h, v1h = View(hold, s0h, a.cfg0), View(hold, s1h, a.cfg1)
    enc_d = np.asarray([float(md[e.ex_id]["enc_tokens"] or 0) for e in dev])
    wall_d = np.asarray([float(md[e.ex_id]["wall_s"] or 0) for e in dev])
    ncol_d = np.asarray([len(e.schema.columns()) for e in dev], float)
    ncol_h = np.asarray([len(e.schema.columns()) for e in hold], float)
    Xh, Xd = feats(v0h), feats(v0d)
    diff = Difficulty().fit(Xh, np.log(v0h.r))
    gain_h = np.log(v0h.r) - np.log(v1h.r)
    g = ridge(Xh, gain_h)
    enc_h = np.asarray([float(mh[e.ex_id]["enc_tokens"] or 0) for e in hold])
    c = ridge(np.column_stack([Xh, np.log1p(ncol_h)]), np.log1p(enc_h))
    stat = {}
    for name, fn in (("budget", lambda X, n: diff.predict(X)[0]),
                     ("gain", lambda X, n: g(X)),
                     ("gaincost", lambda X, n: g(X) - 0.5 * c(np.column_stack([X, np.log1p(n)]))),):
        stat[name] = (fn(Xh, ncol_h), fn(Xd, ncol_d))
    corr = np.corrcoef(stat["gain"][1], np.log(v0d.r) - np.log(v1d.r))[0, 1]
    print(f"dev n={len(dev)} mean log r0={np.mean(np.log(v0d.r)):.3f} r1={np.mean(np.log(v1d.r)):.3f} "
          f"gain-pred corr(dev)={corr:.3f} enc/q={enc_d.mean():.0f} wall/q={wall_d.mean():.2f}s ({time.time()-t0:.0f}s)", flush=True)
    f0, f1 = family(v0d, a.fam0), family(v1d, a.fam1)
    full_tok = full_tokens(dev)
    rng = np.random.default_rng(0)
    rows = []
    for router in a.routers.split(","):
        sh, sd = stat[router]
        for frac in [float(x) for x in a.fracs.split(",")]:
            tau = np.inf if frac <= 0 else (-np.inf if frac >= 1 else float(np.quantile(sh, 1 - frac)))
            R = sd > tau
            for eps in [float(x) for x in a.eps.split(",")]:
                cov, cb, tk = [], [], []
                for rep in range(a.reps):
                    perm = rng.permutation(len(dev)); cal, te = perm[: len(dev) // 2], perm[len(dev) // 2:]
                    cv_all, sz_all, tk_all = [], [], []
                    for route, fam, view in ((False, f0, v0d), (True, f1, v1d)):
                        cc = [i for i in cal if R[i] == route]; tt = [i for i in te if R[i] == route]
                        if not tt:
                            continue
                        lam = calibrate(fam, cc, eps)
                        ks, cv = evaluate_budgets(fam, lam, tt, view.r)
                        cv_all += list(cv); sz_all += [view.size(i, k) for i, k in zip(tt, ks)]
                        if rep < 8:
                            tk_all += [view.tokens(i, k) for i, k in zip(tt, ks)]
                    cov.append(np.mean(cv_all)); cb.append(np.mean(sz_all))
                    if tk_all:
                        tk.append(np.mean(tk_all))
                row = {"router": router, "frac": frac, "esc": float(R.mean()), "eps": eps, "cov": float(np.mean(cov)),
                       "cov_sd": float(np.std(cov)), "cbar": float(np.mean(cb)), "gen_tok": float(np.mean(tk)),
                       "enc_tok": float(np.mean(enc_d * R)), "wall": float(np.mean(wall_d * R))}
                rows.append(row)
                print(f"{router:8s} esc={row['esc']:.2f} eps={eps:.2f} cov={row['cov']:.4f}+-{row['cov_sd']:.4f} "
                      f"Cbar={row['cbar']:.1f} gen_tok={row['gen_tok']:.0f} enc/q={row['enc_tok']:.0f} wall/q={row['wall']:.2f}", flush=True)
    out = os.path.join(RES, f"cascade2_{a.tier1}_{a.fam0}-{a.fam1}_{a.cfg0}-{a.cfg1}{a.tag}.json")
    json.dump({"args": vars(a), "full_tok": float(full_tok.mean()), "rows": rows}, open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
