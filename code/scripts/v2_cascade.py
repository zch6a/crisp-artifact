"""Risk-routed certified cascade: tier-0 (CPU lexical) -> tier-1 (System-One laya) on demand.

Route R(x) = 1  iff  tier-0 predicted budget exp(mu0 + z sig0) > tau   (tier-0 features only)
Each route is calibrated separately (Mondrian split conformal) -> P(cover) >= 1-eps overall.
beta (tier-1 fusion weight) and tau candidates are chosen on BIRD-train holdout only.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, full_tokens, fused_scores, lexical, load_examples, read_meta, read_scores
from jevschema.certify import calibrate, evaluate_budgets, fixed_family, normalized_family
from jevschema.difficulty import Difficulty, shape_features


def tier0_scores(name, split, exs):
    if name == "fused":
        bm, em = lexical(split)
        return [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in exs]
    s = read_scores(os.path.join(CACHE, f"rank_{name}_{split}.jsonl"))
    return [s.get(e.ex_id, {}) for e in exs]


def mm(d, pairs):
    v = np.asarray([d.get(p, 0.0) for p in pairs], dtype=float)
    return (v - v.min()) / max(1e-9, v.max() - v.min()) if len(v) else v


def tier1_scores(exs, t0, laya, beta):
    out = []
    for e, s0, s1 in zip(exs, t0, laya):
        pairs = e.schema.columns()
        n0 = mm(s0, pairs)
        v = []
        for p, x0 in zip(pairs, n0):
            base = s1[p] if p in s1 else -1.0          # ungated columns go after gated ones
            v.append(base + beta * x0)
        out.append(dict(zip(pairs, v)))
    return out


def feats(view):
    return np.asarray([shape_features(view.sorted_scores(i), len(e.schema.columns()), len(e.schema.tables),
                                      len(e.question.split()), bool(e.evidence)) for i, e in enumerate(view.exs)])


def fam_for(view, diff):
    if diff is None:
        return fixed_family(view.r, view.n_cols)
    mu, sig = diff.predict(feats(view))
    return normalized_family(view.r, view.n_cols, mu, sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0", default="fused")
    ap.add_argument("--tier1", default="laya_t10")
    ap.add_argument("--cfg", default="f1p5s15k1")
    ap.add_argument("--betas", default="0,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--eps", default="0.01,0.02,0.03,0.05,0.10")
    ap.add_argument("--route-fracs", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0")
    ap.add_argument("--z", type=float, default=1.645)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    dev, hold = load_examples("dev"), load_examples("holdout")
    s0d, s0h = tier0_scores(a.tier0, "dev", dev), tier0_scores(a.tier0, "holdout", hold)
    ld = read_scores(os.path.join(CACHE, f"{a.tier1}_dev.jsonl"))
    lh = read_scores(os.path.join(CACHE, f"{a.tier1}_holdout.jsonl"))
    md = read_meta(os.path.join(CACHE, f"{a.tier1}_dev.jsonl"))
    keep_d = [i for i, e in enumerate(dev) if e.ex_id in ld]
    keep_h = [i for i, e in enumerate(hold) if e.ex_id in lh]
    print(f"tier1 coverage dev {len(keep_d)}/{len(dev)} holdout {len(keep_h)}/{len(hold)}", flush=True)
    dev = [dev[i] for i in keep_d]; s0d = [s0d[i] for i in keep_d]
    hold = [hold[i] for i in keep_h]; s0h = [s0h[i] for i in keep_h]
    l1d = [ld[e.ex_id] for e in dev]; l1h = [lh[e.ex_id] for e in hold]
    enc = np.asarray([float(md.get(e.ex_id, {}).get("enc_tokens") or 0) for e in dev])
    wall = np.asarray([float(md.get(e.ex_id, {}).get("wall_s") or 0) for e in dev])
    # beta on holdout
    best_b, best_v = 0.0, 1e9
    for b in [float(x) for x in a.betas.split(",")]:
        vh = View(hold, tier1_scores(hold, s0h, l1h, b), a.cfg)
        m = float(np.mean(np.log(vh.r)))
        print(f"  beta={b:<5} holdout mean_log_r={m:.4f} srr@15={np.mean(vh.r <= 15):.4f}", flush=True)
        if m < best_v:
            best_b, best_v = b, m
    print(f"beta*={best_b}", flush=True)
    v0h, v0d = View(hold, s0h, a.cfg), View(dev, s0d, a.cfg)
    v1h = View(hold, tier1_scores(hold, s0h, l1h, best_b), a.cfg)
    v1d = View(dev, tier1_scores(dev, s0d, l1d, best_b), a.cfg)
    d0 = Difficulty().fit(feats(v0h), np.log(v0h.r))
    d1 = Difficulty().fit(feats(v1h), np.log(v1h.r))
    f0, f1 = fam_for(v0d, d0), fam_for(v1d, d1)
    mu0h, sg0h = d0.predict(feats(v0h)); mu0d, sg0d = d0.predict(feats(v0d))
    b0h, b0d = np.exp(mu0h + a.z * sg0h), np.exp(mu0d + a.z * sg0d)
    full_tok = full_tokens(dev)
    print(f"dev n={len(dev)} tier0 mean_log_r={np.mean(np.log(v0d.r)):.3f} tier1={np.mean(np.log(v1d.r)):.3f} "
          f"R2 d0={d0.r2:.3f} d1={d1.r2:.3f}  enc_tok/q={enc.mean():.0f} wall/q={wall.mean():.2f}s  ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(0)
    rows = []
    for frac in [float(x) for x in a.route_fracs.split(",")]:
        tau = np.inf if frac <= 0 else (-np.inf if frac >= 1 else float(np.quantile(b0h, 1 - frac)))
        R = b0d > tau
        for eps in [float(x) for x in a.eps.split(",")]:
            cov, cb, tk, esc, et = [], [], [], [], []
            for rep in range(a.reps):
                perm = rng.permutation(len(dev)); cal, te = perm[: len(dev) // 2], perm[len(dev) // 2:]
                ks_all, cov_all, size_all, tok_all = [], [], [], []
                for route, fam, view in ((False, f0, v0d), (True, f1, v1d)):
                    c = [i for i in cal if R[i] == route]; t = [i for i in te if R[i] == route]
                    if not t:
                        continue
                    lam = calibrate(fam, c, eps)
                    ks, cv = evaluate_budgets(fam, lam, t, view.r)
                    cov_all += list(cv); size_all += [view.size(i, k) for i, k in zip(t, ks)]
                    if rep < 10:
                        tok_all += [view.tokens(i, k) for i, k in zip(t, ks)]
                cov.append(np.mean(cov_all)); cb.append(np.mean(size_all))
                if tok_all:
                    tk.append(np.mean(tok_all))
                esc.append(float(np.mean(R[te]))); et.append(float(np.mean(enc[te] * R[te])))
            row = {"route_frac_target": frac, "eps": eps, "esc": float(np.mean(esc)), "cov": float(np.mean(cov)),
                   "cov_sd": float(np.std(cov)), "cbar": float(np.mean(cb)), "gen_tok": float(np.mean(tk)),
                   "enc_tok": float(np.mean(et)), "full_tok": float(full_tok.mean())}
            rows.append(row)
            print(f"route~{frac:.1f} esc={row['esc']:.2f} eps={eps:.2f} cov={row['cov']:.4f}+-{row['cov_sd']:.4f} "
                  f"Cbar={row['cbar']:.1f} gen_tok={row['gen_tok']:.0f} enc_tok/q={row['enc_tok']:.0f}", flush=True)
    json.dump({"tier0": a.tier0, "tier1": a.tier1, "cfg": a.cfg, "beta": best_b, "n_dev": len(dev),
               "rows": rows}, open(os.path.join(RES, f"cascade_{a.tier0}_{a.tier1}_{a.cfg}.json"), "w"), indent=1)
    print(f"saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
