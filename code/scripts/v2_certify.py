"""Certified budgets on BIRD-Dev: realized coverage vs target and efficiency per family.

Protocols
  A  split-conformal on dev: random 50/50 calibration/test, R repeats (exchangeable -> exact)
  B  leave-one-database-out: calibrate on 10 dev DBs, test on the held-out DB
  C  zero-dev-label transfer: calibrate on BIRD-train holdout, test on all of dev
Difficulty models (normalized family) are always fitted on BIRD-train holdout, never on dev.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, full_tokens, fused_scores, lexical, load_examples, read_scores
from jevschema.certify import calibrate, evaluate_budgets, fixed_family, hybrid_family, mass_family, normalized_family, thresh_family
from jevschema.difficulty import Difficulty, shape_features


def feats_of(view):
    X = []
    for i, e in enumerate(view.exs):
        X.append(shape_features(view.sorted_scores(i), len(e.schema.columns()), len(e.schema.tables),
                                len(e.question.split()), bool(e.evidence)))
    return np.asarray(X)


def families(view, diff):
    fams = {"fixed": fixed_family(view.r, view.n_cols),
            "mass": mass_family(view.r, view.n_cols, [view.cum_mass(i) for i in range(len(view.exs))])}
    ss = [view.sorted_scores(i) for i in range(len(view.exs))]
    fams["thresh"] = thresh_family(view.r, view.n_cols, ss)
    fams["hybrid8"] = hybrid_family(view.r, view.n_cols, ss, 8)
    if diff is not None:
        mu, sig = diff.predict(feats_of(view))
        fams["normalized"] = normalized_family(view.r, view.n_cols, mu, sig)
    return fams


def summarize(view, fam, lam, idx, full_tok, with_tokens):
    ks, cov = evaluate_budgets(fam, lam, idx, view.r)
    size = np.asarray([view.size(i, k) for i, k in zip(idx, ks)], dtype=float)
    out = {"cov": float(cov.mean()), "cbar": float(size.mean()), "k": float(ks.mean())}
    if with_tokens:
        tok = np.asarray([view.tokens(i, k) for i, k in zip(idx, ks)], dtype=float)
        out["tok"] = float(tok.mean())
        out["tok_saving"] = float(1 - tok.sum() / full_tok[idx].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorers", default="fused,laya6")
    ap.add_argument("--cfgs", default="f1k1,f1p3s15k1,f1p5s15k1,f1p8s15k1")
    ap.add_argument("--eps", default="0.01,0.02,0.03,0.05,0.10")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--out", default=os.path.join(RES, "certify_dev.json"))
    ap.add_argument("--families", default="fixed,mass,thresh,hybrid8,normalized")
    a = ap.parse_args()
    eps_list = [float(x) for x in a.eps.split(",")]
    t0 = time.time()
    dev = load_examples("dev")
    hold = load_examples("holdout")
    bm, em = lexical("dev")
    hbm, hem = lexical("holdout")
    sc_dev = {"fused": [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]}
    sc_hold = {"fused": [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]}
    hold_keep = {}
    tier1 = {"laya6": "bird_layapareto_laya_ft_bird_vask_L1024_t6.jsonl", "laya10": "laya_t10_dev.jsonl"}
    for name, fn in tier1.items():
        if name in a.scorers.split(","):
            s = read_scores(os.path.join(CACHE, fn))
            sc_dev[name] = [s.get(e.ex_id, {}) for e in dev]
            hs = read_scores(os.path.join(CACHE, "laya_t10_holdout.jsonl"))
            if len(hs) >= 500:
                sc_hold[name] = [hs.get(e.ex_id, {}) for e in hold]
    for key, src, sfx in (("laya10f", "laya_t10", ""), ("block10f", "block_t10", ""), ("jev10f", "jev_t10", ""), ("dist10f", "block_t10", "_s7")):
        if key not in a.scorers.split(","):
            continue
        from v2_frontier import combine
        ld = read_scores(os.path.join(CACHE, f"{src}_dev{sfx}.jsonl"))
        lh = read_scores(os.path.join(CACHE, f"{src}_holdout{sfx}.jsonl"))
        sc_dev[key] = combine(dev, sc_dev["fused"], [ld.get(e.ex_id, {}) for e in dev], a.beta)
        if len(lh) >= 500:
            sc_hold[key] = combine(hold, sc_hold["fused"], [lh.get(e.ex_id, {}) for e in hold], a.beta)
            hold_keep[key] = [i for i, e in enumerate(hold) if e.ex_id in lh]
    full_tok = full_tokens(dev)
    dbs = sorted({e.db_id for e in dev})
    db_of = np.asarray([dbs.index(e.db_id) for e in dev])
    print(f"loaded dev={len(dev)} holdout={len(hold)} in {time.time()-t0:.0f}s; full-schema tokens {full_tok.mean():.1f}", flush=True)
    rng = np.random.default_rng(0)
    results = []
    for scorer in a.scorers.split(","):
        if scorer not in sc_dev:
            continue
        for cfg in a.cfgs.split(","):
            v = View(dev, sc_dev[scorer], cfg)
            diff, vh = None, None
            if scorer in sc_hold:
                kh = hold_keep.get(scorer)
                hh = [hold[i] for i in kh] if kh else hold
                vh = View(hh, [sc_hold[scorer][i] for i in kh] if kh else sc_hold[scorer], cfg)
                Xh = feats_of(vh)
                diff = Difficulty().fit(Xh, np.log(np.maximum(1, vh.r)))
            fams = families(v, diff)
            fams_h = families(vh, diff) if vh is not None else {}
            print(f"\n== {scorer} {cfg}  mean r={v.r.mean():.1f} median={np.median(v.r):.0f}"
                  f"  diff R2(holdout)={getattr(diff, 'r2', float('nan')):.3f}", flush=True)
            for fname, fam in fams.items():
                if fname not in a.families.split(","):
                    continue
                for eps in eps_list:
                    # protocol A
                    covs, cbs, toks = [], [], []
                    for rep in range(a.reps):
                        perm = rng.permutation(len(dev))
                        cal, te = perm[: len(dev) // 2], perm[len(dev) // 2:]
                        lam = calibrate(fam, cal, eps)
                        s = summarize(v, fam, lam, te, full_tok, with_tokens=(rep < 20))
                        covs.append(s["cov"]); cbs.append(s["cbar"])
                        if "tok" in s:
                            toks.append(s["tok"])
                    A = {"cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)), "cov_p05": float(np.quantile(covs, 0.05)),
                         "cbar": float(np.mean(cbs)), "tok": float(np.mean(toks))}
                    # protocol B (LODO)
                    covB, sizeB, per_db = [], [], {}
                    for d in range(len(dbs)):
                        cal = np.where(db_of != d)[0]; te = np.where(db_of == d)[0]
                        lam = calibrate(fam, cal, eps)
                        s = summarize(v, fam, lam, te, full_tok, with_tokens=False)
                        per_db[dbs[d]] = round(s["cov"], 4)
                        covB += [s["cov"]] * len(te); sizeB += [s["cbar"]] * len(te)
                    B = {"cov": float(np.mean(covB)), "cbar": float(np.mean(sizeB)),
                         "worst_db": min(per_db.items(), key=lambda kv: kv[1]), "per_db": per_db}
                    # protocol C (holdout calibration -> dev)
                    C = None
                    if fname in fams_h:
                        lam = calibrate(fams_h[fname], range(len(vh.exs)), eps)
                        C = summarize(v, fam, lam, np.arange(len(dev)), full_tok, with_tokens=True)
                    row = {"scorer": scorer, "cfg": cfg, "family": fname, "eps": eps, "A": A, "B": B, "C": C}
                    results.append(row)
                    cs = f"C cov={C['cov']:.4f} Cbar={C['cbar']:.1f}" if C else "C -"
                    print(f"  {fname:10s} eps={eps:.2f} | A cov={A['cov']:.4f}+-{A['cov_sd']:.4f} Cbar={A['cbar']:.1f} tok={A['tok']:.0f}"
                          f" | B cov={B['cov']:.4f} Cbar={B['cbar']:.1f} worst={B['worst_db'][0]}:{B['worst_db'][1]:.3f} | {cs}", flush=True)
    json.dump({"n_dev": len(dev), "n_holdout": len(hold), "full_tokens": float(full_tok.mean()), "rows": results},
              open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
