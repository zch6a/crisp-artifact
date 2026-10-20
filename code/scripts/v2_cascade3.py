"""Reader cascade: block reader for every question (cheap), pointwise reader only where the block
reader is uncertain.  Router = difficulty model on block-score shape (fitted on BIRD-train holdout);
per-route Mondrian calibration keeps P(cover) >= 1-eps."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_meta, read_scores
from v2_frontier import combine
from v2_cascade import feats
from jevschema.certify import calibrate, evaluate_budgets, fixed_family
from jevschema.difficulty import Difficulty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgB", default="f1p5s15k1")
    ap.add_argument("--cfgP", default="f1k1")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--eps", default="0.01,0.02,0.03,0.05")
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    dev, hold = load_examples("dev"), load_examples("holdout")
    bm, em = lexical("dev"); hbm, hem = lexical("holdout")
    f_d = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
    f_h = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
    S = {}
    for key in ("block_t10", "laya_t10"):
        d = read_scores(os.path.join(CACHE, f"{key}_dev.jsonl")); h = read_scores(os.path.join(CACHE, f"{key}_holdout.jsonl"))
        S[key] = (combine(dev, f_d, [d[e.ex_id] for e in dev], 0.02), combine(hold, f_h, [h[e.ex_id] for e in hold], 0.02))
    mB = read_meta(os.path.join(CACHE, "block_t10_dev.jsonl")); mP = read_meta(os.path.join(CACHE, "laya_t10_dev.jsonl"))
    encB = np.asarray([mB[e.ex_id]["enc_tokens"] for e in dev], float); encP = np.asarray([mP[e.ex_id]["enc_tokens"] for e in dev], float)
    wB = np.asarray([mB[e.ex_id]["wall_s"] for e in dev], float); wP = np.asarray([mP[e.ex_id]["wall_s"] for e in dev], float)
    vB, vP = View(dev, S["block_t10"][0], a.cfgB), View(dev, S["laya_t10"][0], a.cfgP)
    vBh = View(hold, S["block_t10"][1], a.cfgB)
    diff = Difficulty().fit(feats(vBh), np.log(vBh.r))
    sh, sd = diff.predict(feats(vBh))[0], diff.predict(feats(vB))[0]
    gain_d = np.log(vB.r) - np.log(vP.r)
    print(f"router R2(holdout)={diff.r2:.3f}  corr(router, true gain) on dev = {np.corrcoef(sd, gain_d)[0,1]:.3f}", flush=True)
    fB, fP = fixed_family(vB.r, vB.n_cols), fixed_family(vP.r, vP.n_cols)
    rng = np.random.default_rng(0); rows = []
    for frac in [float(x) for x in a.fracs.split(",")]:
        tau = np.inf if frac <= 0 else (-np.inf if frac >= 1 else float(np.quantile(sh, 1 - frac)))
        R = sd > tau
        for eps in [float(x) for x in a.eps.split(",")]:
            cov, cb = [], []
            for rep in range(a.reps):
                p = rng.permutation(len(dev)); cal, te = p[: len(dev) // 2], p[len(dev) // 2:]
                cv_all, sz = [], []
                for route, fam, v in ((False, fB, vB), (True, fP, vP)):
                    cc = [i for i in cal if R[i] == route]; tt = [i for i in te if R[i] == route]
                    if not tt:
                        continue
                    lam = calibrate(fam, cc, eps); ks, cv = evaluate_budgets(fam, lam, tt, v.r)
                    cv_all += list(cv); sz += [v.size(i, k) for i, k in zip(tt, ks)]
                cov.append(np.mean(cv_all)); cb.append(np.mean(sz))
            row = {"frac": frac, "esc": float(R.mean()), "eps": eps, "cov": float(np.mean(cov)), "cbar": float(np.mean(cb)),
                   "enc": float(np.mean(encB + encP * R)), "wall": float(np.mean(wB + wP * R))}
            rows.append(row)
            print(f"esc={row['esc']:.2f} eps={eps:.2f} cov={row['cov']:.4f} Cbar={row['cbar']:.1f} enc/q={row['enc']:.0f} wall/q={row['wall']:.2f}s", flush=True)
    json.dump(rows, open(os.path.join(RES, "cascade_reader.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
