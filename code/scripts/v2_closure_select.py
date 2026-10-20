"""Closure configuration selected on BIRD-train holdout only (certified efficiency), then evaluated once on dev.
Removes any 'closure hyper-parameters tuned on dev' concern."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, evaluate_budgets, fixed_family

CFGS = ["f1", "f1k1", "f1p3k1", "f1p5k1", "f1p8k1", "f1p0s15k1", "f1p5s15k1", "f1p8s15k1"]
rng = np.random.default_rng(0)


def cert(v, eps, reps=60):
    fam = fixed_family(v.r, v.n_cols); cov, cb = [], []
    n = len(v.exs)
    for _ in range(reps):
        p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
        lam = calibrate(fam, cal, eps); ks, cv = evaluate_budgets(fam, lam, te, v.r)
        cov.append(cv.mean()); cb.append(np.mean([v.size(i, k) for i, k in zip(te, ks)]))
    return float(np.mean(cov)), float(np.mean(cb))


dev, hold = load_examples("dev"), load_examples("holdout")
bm, em = lexical("dev"); hbm, hem = lexical("holdout")
fd = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
fh = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
readers = {"fused": (fd, fh, list(range(len(hold))))}
for key, src in (("laya", "laya_t10"), ("block", "block_t10"), ("jev", "jev_t10")):
    D = read_scores(os.path.join(CACHE, f"{src}_dev.jsonl")); H = read_scores(os.path.join(CACHE, f"{src}_holdout.jsonl"))
    idx = [i for i, e in enumerate(hold) if e.ex_id in H]
    readers[key] = (combine(dev, fd, [D[e.ex_id] for e in dev], 0.02),
                    combine([hold[i] for i in idx], [fh[i] for i in idx], [H[hold[i].ex_id] for i in idx], 0.02), idx)
out = []
for name, (sd, sh, idx) in readers.items():
    hh = [hold[i] for i in idx]
    for eps in (0.02, 0.05):
        best = None
        for cfg in CFGS:
            c, b = cert(View(hh, sh, cfg), eps)
            if best is None or b < best[1]:
                best = (cfg, b)
        cov, cb = cert(View(dev, sd, best[0]), eps, reps=200)
        out.append({"reader": name, "eps": eps, "cfg_selected_on_holdout": best[0], "dev_cov": cov, "dev_cbar": cb})
        print(f"{name:6s} eps={eps:.2f} holdout-selected cfg={best[0]:10s} -> dev certified cov={cov:.4f} Cbar={cb:.1f}", flush=True)
json.dump(out, open(os.path.join(RES, "closure_selected_on_holdout.json"), "w"), indent=1)
