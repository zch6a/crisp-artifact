"""Dev-only certificate check (no holdout, no train annotations, no API).

    python reproduce/certificate_dev.py --eps 0.05

Reproduces the main validity number of the paper on BIRD-Dev with the fixed budget family
and two-fold cross-fitting: expected coverage 0.955 and 20.2 columns at eps=.05.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "code", "scripts"))

import numpy as np
from jevschema.certify import calibrate, covers, fixed_family, kept_at
from jevschema.paths import CACHE
from v2data import View, fused_scores, lexical, load_examples, read_scores
from v2_cascade import tier1_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--tier1", default="jev_t10")
    ap.add_argument("--beta", type=float, default=0.02)
    a = ap.parse_args()
    dev = load_examples("dev")
    bm, em = lexical("dev")
    s0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
    ld = read_scores(os.path.join(CACHE, f"{a.tier1}_dev.jsonl"))
    s1 = tier1_scores(dev, s0, [ld[e.ex_id] for e in dev], a.beta)
    v = View(dev, s1, "f1k1")
    fam = fixed_family(v.r, v.n_cols)
    fold = np.random.default_rng(0).permutation(len(dev)) % 2
    lam = np.zeros(len(dev))
    for f in (0, 1):
        cal = [i for i in range(len(dev)) if fold[i] != f]
        te = [i for i in range(len(dev)) if fold[i] == f]
        L = calibrate(fam, cal, a.eps)
        for i in te:
            lam[i] = L
    kept = [kept_at(dev[i].schema, v.orders[i], fam.budget(i, lam[i]), v.cfg) for i in range(len(dev))]
    cov = float(np.mean([1.0 if covers(dev[i].gold, set(kept[i])) else 0.0 for i in range(len(dev))]))
    size = float(np.mean([len(k) for k in kept]))
    print(json.dumps({"eps": a.eps, "coverage": cov, "columns": size, "k_hat": sorted(set(int(x) for x in lam))}))


if __name__ == "__main__":
    main()
