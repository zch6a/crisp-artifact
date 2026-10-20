"""Reader contamination test: recall on BIRD-Dev vs the schema-abbreviated copy (devabbr), tier-1 scores only.

For every reader with scores on both splits: SRR@k (closure f1k1, k = 5/10/15/20) and the certified Cbar at
eps = 0.05 / 0.02 (protocol A, R random halves).  A reader that memorized BIRD-Dev schemas should lose much more
recall under abbreviation than readers that never saw Dev (the in-house block / distilled readers).
  v2_perturb_eval.py --readers jev_t10,block_t10,block_t10:_s7
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, load_examples, read_scores
from jevschema.certify import calibrate, evaluate_budgets, fixed_family

ap = argparse.ArgumentParser(); ap.add_argument("--readers", default="jev_t10"); ap.add_argument("--R", type=int, default=100)
a = ap.parse_args()
splits = {"dev": load_examples("dev"), "devabbr": load_examples("devabbr")}
rng = np.random.default_rng(0)
out = {}
for spec in a.readers.split(","):
    src, sfx = (spec.split(":") + [""])[:2]
    row = {}
    for sp, exs in splits.items():
        S = read_scores(os.path.join(CACHE, f"{src}_{sp}{sfx}.jsonl"))
        keep = [i for i, e in enumerate(exs) if e.ex_id in S]
        if len(keep) < 0.95 * len(exs):
            print(f"{spec} {sp}: only {len(keep)} scored, skipped"); continue
        ex2 = [exs[i] for i in keep]
        v = View(ex2, [S[e.ex_id] for e in ex2], "f1k1")
        r = {f"srr@{k}": float(np.mean(v.r <= k)) for k in (5, 10, 15, 20)}
        fam = fixed_family(v.r, v.n_cols); n = len(ex2)
        for eps in (0.05, 0.02):
            cb, cv = [], []
            for _ in range(a.R):
                p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
                ks, c = evaluate_budgets(fam, calibrate(fam, cal, eps), te, v.r)
                cv.append(c.mean()); cb.append(np.mean([v.size(i, k) for i, k in zip(te, ks)]))
            r[f"cbar_e{eps}"] = float(np.mean(cb)); r[f"cov_e{eps}"] = float(np.mean(cv))
        r["n"] = n
        row[sp] = r
    out[spec] = row
    if len(row) == 2:
        d, p = row["dev"], row["devabbr"]
        print(f"{spec:16s} " + "  ".join(f"SRR@{k} {d[f'srr@{k}']:.3f}->{p[f'srr@{k}']:.3f}" for k in (5, 10, 20)) +
              f" | cert Cbar e.05 {d['cbar_e0.05']:.1f}->{p['cbar_e0.05']:.1f}  e.02 {d['cbar_e0.02']:.1f}->{p['cbar_e0.02']:.1f}")
json.dump(out, open(os.path.join(RES, "perturb_eval.json"), "w"), indent=1)
