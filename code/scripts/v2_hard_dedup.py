"""Remedy for the hard benchmark: duplicate-aware ranking (schema de-duplication).

Tables with an identical column-name signature form an equivalence class.  The representative is the
shortest table name in the class (ties: lexicographic); columns of non-representative tables keep their
relative order but are ranked after every representative column, so the closure only expands copies
when the budget is large enough to reach them.  Rule uses no knowledge of the suffixes we generated,
but it does assume base tables have the shortest name - reported as an optional pre-processing step.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, full_tokens, fused_scores, load_examples, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, evaluate_budgets, fixed_family


def dedup(e, s):
    sig = {}
    for t in e.schema.tables:
        sig.setdefault(tuple(sorted(c.name.lower() for c in t.columns)), []).append(t.name)
    rep = {}
    for names in sig.values():
        r = sorted(names, key=lambda x: (len(x), x))[0]
        for nm in names:
            rep[nm] = (nm == r)
    lo = min(s.values()) if s else 0.0
    return {p: (v if rep.get(p[0], True) else v - 10.0 - abs(lo)) for p, v in s.items()}


n = 2000
exs0 = load_examples(f"hard{n}")
J = read_scores(os.path.join(CACHE, f"jev_fk_hard{n}.jsonl"))
exs = [e for e in exs0 if e.ex_id in J]
bm = read_scores(os.path.join(CACHE, f"hard{n}_bm25.jsonl")); em = read_scores(os.path.join(CACHE, f"hard{n}_emb.jsonl"))
f0 = [fused_scores(e, bm[e.ex_id], em[e.ex_id]) for e in exs]
full = full_tokens(exs)
rng = np.random.default_rng(0)
rows = []
for name, sc in (("fused", f0), ("fused+dedup", [dedup(e, s) for e, s in zip(exs, f0)]),
                 ("jev", combine(exs, f0, [J[e.ex_id] for e in exs], 0.02)),
                 ("jev+dedup", [dedup(e, s) for e, s in zip(exs, combine(exs, f0, [J[e.ex_id] for e in exs], 0.02))])):
    for cfg in ("f1k1", "f1p5s15k1"):
        v = View(exs, sc, cfg)
        line = " ".join(f"@{k}:{np.mean(v.r <= k):.3f}/{np.mean([v.size(i, k) for i in range(len(exs))]):.0f}" for k in (10, 20, 35))
        fam = fixed_family(v.r, v.n_cols); cert = []
        for eps in (0.02, 0.05):
            cov, cb, tk = [], [], []
            for rep in range(100):
                p = rng.permutation(len(exs)); cal, te = p[: len(exs) // 2], p[len(exs) // 2:]
                lam = calibrate(fam, cal, eps); ks, cv = evaluate_budgets(fam, lam, te, v.r)
                cov.append(cv.mean()); cb.append(np.mean([v.size(i, k) for i, k in zip(te, ks)]))
                if rep < 3:
                    tk.append(np.mean([v.tokens(i, k) for i, k in zip(te[:150], ks[:150])]))
            cert.append(f"eps{eps}: cov={np.mean(cov):.3f} Cbar={np.mean(cb):.0f} saved={100*(1-np.mean(tk)/full.mean()):.0f}%")
            rows.append({"reader": name, "cfg": cfg, "eps": eps, "cov": float(np.mean(cov)), "cbar": float(np.mean(cb)), "tok": float(np.mean(tk))})
        print(f"{name:12s} {cfg:10s} SRR/Cbar {line} | " + " | ".join(cert), flush=True)
json.dump(rows, open(os.path.join(RES, "hard_dedup.json"), "w"), indent=1)
