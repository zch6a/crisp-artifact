"""Spider 2.0-lite (real enterprise schemas): SRR and certified schema size, Jev vs lexical BM25 reader.

Questions: the 242 instances with public gold SQL and parsed requirements.  Schema: canonicalized (identical-signature
shards merged) -- the de-duplication step; raw column counts reported for reference.  No FKs in Spider2, so the
closure is PK/none ("f0").  Certification: split conformal, fixed family, R random halves (n is small).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, load_examples, read_scores
from jevschema.certify import calibrate, evaluate_budgets, fixed_family
from jevschema.summary import bm25_scores

R = 200
exs = load_examples("spider2"); raw = {e.ex_id: len(e.schema.columns()) for e in load_examples("spider2raw")}
J = read_scores(os.path.join(CACHE, (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SP2_JEV", "jev_t10_spider2.jsonl"))))
exs = [e for e in exs if e.ex_id in J]
print(f"{len(exs)} questions with Jev scores; mean cols raw={np.mean([raw[e.ex_id] for e in exs]):.0f} "
      f"canonical={np.mean([len(e.schema.columns()) for e in exs]):.0f}")


def bm25(e):
    pairs = e.schema.columns()
    docs = [f"{t} {t.replace('_', ' ')} {c} {c.replace('_', ' ')}" for t, c in pairs]
    return dict(zip(pairs, bm25_scores(e.question, docs)))


views = {"jev": View(exs, [J[e.ex_id] for e in exs], "f0"), "bm25": View(exs, [bm25(e) for e in exs], "f0")}
rng = np.random.default_rng(0); n = len(exs)
splits = [rng.permutation(n) for _ in range(R)]
kinds = ("bigquery", "snowflake", "sqlite")
for name, v in views.items():
    srr = "  ".join(f"@{k}:{np.mean(v.r <= k):.3f}" for k in (10, 20, 50, 100))
    fam = fixed_family(v.r, v.n_cols)
    cert = []
    for eps in (0.05, 0.10):
        cov, cb, frac = [], [], []
        for p in splits:
            cal, te = p[: n // 2], p[n // 2:]
            ks, c = evaluate_budgets(fam, calibrate(fam, cal, eps), te, v.r)
            sz = [v.size(i, k) for i, k in zip(te, ks)]
            cov.append(c.mean()); cb.append(np.mean(sz)); frac.append(np.mean([s / raw[exs[i].ex_id] for i, s in zip(te, sz)]))
        cert.append(f"eps{eps}: cov={np.mean(cov):.3f} Cbar={np.mean(cb):.1f} ({100*np.mean(frac):.1f}% of raw cols)")
    print(f"{name:5s} SRR {srr} | " + " | ".join(cert))
    for kd in kinds:
        ii = [i for i, e in enumerate(exs) if e.difficulty == kd]
        print(f"      {kd:9s} n={len(ii):3d} SRR@20={np.mean(v.r[ii] <= 20):.3f} SRR@50={np.mean(v.r[ii] <= 50):.3f} "
              f"median raw cols={int(np.median([raw[exs[i].ex_id] for i in ii]))}")
