"""Spider 2.0-lite under the revised shard canonicalization (same column signature AND same name stem), lifted coverage.

Coverage is computed on the canonical schema with requirements mapped to their classes, which equals lifted coverage
on the raw schema (Proposition 3).  Size is reported three ways: canonical columns kept; physical columns made
available after lifting (a kept representative column counts once per member table); and prompt items (canonical
columns + one name per additional member table, which is what the prompt shows).
Also exports the calibrated budget k_hat (fixed family, all instances with public gold except the local/SQLite ones,
which are the test set of the Spider2-SQLite EX experiment) -> results/spider2_shard_eval.json
  python v2_spider2_shard_eval.py [jev cache file]
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, load_examples, read_scores
from jevschema.certify import calibrate, evaluate_budgets, fixed_family, kept_at
from jevschema.summary import bm25_scores

R = 200
CFG = "f1k1"
exs = load_examples("spider2shard"); raw = {e.ex_id: len(e.schema.columns()) for e in load_examples("spider2raw")}
J = read_scores(os.path.join(CACHE, sys.argv[1] if len(sys.argv) > 1 else "jev_t10_spider2shard.jsonl"))
exs = [e for e in exs if e.ex_id in J]
canon = {e.ex_id: len(e.schema.columns()) for e in exs}
print(f"{len(exs)} questions with Jev scores; mean cols raw={np.mean([raw[e.ex_id] for e in exs]):.0f} "
      f"canonical={np.mean(list(canon.values())):.0f}")


def bm25(e):
    pairs = e.schema.columns()
    docs = [f"{t} {t.replace('_', ' ')} {c} {c.replace('_', ' ')}" for t, c in pairs]
    return dict(zip(pairs, bm25_scores(e.question, docs)))


_memo = {}


def sizes(v, i, k):
    key = (id(v), i, int(k))
    if key not in _memo:
        _memo[key] = _sizes(v, i, k)
    return _memo[key]


def _sizes(v, i, k):
    e = v.exs[i]; kept = kept_at(e.schema, v.orders[i], k, v.cfg)
    mem = e.shard_members
    lifted = sum(len(mem.get(t, [t])) for t, _ in kept)
    tabs = {t for t, _ in kept}
    prompt = len(kept) + sum(len(mem.get(t, [t])) - 1 for t in tabs)
    return len(kept), lifted, prompt


out = {"n": len(exs), "raw_cols": float(np.mean([raw[e.ex_id] for e in exs])), "canonical_cols": float(np.mean(list(canon.values())))}
views = {"jev": View(exs, [J[e.ex_id] for e in exs], CFG), "bm25": View(exs, [bm25(e) for e in exs], CFG)}
rng = np.random.default_rng(0); n = len(exs)
splits = [rng.permutation(n) for _ in range(R)]
for name, v in views.items():
    fam = fixed_family(v.r, v.n_cols)
    res = {"srr": {k: float(np.mean(v.r <= k)) for k in (10, 20, 50, 100)}}
    for eps in (0.05, 0.10):
        cov, sc, sl, sp, fr = [], [], [], [], []
        for p in splits:
            cal, te = p[: n // 2], p[n // 2:]
            ks, c = evaluate_budgets(fam, calibrate(fam, cal, eps), te, v.r)
            S = [sizes(v, i, k) for i, k in zip(te, ks)]
            cov.append(c.mean()); sc.append(np.mean([s[0] for s in S])); sl.append(np.mean([s[1] for s in S]))
            sp.append(np.mean([s[2] for s in S])); fr.append(np.mean([s[1] / raw[v.exs[i].ex_id] for i, s in zip(te, S)]))
        res[f"eps{eps}"] = {"coverage": float(np.mean(cov)), "canonical_kept": float(np.mean(sc)), "lifted_physical": float(np.mean(sl)),
                            "prompt_items": float(np.mean(sp)), "lifted_frac_of_raw": float(np.mean(fr))}
        print(f"{name:5s} eps={eps}: cov={np.mean(cov):.3f} kept canon={np.mean(sc):.1f} lifted phys={np.mean(sl):.1f} "
              f"({100*np.mean(fr):.1f}% of raw) prompt items={np.mean(sp):.1f}")
    out[name] = res
# calibrated budget for the SQLite EX experiment: calibrate on every instance that is not a local (SQLite) one
v = views["jev"]
cal = np.asarray([i for i, e in enumerate(exs) if not str(e.ex_id).startswith("local")])
for eps in (0.05, 0.10):
    rr = np.sort(v.r[cal]); m = len(rr); j = math.ceil((m + 1) * (1 - eps))
    out[f"k_hat_eps{eps}"] = int(rr[j - 1]) if j <= m else None
    print(f"k_hat(eps={eps}) from {m} non-local instances: {out[f'k_hat_eps{eps}']}")
# same-distribution alternative: calibrate on the local (SQLite) instances with public gold only; for those instances
# themselves the EX experiment uses leave-one-out (threshold from the other local instances)
loc = [i for i, e in enumerate(exs) if str(e.ex_id).startswith("local")]
out["local_ids"] = [exs[i].ex_id for i in loc]; out["local_r"] = [int(v.r[i]) for i in loc]
for eps in (0.05, 0.10):
    rr = np.sort(v.r[loc]); m = len(rr); j = math.ceil((m + 1) * (1 - eps))
    out[f"k_hat_local_eps{eps}"] = int(rr[j - 1]) if j <= m else None
    print(f"k_hat_local(eps={eps}) from {m} local instances: {out[f'k_hat_local_eps{eps}']}  (r: {sorted(out['local_r'])})")
json.dump(out, open(os.path.join(RES, "spider2_shard_eval.json"), "w"), indent=1)
