"""Matched-budget curve (reviewer request): fixed top-k pools k in {4,6,10,16,25,40} with the same reader and closure,
single-stage and with an open selector (Qwen3-8B) inside each pool; k=10 is exactly the eps=0.05 certificate.
For each generator: EX per system, paired difference to the full schema; pooled over generators: question-clustered
bootstrap (10,000 resamples) of the mean paired difference.  Questions where any compared flag is None are dropped.
  python v2_topk_curve.py gen1:flagsdir_prefix ...   (reads <CRISP_BOX>/flags_{full,topk_single,topk_two}_<gen>.json)
-> results/topk_curve.json"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import BOX, RESULTS
B = os.path.join(BOX, "")
gens = sys.argv[1:] or ["q7b", "l8b", "q3"]
rng = np.random.default_rng(0)
out, pooled = {}, {}
for g in gens:
    F = {}
    for part in ("full", "topk_single", "topk_two", "topk_tab", "topk_tab32"):
        fn = f"{B}flags_{part}_{g}.json"
        if os.path.exists(fn):
            F.update(json.load(open(fn)))
    if "full" not in F:
        continue
    res = {}
    for s, v in F.items():
        if s == "full":
            continue
        ii = [i for i in range(len(v)) if v[i] is not None and F["full"][i] is not None]
        d = np.asarray([v[i] - F["full"][i] for i in ii], float)
        res[s] = {"ex": float(np.mean([v[i] for i in ii])), "full": float(np.mean([F["full"][i] for i in ii])),
                  "delta": float(d.mean()), "n": len(ii), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
        pooled.setdefault(s, {})[g] = dict(zip(ii, d))
    out[g] = res
    print(g, "  ".join(f"{s}:{100*r['ex']:.2f}({100*r['delta']:+.2f})" for s, r in sorted(res.items())))
P = {}
for s, by in pooled.items():
    common = sorted(set.intersection(*[set(d) for d in by.values()]))
    M = np.asarray([[by[g][i] for g in by] for i in common])        # questions x generators
    q = M.mean(1)
    bs = [q[rng.integers(0, len(q), len(q))].mean() for _ in range(10000)]
    P[s] = {"gens": list(by), "n": len(common), "delta": float(q.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(np.asarray(bs) <= 0), np.mean(np.asarray(bs) >= 0))))}
    print(f"pooled {s:14s} {100*P[s]['delta']:+.2f} [{100*P[s]['ci'][0]:+.2f}, {100*P[s]['ci'][1]:+.2f}] p={P[s]['p_two_sided']:.4f} gens={P[s]['gens']}")
json.dump({"per_generator": out, "pooled": P}, open(os.path.join(RESULTS, "topk_curve.json"), "w"), indent=1)
