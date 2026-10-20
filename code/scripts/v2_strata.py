"""BIRD difficulty strata (simple / moderate / challenging): EX gain of a certified system over the full schema,
pooled across generators with the question-clustered bootstrap, plus SRR / Cbar of the certified sets per stratum.
  v2_strata.py [--system t1_e0.05] [--tier1 jev_t10]
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, load_examples
from v2_e2e import build_systems

ap = argparse.ArgumentParser(); ap.add_argument("--system", default="t1_e0.05"); ap.add_argument("--tier1", default="jev_t10")
a = ap.parse_args()
dev, hold = load_examples("dev"), load_examples("holdout")
diff = [e.difficulty or "unknown" for e in dev]
S, _ = build_systems(argparse.Namespace(tier1=a.tier1, beta=0.02, route_frac=0.4), dev, hold)
kept = S[a.system]
rows = {}
for p in sorted(glob.glob(os.path.join(RES, f"e2e_*_{a.tier1}_*.json"))):
    d = json.load(open(p)); fl = d.get("flags", {})
    if "full" not in fl or a.system not in fl:
        continue
    tag = os.path.basename(p)[4:-5]; stride = int(re.search(r"_s(\d+)", tag).group(1))
    for i, (x, y) in enumerate(zip(fl["full"], fl[a.system])):
        if x is not None and y is not None:
            rows.setdefault(i * stride, []).append((tag, float(y) - float(x), float(x), float(y)))
rng = np.random.default_rng(0)
print(f"system {a.tier1}:{a.system}; generators: {sorted({t for v in rows.values() for t, *_ in v})}")
out = {}
for s in ("simple", "moderate", "challenging"):
    qs = [q for q in rows if diff[q] == s]
    sums = np.array([sum(r[1] for r in rows[q]) for q in qs]); cnt = np.array([len(rows[q]) for q in qs])
    exf = np.mean([r[2] for q in qs for r in rows[q]]); exs = np.mean([r[3] for q in qs for r in rows[q]])
    idx = rng.integers(0, len(qs), (10000, len(qs))); bs = sums[idx].sum(1) / cnt[idx].sum(1)
    m = sums.sum() / cnt.sum(); lo, hi = np.quantile(bs, [.025, .975]); pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    ii = [i for i in range(len(dev)) if diff[i] == s]
    srr = np.mean([all(req & set(kept[i]) for req in dev[i].gold.requirements) for i in ii])
    cb = np.mean([len(kept[i]) for i in ii]); full_cols = np.mean([len(dev[i].schema.columns()) for i in ii])
    out[s] = {"n": len(qs), "ex_full": exf, "ex_sys": exs, "diff": m, "ci": [lo, hi], "p": pv, "srr": srr, "cbar": cb}
    print(f"{s:12s} n={len(qs):4d}  EX full={exf:.4f} -> {exs:.4f}  diff={m:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.4f}  "
          f"SRR={srr:.3f}  Cbar={cb:.1f}/{full_cols:.1f}")
json.dump(out, open(os.path.join(RES, f"strata_{a.tier1}_{a.system}.json"), "w"), indent=1, default=float)
