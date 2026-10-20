"""Closure ablation x reader, and difficulty breakdown (BIRD-Dev 1534). Paired bootstrap vs full closure."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine

dev = load_examples("dev"); bm, em = lexical("dev")
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
R = {"fused": f0}
for k, src in (("laya", "laya_t10"), ("block", "block_t10"), ("jev", "jev_t10")):
    S = read_scores(os.path.join(CACHE, f"{src}_dev.jsonl")); R[k] = combine(dev, f0, [S[e.ex_id] for e in dev], 0.02)
cfgs = ["f0", "f1", "f1k1", "f1p5", "f1p5k1", "f1s15k1", "f1p5s15k1"]
rng = np.random.default_rng(0); out = []
for name, sc in R.items():
    base = None
    for cfg in cfgs:
        v = View(dev, sc, cfg)
        row = {"reader": name, "cfg": cfg}
        for k in (10, 20):
            cov = (v.r <= k).astype(float); cb = float(np.mean([v.size(i, k) for i in range(len(dev))]))
            row[f"srr@{k}"] = float(cov.mean()); row[f"cbar@{k}"] = cb; row[f"_c{k}"] = cov
        out.append(row)
    full = [r for r in out if r["reader"] == name and r["cfg"] == "f1p5s15k1"][0]
    for r in [r for r in out if r["reader"] == name]:
        d = full["_c20"] - r["_c20"]; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
        r["gain_full_minus_this@20"] = float(d.mean()); r["ci"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        print(f"{name:6s} {r['cfg']:10s} SRR@10={r['srr@10']:.4f} ({r['cbar@10']:.1f})  SRR@20={r['srr@20']:.4f} ({r['cbar@20']:.1f})"
              f"  full-this@20={r['gain_full_minus_this@20']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]", flush=True)
print("\n== difficulty breakdown (f1p5s15k1, k=15) ==")
diffs = sorted({e.difficulty for e in dev})
for name, sc in R.items():
    v = View(dev, sc, "f1p5s15k1")
    parts = []
    for dname in diffs:
        idx = [i for i, e in enumerate(dev) if e.difficulty == dname]
        parts.append(f"{dname}(n={len(idx)}): {np.mean(v.r[idx] <= 15):.4f}")
    print(f"{name:6s} " + "  ".join(parts), flush=True)
json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in out], open(os.path.join(RES, "ablation_closure.json"), "w"), indent=1)
