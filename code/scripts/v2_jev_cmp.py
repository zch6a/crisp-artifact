"""Jev vs laya (pointwise / block) on the same dev questions (those Jev has scored)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, View, fused_scores, lexical, load_examples, read_meta, read_scores
from v2_frontier import combine
from jevschema.metrics import auroc

dev = load_examples("dev"); bm, em = lexical("dev")
J = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl")); mJ = read_meta(os.path.join(CACHE, "jev_t10_dev.jsonl"))
exs = [e for e in dev if e.ex_id in J]
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in exs]
src = {"jev": J, "pointwise": read_scores(os.path.join(CACHE, "laya_t10_dev.jsonl")), "block": read_scores(os.path.join(CACHE, "block_t10_dev.jsonl"))}
print(f"n={len(exs)}  jev in_tok/q={np.mean([mJ[e.ex_id]['enc_tokens'] for e in exs]):.0f} wall/q={np.mean([mJ[e.ex_id]['wall_s'] for e in exs]):.1f}s")
R = {}
for name, S in src.items():
    ys, ps = [], []
    for e in exs:
        g = e.gold.flat_pairs()
        for p, v in S[e.ex_id].items():
            ps.append(v); ys.append(1 if p in g else 0)
    sc = combine(exs, f0, [S[e.ex_id] for e in exs], 0.02)
    for cfg in ("f1k1", "f1p5s15k1"):
        v = View(exs, sc, cfg); R[(name, cfg)] = v.r
        line = " ".join(f"@{k}:{np.mean(v.r <= k):.4f}/{np.mean([v.size(i, k) for i in range(len(exs))]):.1f}" for k in (5, 10, 15, 20, 30))
        print(f"{name:9s} {cfg:10s} AUROC={auroc(np.asarray(ps), np.asarray(ys)):.4f} mean_log_r={np.mean(np.log(v.r)):.4f} SRR/Cbar {line}")
rng = np.random.default_rng(0)
for cfg in ("f1k1", "f1p5s15k1"):
    for k in (10, 15, 20):
        a = (R[("jev", cfg)] <= k).astype(float); b = (R[("pointwise", cfg)] <= k).astype(float)
        d = a - b; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        print(f"  jev - pointwise {cfg} SRR@{k}: {d.mean():+.4f}  95% CI [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")
