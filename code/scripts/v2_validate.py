"""Sanity: (1) closure sets are nested in k, (2) oracle-depth SRR@k reproduces README numbers."""
import os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, View, fused_scores, lexical, load_examples, read_scores
from jevschema.certify import check_nested
from jevschema.metrics import strict_recall
from jevschema.certify import kept_at

t0 = time.time()
dev = load_examples("dev")
bm, em = lexical("dev")
fused = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
l6 = read_scores(os.path.join(CACHE, "bird_layapareto_laya_ft_bird_vask_L1024_t6.jsonl"))
laya6 = [l6.get(e.ex_id, {}) for e in dev]
print(f"dev n={len(dev)} loaded {time.time()-t0:.0f}s", flush=True)
for name, sc in (("fused", fused), ("laya6", laya6)):
    for cfg in ("f1p5s15k1", "f1p8k1"):
        v = View(dev, sc, cfg)
        rng = random.Random(0)
        bad = sum(0 if check_nested(dev[i].schema, v.orders[i], v.cfg, range(1, v.n_cols[i] + 1)) else 1
                  for i in rng.sample(range(len(dev)), 150))
        line = []
        for k in (12, 15, 25, 35):
            srr = float(np.mean(v.r <= k))
            cb = float(np.mean([v.size(i, k) for i in range(len(dev))]))
            # cross-check against strict_recall on the actual kept set
            chk = float(np.mean([strict_recall(dev[i].gold, kept_at(dev[i].schema, v.orders[i], k, v.cfg)) for i in range(len(dev))]))
            line.append(f"k={k}: SRR={srr:.4f} (chk {chk:.4f}) Cbar={cb:.1f}")
        print(f"{name:6s} {cfg:10s} nested_violations={bad}/150 unreachable={int((v.r > v.n_cols).sum())} | " + " | ".join(line), flush=True)
print(f"done {time.time()-t0:.0f}s")
