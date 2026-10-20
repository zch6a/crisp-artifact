"""Reader ensemble: pointwise laya + block reader (+ tiny fused prior). Weight chosen on holdout."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, View, fused_scores, lexical, load_examples, read_scores


def rank01(d, pairs):
    v = np.asarray([d.get(p, -1.0) for p in pairs], float)
    r = np.empty(len(v)); r[np.argsort(-v, kind="stable")] = np.arange(len(v))
    return 1.0 - r / max(1, len(v) - 1)


def ens(exs, f, L, B, w, mode):
    out = []
    for e in exs:
        pairs = e.schema.columns(); l, b = L[e.ex_id], B[e.ex_id]
        f0 = fused_scores(e, *f(e))
        fv = np.asarray([f0[p] for p in pairs]); fv = (fv - fv.min()) / max(1e-9, fv.max() - fv.min())
        if mode == "prob":
            lv = np.asarray([l.get(p, -1.0) for p in pairs]); bv = np.asarray([b.get(p, -1.0) for p in pairs])
            s = (1 - w) * lv + w * bv + 0.02 * fv
        else:
            s = (1 - w) * rank01(l, pairs) + w * rank01(b, pairs) + 0.02 * fv
        out.append(dict(zip(pairs, s)))
    return out


dev, hold = load_examples("dev"), load_examples("holdout")
bm, em = lexical("dev"); hbm, hem = lexical("holdout")
Ld, Bd = read_scores(os.path.join(CACHE, "laya_t10_dev.jsonl")), read_scores(os.path.join(CACHE, "block_t10_dev.jsonl"))
Lh, Bh = read_scores(os.path.join(CACHE, "laya_t10_holdout.jsonl")), read_scores(os.path.join(CACHE, "block_t10_holdout.jsonl"))
fd = lambda e: (bm.get(e.ex_id, {}), em.get(e.ex_id, {}))
fh = lambda e: (hbm.get(e.ex_id, {}), hem.get(e.ex_id, {}))
for cfg in ("f1k1", "f1p5s15k1"):
    best = None
    for mode in ("prob", "rank"):
        for w in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
            vh = View(hold, ens(hold, fh, Lh, Bh, w, mode), cfg)
            mh = float(np.mean(np.log(vh.r)))
            vd = View(dev, ens(dev, fd, Ld, Bd, w, mode), cfg)
            line = " ".join(f"@{k}:{np.mean(vd.r <= k):.4f}/{np.mean([vd.size(i, k) for i in range(len(dev))]):.1f}" for k in (15, 20, 25, 30))
            print(f"{cfg:10s} {mode} w={w:<4} holdout mlr={mh:.4f} | dev mlr={np.mean(np.log(vd.r)):.4f} SRR/Cbar {line}", flush=True)
            if best is None or mh < best[0]:
                best = (mh, mode, w)
    print(f"  -> {cfg} holdout-selected: mode={best[1]} w={best[2]}", flush=True)
