import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, read_meta
for name in ("laya_t10", "block_t10"):
    for split in ("dev", "holdout"):
        m = read_meta(os.path.join(CACHE, f"{name}_{split}.jsonl"))
        enc = np.asarray([v["enc_tokens"] for v in m.values()], float)
        seq = np.asarray([v["n_seq"] for v in m.values()], float)
        wall = np.asarray([v["wall_s"] for v in m.values()], float)
        print(f"{name:10s} {split:8s} n={len(m)} enc_tok/q mean={enc.mean():.0f} p90={np.quantile(enc,0.9):.0f} "
              f"seq/q={seq.mean():.1f} wall/q={wall.mean():.3f}s p90={np.quantile(wall,0.9):.3f}s")
