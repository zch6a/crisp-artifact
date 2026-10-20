import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, read_scores
new = read_scores(os.path.join(CACHE, "laya_t10_dev.jsonl"))
old = read_scores(os.path.join(CACHE, "bird_layapareto_laya_ft_bird_vask_L1024_t10.jsonl"))
ids = [i for i in new if i in old]
a, b, same_top = [], [], []
for i in ids:
    ks = [k for k in new[i] if k in old[i]]
    x = np.asarray([new[i][k] for k in ks]); y = np.asarray([old[i][k] for k in ks])
    a += list(x); b += list(y)
    same_top.append(len(set(np.argsort(-x)[:10]) & set(np.argsort(-y)[:10])) / 10)
a, b = np.asarray(a), np.asarray(b)
print(f"overlap questions {len(ids)} pairs {len(a)} pearson {np.corrcoef(a, b)[0,1]:.5f} max|diff| {np.abs(a-b).max():.4f} top10 overlap {np.mean(same_top):.3f}")
