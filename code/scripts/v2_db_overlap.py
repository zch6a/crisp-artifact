import json, os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from jevschema.bird_train import load_bird_train
from jevschema.paths import RESULTS
exs, st = load_bird_train(limit=None, log_every=0)
sp = json.load(open(os.path.join(RESULTS, "bird_train_split.json")))
byid = {e.ex_id: e for e in exs}
def tset(e): return frozenset(t.name.lower() for t in e.schema.tables)
pool = [byid[i] for i in sp["pool_ids"]]; hold = [byid[i] for i in sp["holdout_ids"]]
pool_sets = list({tset(e) for e in pool}); hold_sets = collections.Counter(tset(e) for e in hold)
def jac(a, b): return len(a & b) / max(1, len(a | b))
mx = []
for s, c in hold_sets.items():
    m = max(jac(s, p) for p in pool_sets)
    mx += [m] * c
mx = np.asarray(mx)
print("holdout examples:", len(mx))
for th in (0.3, 0.5, 0.8, 1.0):
    print(f"  max table-set Jaccard with laya pool >= {th}: {np.mean(mx >= th):.3f}")
# cluster all train table sets into underlying dbs by union-find on Jaccard>=0.5
allsets = list({tset(e) for e in exs})
par = list(range(len(allsets)))
def f(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
for i in range(len(allsets)):
    for j in range(i + 1, len(allsets)):
        if jac(allsets[i], allsets[j]) >= 0.5:
            par[f(i)] = f(j)
roots = collections.Counter(f(i) for i in range(len(allsets)))
print("underlying clusters (Jaccard>=0.5):", len(roots))
cid = {s: f(i) for i, s in enumerate(allsets)}
pc = {cid[tset(e)] for e in pool}; hc = collections.Counter(cid[tset(e)] for e in hold)
print("pool clusters:", len(pc), " holdout clusters:", len(hc), " holdout clusters disjoint from pool:",
      sum(1 for c in hc if c not in pc), " examples in disjoint clusters:", sum(n for c, n in hc.items() if c not in pc))
