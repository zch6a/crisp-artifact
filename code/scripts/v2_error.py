"""Error analysis of the strongest reader (Jev + closure f1p5s15k1, k=15): why is gold still missed?"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from jevschema.certify import kept_at

dev = load_examples("dev"); bm, em = lexical("dev")
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
J = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl"))
v = View(dev, combine(dev, f0, [J[e.ex_id] for e in dev], 0.02), "f1p5s15k1")
K = 15
cats, ex = collections.Counter(), []
for i, e in enumerate(dev):
    kept = set(kept_at(e.schema, v.orders[i], K, v.cfg))
    miss = [req for req in e.gold.requirements if not (req & kept)]
    if not miss:
        continue
    for req in miss:
        t, c = sorted(req)[0]
        p = J[e.ex_id].get((t, c))
        fk = {(f.src_table, f.src_col) for f in e.schema.fks} | {(f.dst_table, f.dst_col) for f in e.schema.fks}
        if len(req) > 1:
            k = "ambiguous gold (multi-candidate)"
        elif p is None:
            k = "table outside gate"
        elif (t, c) in fk:
            k = "join key not covered"
        elif p < 0.1:
            k = "reader says irrelevant (p<0.1)"
        else:
            k = "ranked below budget (p>=0.1)"
        cats[k] += 1
        if len(ex) < 12:
            ex.append((e.db_id, e.question[:70], f"{t}.{c}", None if p is None else round(p, 3), k))
n_fail = sum(1 for i, e in enumerate(dev) if not all(req & set(kept_at(e.schema, v.orders[i], K, v.cfg)) for req in e.gold.requirements))
print(f"failed questions at k={K}: {n_fail}/{len(dev)}  (SRR {1-n_fail/len(dev):.4f}); missed requirements by cause:")
for k, c in cats.most_common():
    print(f"  {c:4d}  {k}")
for x in ex:
    print("   ", x)
