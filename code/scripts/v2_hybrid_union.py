"""Union of second-stage selections (each already FK-closed, all inside the CRISP pool): raises second-stage recall
at a small size cost.  v2_hybrid_union.py <out_tag> <tag1> <tag2> [...]  (reads results/<tag>_sets.json)
-> results/<out_tag>_sets.json, <out_tag>_genprompts.jsonl, tasks_<out_tag>.jsonl"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, load_examples
from v2_e2e import prompt_of
from jevschema.certify import covers

out, tags = sys.argv[1], sys.argv[2:]
dev = load_examples("dev")
S = [json.load(open(os.path.join(RES, f"{t}_sets.json")))["sets"] for t in tags]
sets = []
for i, e in enumerate(dev):
    u = []
    for s in S:
        for x in s[i]:
            if tuple(x) not in u:
                u.append(tuple(x))
    sets.append(u)
for t, s in zip(tags, S):
    print(f"{t:14s} SRR={np.mean([covers(e.gold, set(map(tuple, x))) for e, x in zip(dev, s)]):.4f} Cbar={np.mean([len(x) for x in s]):.1f}")
print(f"{out:14s} SRR={np.mean([covers(e.gold, set(x)) for e, x in zip(dev, sets)]):.4f} Cbar={np.mean([len(x) for x in sets]):.1f}")
json.dump({"ids": [e.ex_id for e in dev], "sets": [[list(x) for x in s] for s in sets]}, open(os.path.join(RES, f"{out}_sets.json"), "w"))
seen = set()
with open(os.path.join(RES, f"{out}_genprompts.jsonl"), "w", encoding="utf-8") as f, \
        open(os.path.join(RES, f"tasks_{out}.jsonl"), "w", encoding="utf-8") as ft:
    for i, (e, s) in enumerate(zip(dev, sets)):
        gp = prompt_of(e, s); h = hashlib.md5(gp.encode()).hexdigest()
        ft.write(json.dumps({"sys": out, "i": i, "h": h, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
        if h not in seen:
            seen.add(h); f.write(json.dumps({"h": h, "prompt": gp}, ensure_ascii=False) + "\n")
