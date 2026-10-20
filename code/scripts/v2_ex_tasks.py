"""Export execution-evaluation tasks for the parallel remote evaluator (box/ex_remote.py on the GPU box).
  v2_ex_tasks.py <map.json> <out.jsonl> [--systems a,b] [--prefix name_prefix]
map.json: {"ids": [...dev ids...], "systems": {name: [prompt hash per question]}}  (the *_map.json of every dump)
Each task: {"sys", "i", "h", "db", "gold"}; the evaluator joins generations by prompt hash.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import load_examples

ap = argparse.ArgumentParser(); ap.add_argument("map"); ap.add_argument("out")
ap.add_argument("--systems", default=""); ap.add_argument("--prefix", default="")
a = ap.parse_args()
M = json.load(open(a.map)); dev = load_examples("dev"); assert M["ids"] == [e.ex_id for e in dev]
names = a.systems.split(",") if a.systems else list(M["systems"])
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for s in names:
        for i, (e, h) in enumerate(zip(dev, M["systems"][s])):
            f.write(json.dumps({"sys": a.prefix + s, "i": i, "h": h, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n"); n += 1
print(f"{n} tasks, systems: {names}")
