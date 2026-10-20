"""EX task file for the GPT-6-Luna SC run: full-schema tasks (tasks_core4 'full') + value-aware two-stage tasks."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import BOX, CACHE
B = os.path.join(BOX, "")
T = [l for l in open(B + "tasks_core4.jsonl", encoding="utf-8") if json.loads(l)["sys"] == "full"]
T += open(B + "tasks_hybridv_g6l.jsonl", encoding="utf-8").readlines()
open(B + "tasks_sc_g6l.jsonl", "w", encoding="utf-8").write("".join(x if x.endswith("\n") else x + "\n" for x in T))
print("tasks", len(T))
# merge the two workers' generations (forward key + reverse key) into one file for ex_sc.py
seen, rows = set(), []
for fn in (os.path.join(CACHE, "gen_sc5_g6l.jsonl"), os.path.join(CACHE, "gen_sc5_g6lb.jsonl")):
    try:
        for l in open(fn, encoding="utf-8"):
            h = json.loads(l)["h"]
            if h not in seen:
                seen.add(h); rows.append(l if l.endswith("\n") else l + "\n")
    except FileNotFoundError:
        pass
open(B + "gen_sc5_g6l_all.jsonl", "w", encoding="utf-8").write("".join(rows))
print("merged generations", len(rows))
