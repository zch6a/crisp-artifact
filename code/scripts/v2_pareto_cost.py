"""Cost side of the cost/EX Pareto table: mean kept columns and mean generator-prompt tokens (tiktoken, as everywhere
in the project) per system, for our prompts and for OmniSQL's official inputs (official text pruned with the same sets).
  -> results/pareto_cost.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, BOX
from jevschema.tokens import n_tokens

V = os.path.join(BOX, "")
out = {}
def tok_stats(tasks_file, prompt_files, systems=None):
    P = {}
    for pf in prompt_files:
        for l in open(pf, encoding="utf-8"):
            r = json.loads(l); P[r["h"]] = r["prompt"]
    acc = {}
    for l in open(tasks_file, encoding="utf-8"):
        t = json.loads(l)
        if systems and t["sys"] not in systems or t["h"] not in P:
            continue
        acc.setdefault(t["sys"], []).append(n_tokens(P[t["h"]]))
    return {s: float(np.mean(v)) for s, v in acc.items()}
off = tok_stats(V + "tasks_q18_off.jsonl", [V + "queue2/q18_off.jsonl"])
off.update(tok_stats(V + "tasks_q21_offv.jsonl", [V + "queue2/q21_offv.jsonl"]))
off.update(tok_stats(V + "tasks_q22_offg6l.jsonl", [V + "queue2/q22_offg6l.jsonl"]))
off.update(tok_stats(V + "tasks_q25_fullpool.jsonl", [V + "queue2/q25_fullpool.jsonl"]))
out["official_prompt_tokens"] = off
sets = {}
for tag in ("hybrid_ds", "hybridv_ds", "hybridv_g6l", "hybridv_fullpool", "hybrid_distds"):
    p = os.path.join(RES, f"{tag}_sets.json")
    if os.path.exists(p):
        sets[tag] = float(np.mean([len(s) for s in json.load(open(p))["sets"]]))
out["mean_cols"] = sets
json.dump(out, open(os.path.join(RES, "pareto_cost.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
