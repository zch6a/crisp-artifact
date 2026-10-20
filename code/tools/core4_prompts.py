"""Prompt file for new generators: full, CRISP eps=0.05 (jev_t10:t1_e0.05), two-stage (hybrid_ds), zero-Jev two-stage
(hybrid_distds) -> results/prompts_core4.jsonl + results/tasks_core4.jsonl (BIRD-Dev, our prompt format)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import RESULTS
R = os.path.join(RESULTS, "")
T = [json.loads(l) for l in open(R + "tasks_full_all2.jsonl", encoding="utf-8")]
T += [t for t in map(json.loads, open(R + "tasks_main4_all2.jsonl", encoding="utf-8")) if t["sys"] == "jev_t10:t1_e0.05"]
for tag in ("hybrid_ds", "hybrid_distds"):
    T += [json.loads(l) for l in open(R + f"tasks_{tag}.jsonl", encoding="utf-8")]
P = {}
for fn in ("prompts_all2.jsonl", "hybrid_ds_genprompts.jsonl", "hybrid_distds_genprompts.jsonl"):
    for l in open(R + fn, encoding="utf-8"):
        r = json.loads(l); P.setdefault(r["h"], r["prompt"])
seen, miss = [], 0
for t in T:
    if t["h"] not in P:
        miss += 1
    elif t["h"] not in seen:
        seen.append(t["h"])
S = set()
with open(R + "prompts_core4.jsonl", "w", encoding="utf-8") as f:
    for h in seen:
        if h not in S:
            S.add(h); f.write(json.dumps({"h": h, "prompt": P[h]}, ensure_ascii=False) + "\n")
with open(R + "tasks_core4.jsonl", "w", encoding="utf-8") as f:
    for t in T:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
from collections import Counter
print("tasks", len(T), Counter(t["sys"] for t in T), "unique prompts", len(S), "missing", miss)
