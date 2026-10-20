"""Prompt file for API self-consistency: full-schema prompts (from tasks_core4 'full') + a two-stage genprompt file.
  sc_prompts.py <genprompts.jsonl> <out.jsonl>"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import RESULTS
R = os.path.join(RESULTS, "")
full_h = [json.loads(l)["h"] for l in open(R + "tasks_core4.jsonl", encoding="utf-8") if '"sys": "full"' in l]
P = {json.loads(l)["h"]: json.loads(l)["prompt"] for l in open(R + "prompts_core4.jsonl", encoding="utf-8")}
seen = set()
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for h in full_h:
        if h not in seen:
            seen.add(h); f.write(json.dumps({"h": h, "prompt": P[h]}, ensure_ascii=False) + "\n")
    for l in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(l)
        if r["h"] not in seen:
            seen.add(r["h"]); f.write(l if l.endswith("\n") else l + "\n")
print("sc prompts", len(seen))
