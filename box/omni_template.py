"""Re-wrap our generation prompts in OmniSQL's official template (same schema text, same question + evidence), for a
fair plug-in test of the pruned schemas with a SOTA open text-to-SQL model.
Writes queue/q2_omnitpl.jsonl (prompts) and tasks_omnitpl.jsonl (EX tasks with the new hashes, systems 'tpl:<name>').
"""
import hashlib, json, os, re, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS

SYSTEMS = {"full", "jev_t10:t1_e0.05", "jev_t10:t1_e0.1", "oracle", "t0_e0.05", "mondrian3_hard0.02", "hybrid"}
TPL = open(MODELS + "/OmniSQL-7B/README.md", encoding="utf-8").read()
TPL = TPL.split("input_prompt_template = '''", 1)[1].split("'''", 1)[0]
P = {}
for l in open(BOX + "/prompts_omni.jsonl", encoding="utf-8"):
    r = json.loads(l); P[r["h"]] = r["prompt"]
pat = re.compile(r"Schema:\n(.*?)\n\n(?:Evidence: (.*?)\n)?Question: (.*?)\n\nReturn only the SQL query", re.S)
newp, tasks, bad = {}, [], 0
for l in open(BOX + "/tasks_omni.jsonl", encoding="utf-8"):
    t = json.loads(l)
    if t["sys"] not in SYSTEMS:
        continue
    m = pat.search(P[t["h"]])
    if not m:
        bad += 1; continue
    schema, ev, q = m.group(1), m.group(2), m.group(3)
    question = q + (f"\n{ev}" if ev else "")
    p = TPL.format(db_details=schema, question=question)
    h = hashlib.md5(p.encode("utf-8")).hexdigest()
    newp[h] = p
    tasks.append({**t, "h": h, "sys": "tpl:" + t["sys"]})
with open(BOX + "/queue/q2_omnitpl.jsonl", "w", encoding="utf-8") as f:
    for h, p in newp.items():
        f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
with open(BOX + "/tasks_omnitpl.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"{len(newp)} template prompts, {len(tasks)} tasks, {bad} unparsed")
