"""Generic re-wrap into OmniSQL's official template.
  python omni_template2.py <out_name> <prompts1.jsonl,...> <tasks1.jsonl,...>
-> queue2/<out_name>.jsonl (prompts) and tasks_<out_name>.jsonl (EX tasks, systems prefixed 'tpl:').
"""
import hashlib, json, os, re, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS

name, pfiles, tfiles = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
TPL = open(MODELS + "/OmniSQL-7B/README.md", encoding="utf-8").read()
TPL = TPL.split("input_prompt_template = '''", 1)[1].split("'''", 1)[0]
P = {}
for fn in pfiles:
    for l in open(fn, encoding="utf-8"):
        r = json.loads(l); P[r["h"]] = r["prompt"]
pat = re.compile(r"Schema:\n(.*?)\n\n(?:Evidence: (.*?)\n)?Question: (.*?)\n\nReturn only the SQL query", re.S)
newp, tasks, bad = {}, [], 0
for fn in tfiles:
    for l in open(fn, encoding="utf-8"):
        t = json.loads(l)
        m = pat.search(P.get(t["h"], ""))
        if not m:
            bad += 1; continue
        q = m.group(3) + (f"\n{m.group(2)}" if m.group(2) else "")
        p = TPL.format(db_details=m.group(1), question=q)
        h = hashlib.md5(p.encode("utf-8")).hexdigest(); newp[h] = p
        tasks.append({**t, "h": h, "sys": "tpl:" + t["sys"]})
with open(f"{BOX}/queue2/{name}.jsonl", "w", encoding="utf-8") as f:
    for h, p in newp.items():
        f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
with open(f"{BOX}/tasks_{name}.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"{name}: {len(newp)} prompts, {len(tasks)} tasks, {bad} unparsed")
