"""SOFT certified pruning on OmniSQL's official inputs: every table and column stays in the schema (recall is never
lost), but only the certified / selected columns keep their official comment (question-matched example values and
description); all other columns are reduced to 'name type'.  PK / FK lines are kept.
  python official_soft.py <name> <prompts.jsonl,...> <tasks.jsonl,...> [sys1,sys2,...]
-> queue2/<name>.jsonl, tasks_<name>.jsonl with systems prefixed 'soft:'."""
import hashlib, json, os, re, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
import importlib.util
spec = importlib.util.spec_from_file_location("op", os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_prune_lib.py"))
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)

OFF = [d["input_seq"] for d in json.load(open(EXT + "/omnidata/data/dev_bird.json"))]
name, pfiles, tfiles = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
P = {}
for fn in pfiles:
    for l in open(fn, encoding="utf-8"):
        r = json.loads(l); P[r["h"]] = r["prompt"]


def soft(off, blocks):
    a = off.index("Database Schema:\n") + len("Database Schema:\n")
    b = off.index("\nThis schema describes")
    out, kept, total = [], 0, 0
    for blk in [t for t in re.split(r"\n\n(?=CREATE TABLE)", off[a:b].strip()) if t.strip()]:
        lines = blk.split("\n")
        tname = op.ident(lines[0][len("CREATE TABLE "):].rstrip(" ("))
        new = [lines[0]]
        for it in lines[1:]:
            s = it.strip()
            if s in (");", ")") or s.upper().startswith(("PRIMARY KEY", "CONSTRAINT", "FOREIGN KEY", "UNIQUE")):
                new.append(it); continue
            total += 1
            if op.kept_cols(blocks, tname, op.ident(s)):
                new.append(it); kept += 1
            else:
                d, sep, cm = it.partition(" -- ")
                new.append(d.rstrip())          # 'name type,' (separator comma is before the comment)
        out.append("\n".join(new))
    return off[:a] + "\n\n".join(out) + "\n" + off[b:], (kept, total)


newp, tasks, cov = {}, [], [0, 0]
for fn in tfiles:
    for l in open(fn, encoding="utf-8"):
        t = json.loads(l)
        if (only and t["sys"] not in only) or t["sys"] == "full" or t["h"] not in P:
            continue
        p, st = soft(OFF[t["i"]], op.our_blocks(P[t["h"]]))
        cov[0] += st[0]; cov[1] += st[1]
        h = hashlib.md5(p.encode("utf-8")).hexdigest(); newp[h] = p
        tasks.append({**t, "h": h, "sys": "soft:" + t["sys"]})
with open(f"{BOX}/queue2/{name}.jsonl", "w", encoding="utf-8") as f:
    for h, p in newp.items():
        f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
with open(f"{BOX}/tasks_{name}.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
from collections import Counter
print(name, Counter(t["sys"] for t in tasks), "prompts", len(newp), f"annotated/total columns {cov[0]}/{cov[1]}")
