"""Spider-Dev SOFT certified pruning on OmniSQL's official inputs (all columns kept; only certified columns keep their
official example-value comment).  -> queue2/q24sp.jsonl, tasks_q24sp.jsonl (systems 'spsoft:<sys>')."""
import hashlib, json, os, re, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
import importlib.util
spec = importlib.util.spec_from_file_location("op", os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_prune_lib.py"))
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)
OFFIN = [d["input_seq"] for d in json.load(open(EXT + "/omnidata/data/dev_spider.json"))]
GOLD = json.load(open(EXT + "/omnidata/data/spider/dev.json"))
key = {}
for j, g in enumerate(GOLD):
    key.setdefault((g["db_id"], " ".join(g["question"].split())), j)
S = json.load(open(BOX + "/spider_sets.json"))


def soft(off, blocks):
    a = off.index("Database Schema:\n") + len("Database Schema:\n")
    b = off.index("\nThis schema describes")
    out = []
    for blk in [t for t in re.split(r"\n\n(?=CREATE TABLE)", off[a:b].strip()) if t.strip()]:
        lines = blk.split("\n"); tname = op.ident(lines[0][len("CREATE TABLE "):].rstrip(" ("))
        new = [lines[0]]
        for it in lines[1:]:
            s = it.strip()
            if s in (");", ")") or s.upper().startswith(("PRIMARY KEY", "CONSTRAINT", "FOREIGN KEY", "UNIQUE")) \
                    or op.kept_cols(blocks, tname, op.ident(s)):
                new.append(it)
            else:
                new.append(it.partition(" -- ")[0].rstrip())
        out.append("\n".join(new))
    return off[:a] + "\n\n".join(out) + "\n" + off[b:]


newp, tasks = {}, []
for i, (db, q) in enumerate(zip(S["db"], S["question"])):
    j = key[(db, " ".join(q.split()))]
    for name in ("jev_t10:t1_e0.05", "jev_t10:t1_e0.1"):
        blocks = {}
        for t, c in S["sets"][name][i]:
            blocks.setdefault(t.lower(), []).append(f"  {c.lower()} ")
        p = soft(OFFIN[j], blocks); h = hashlib.md5(p.encode("utf-8")).hexdigest(); newp[h] = p
        tasks.append({"sys": "spsoft:" + name, "i": i, "h": h, "db": db, "gold": GOLD[j]["query"]})
with open(BOX + "/queue2/q24sp.jsonl", "w", encoding="utf-8") as f:
    for h, p in newp.items():
        f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
with open(BOX + "/tasks_q24sp.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print("spider soft tasks", len(tasks), "prompts", len(newp))
