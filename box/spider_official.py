"""Spider-Dev on OmniSQL's OFFICIAL inputs (data/dev_spider.json) with our certified kept sets (results/spider_sets.json,
built by code/scripts/v2_spider_sets.py).  Questions are matched to the official order by (db_id, question).
-> queue2/q20_spider.jsonl, tasks_q20_spider.jsonl (systems 'sp:full', 'sp:<sys>'; db path = spider database dir)."""
import hashlib, json, os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
sys.argv = [sys.argv[0], "x", "/dev/null", "/dev/null"]          # official_prune's argv parsing is not used here
import importlib.util
spec = importlib.util.spec_from_file_location("op", os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_prune_lib.py"))
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)

OFFIN = [d["input_seq"] for d in json.load(open(EXT + "/omnidata/data/dev_spider.json"))]
GOLD = json.load(open(EXT + "/omnidata/data/spider/dev.json"))
assert len(OFFIN) == len(GOLD)
key = {}
for j, g in enumerate(GOLD):
    key.setdefault((g["db_id"], " ".join(g["question"].split())), j)
S = json.load(open(BOX + "/spider_sets.json"))
newp, tasks, miss = {}, [], 0
for i, (db, q) in enumerate(zip(S["db"], S["question"])):
    j = key.get((db, " ".join(q.split())))
    if j is None:
        miss += 1; continue
    systems = {"full": None, **{k: v[i] for k, v in S["sets"].items()}}
    for name, kept in systems.items():
        if kept is None:
            p = OFFIN[j]
        else:
            blocks = {}
            for t, c in kept:
                blocks.setdefault(t.lower(), []).append(f"  {c.lower()} ")
            p, _ = op.prune(OFFIN[j], blocks)
        h = hashlib.md5(p.encode("utf-8")).hexdigest(); newp[h] = p
        tasks.append({"sys": "sp:" + name, "i": i, "h": h, "db": db, "gold": GOLD[j]["query"]})
with open(BOX + "/queue2/q20_spider.jsonl", "w", encoding="utf-8") as f:
    for h, p in newp.items():
        f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
with open(BOX + "/tasks_q20_spider.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print("matched", len(S["db"]) - miss, "missing", miss, "tasks", len(tasks), "prompts", len(newp))
