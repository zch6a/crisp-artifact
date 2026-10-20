"""Audit generated SQL of a model cache on the full-schema prompts: empty / non-SELECT / exec errors."""
import hashlib, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import CACHE, load_examples
from v2_e2e import prompt_of
from jevschema.exec_eval import execute_sql, bird_sqlite_path
model = sys.argv[1]
gc = {}
for l in open(os.path.join(CACHE, f"gen_{model}.jsonl"), encoding="utf-8"):
    r = json.loads(l); gc[r["h"]] = r["sql"]
dev = load_examples("dev")[::6]
kinds = collections.Counter(); samples = []
for e in dev:
    s = gc.get(hashlib.md5(prompt_of(e, None).encode("utf-8")).hexdigest(), None)
    if s is None:
        kinds["missing"] += 1; continue
    if not s.strip():
        kinds["empty"] += 1; samples.append(("EMPTY", e.question[:60], "")); continue
    rows, err = execute_sql(bird_sqlite_path(e.db_id), s, 20)
    if err:
        k = err.split(":")[0]; kinds["err:" + k] += 1
        if len(samples) < 6:
            samples.append((err[:70], e.question[:50], s[:200].replace("\n", " ")))
    else:
        kinds["ok_exec"] += 1
print(dict(kinds))
for x in samples[:6]:
    print(" |", x)
