"""jev_fkm = FK structural gate applied uniformly (identical to top-10 gate on DBs with <= 10 tables)."""
import json, os, shutil
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import CACHE as C
fk = {}
for l in open(os.path.join(C, "jev_fk_dev.jsonl"), encoding="utf-8"):
    r = json.loads(l); fk[r["id"]] = l
n = 0
with open(os.path.join(C, "jev_fkm_dev.jsonl"), "w", encoding="utf-8") as f:
    seen = set()
    for l in open(os.path.join(C, "jev_t10_dev.jsonl"), encoding="utf-8"):
        i = json.loads(l)["id"]
        if i in seen:
            continue
        seen.add(i)
        if i in fk:
            f.write(fk[i]); n += 1
        else:
            f.write(l)
shutil.copy(os.path.join(C, "jev_t10_holdout.jsonl"), os.path.join(C, "jev_fkm_holdout.jsonl"))
print("merged, replaced", n)
