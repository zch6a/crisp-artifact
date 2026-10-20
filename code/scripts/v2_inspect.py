import json, os, collections
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import CACHE as C
for f in sorted(os.listdir(C)):
    if not f.endswith(".jsonl"):
        continue
    p = os.path.join(C, f)
    cnt = collections.Counter(); n = 0; ncols = []
    try:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); n += 1
            cnt[r["id"].split("-")[0]] += 1
            if "s" in r and isinstance(r["s"], dict):
                ncols.append(len(r["s"]))
    except Exception as e:
        print(f, "ERR", e); continue
    avg = sum(ncols) / len(ncols) if ncols else 0
    print(f"{f:<58} n={n:<6} prefixes={dict(cnt)} avg_cols={avg:.1f}")
