"""Audit of Spider 2.0-lite canonicalization (reviewer: identical column signatures do not imply interchangeable tables).
Signature-only merging (the paper's earlier rule) vs a shard rule: merge only tables with the same column signature
AND the same name stem after removing a trailing date / number / copy suffix (events_20210101 -> events_).
Reports how many signature groups contain different stems, and for each instance how many raw tables / columns the
certificate expands to when every kept representative is lifted back to all of its member tables.
"""
import json, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import DATA, RES

D = json.load(open(os.path.join(DATA, "spider2_lite.json"), encoding="utf-8"))
SUF = re.compile(r"(_?\d{4,8}|_?\d+|_(archive|backup|bak|old|copy|staging|tmp))$", re.I)


def stem(name):
    base = name.split(".")[-1]
    prev = None
    while prev != base:
        prev, base = base, SUF.sub("", base)
    return name[: len(name) - len(name.split(".")[-1])] + base.lower()


tot_groups = mixed_groups = tables_in_mixed = 0
per_db = {}
for db, sch in D["schemas"].items():
    sig = defaultdict(list)
    for t in sch["tables"]:
        sig[tuple(sorted(c.lower() for c in t["columns"]))].append(t["name"])
    groups = [g for g in sig.values() if len(g) > 1]
    tot_groups += len(groups)
    mixed = [g for g in groups if len({stem(n) for n in g}) > 1]
    mixed_groups += len(mixed); tables_in_mixed += sum(len(g) for g in mixed)
    shard = defaultdict(list)
    for t in sch["tables"]:
        shard[(tuple(sorted(c.lower() for c in t["columns"])), stem(t["name"]))].append(t["name"])
    per_db[db] = {"raw_tables": len(sch["tables"]), "signature_classes": len(sig), "shard_classes": len(shard),
                  "mixed_signature_groups": len(mixed),
                  "example_mixed": [g[:4] for g in mixed[:2]]}
print(f"signature groups with >1 table: {tot_groups}; of these with different name stems: {mixed_groups} "
      f"({tables_in_mixed} tables)")
rows = sorted(per_db.items(), key=lambda kv: -kv[1]["mixed_signature_groups"])
for db, r in rows[:12]:
    print(db, r)
sig_cls = sum(r["signature_classes"] for r in per_db.values()); sh_cls = sum(r["shard_classes"] for r in per_db.values())
raw = sum(r["raw_tables"] for r in per_db.values())
print(f"tables raw {raw} -> signature classes {sig_cls} -> shard classes {sh_cls}")
json.dump({"per_db": per_db, "signature_groups": tot_groups, "mixed_groups": mixed_groups, "raw_tables": raw,
           "signature_classes": sig_cls, "shard_classes": sh_cls},
          open(os.path.join(RES, "spider2_canon_audit.json"), "w"), indent=1)
