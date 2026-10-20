"""Spider 2.0-lite loader (compact file produced by box_scripts/spider2_extract.py).

Only instances with a public gold SQL (and parsed requirements) are returned.  Tables whose column-name signature is
identical (date-sharded BigQuery tables such as events_20210101 ... events_20210131) are canonicalized to one
representative BEFORE any reading (shortest name, then lexicographic), exactly the de-duplication step validated on
the self-copy benchmark; gold requirements are mapped to the representatives.  Spider2 ships no foreign keys.
"""
from __future__ import annotations

import json

import re

from .data_loader import Example
from .gold_labels import GoldSet
from .schema_graph import Column, SchemaGraph, Table

# Shard rule (revision): identical column signature is NOT enough -- tables are merged only if they also share the
# name stem left after removing a trailing date / number / copy suffix (events_20210101 ~ events_20210131), i.e. they
# are partitions of one logical relation.  The member list of every class is kept (``shard_members``) so that the
# prompt can list the shard names and a kept representative is lifted back to all of its members.
_SUF = re.compile(r"(_?\d{4,8}|_?\d+|_(archive|backup|bak|old|copy|staging|tmp))$", re.I)


def name_stem(name: str) -> str:
    base = name.split(".")[-1]
    prev = None
    while prev != base:
        prev, base = base, _SUF.sub("", base)
    return name[: len(name) - len(name.split(".")[-1])] + base.lower()


def load_spider2(path: str, canonicalize=True):
    """canonicalize: False (raw), True / "signature" (earlier rule: identical column signature), "shard" (revision)."""
    D = json.load(open(path, encoding="utf-8"))
    schemas, out = {}, []
    for it in D["instances"]:
        if not it.get("requirements"):
            continue
        db = it["db"]
        if db not in schemas:
            tabs = D["schemas"][db]["tables"]
            rep_of, members = {}, {}
            if canonicalize:
                sig = {}
                for t in tabs:
                    key = tuple(sorted(c.lower() for c in t["columns"]))
                    if canonicalize == "shard":
                        key = (key, name_stem(t["name"]))
                    sig.setdefault(key, []).append(t["name"])
                for names in sig.values():
                    r = sorted(names, key=lambda x: (len(x), x))[0]
                    members[r] = sorted(names)
                    for nm in names:
                        rep_of[nm] = r
            keep, seen = [], set()
            for t in tabs:
                nm = rep_of.get(t["name"], t["name"])
                if nm != t["name"] or nm in seen:
                    continue
                seen.add(nm)
                cols, seen_c = [], set()
                for c, ty in zip(t["columns"], t["types"]):
                    if c.lower() in seen_c:
                        continue
                    seen_c.add(c.lower()); cols.append(Column(c, (ty or "text").lower(), False, []))
                keep.append(Table(nm, cols))
            schemas[db] = (SchemaGraph(db_id=db, tables=keep, fks=[]), rep_of,
                           {t.name: {c.name.lower(): c.name for c in t.columns} for t in keep},
                           {t.name: members.get(t.name, [t.name]) for t in keep})
        sch, rep_of, colmap, mem = schemas[db]
        reqs = []
        for req in (it.get("requirements2") or it["requirements"]):
            s = set()
            for t, c in req:
                rt = rep_of.get(t, t)
                cc = colmap.get(rt, {}).get(c.lower())
                if cc is not None:
                    s.add((rt, cc))
            if s:
                reqs.append(s)
        if not reqs:
            continue
        ex = Example(ex_id=it["id"], db_id=db, question=it["question"], sql=it.get("gold_sql") or "", schema=sch,
                     evidence="", difficulty=it["kind"])
        g = GoldSet(); g.requirements = reqs; g.tables = {t for r in reqs for t, _ in r}
        ex.gold = g
        ex.shard_members = mem
        out.append(ex)
    return out
