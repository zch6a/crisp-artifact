"""Precise gold-column extraction for Spider 2.0-lite (scope-resolved with sqlglot).

For every column reference in the gold SQL, resolve its source in the query scope: only references whose source is
a *base table* of the database count (CTE / subquery / alias columns are traced to their base columns through the
scope tree; UNNEST(t.col) nested fields are attributed to the top-level column t.col).  One requirement per
distinct base column; for wildcard/sharded tables the requirement is the set of that column over all matching shards.
Falls back to the permissive rule (spider2_extract.py) when qualification fails; reports how often.
Adds "requirements2" / "gold_mode" to spider2_lite.json.
"""
import json, os, re, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

D = json.load(open(BOX + "/spider2_lite.json"))
DIALECT = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}
stats = {"precise": 0, "fallback": 0, "req_before": 0, "req_after": 0}


def table_matches(tbl_name, ref):
    nm, r = tbl_name.lower(), ref.lower()
    return nm == r or (r.endswith("*") and nm.startswith(r[:-1])) or (nm.startswith(r) and re.fullmatch(r"_?\d{6,8}", nm[len(r):] or "x"))


for it in D["instances"]:
    if not it.get("gold_sql") or not it.get("requirements"):
        continue
    sch = D["schemas"][it["db"]]
    dialect = DIALECT[sch["kind"]]
    base = {}                      # (base table ref, column lower) -> set of (table, column) in schema
    try:
        tree = sqlglot.parse_one(it["gold_sql"], read=dialect)
        tree = qualify(tree, dialect=dialect, validate_qualify_columns=False, identify=False)
        for scope in traverse_scope(tree):
            for col in scope.columns:
                src = scope.sources.get(col.table)
                if isinstance(src, exp.Table):
                    ref = src.name
                    top = col.name
                elif isinstance(src, exp.Unnest) or (src is not None and not hasattr(src, "sources") and isinstance(getattr(src, "expression", None), exp.Unnest)):
                    un = src if isinstance(src, exp.Unnest) else src.expression
                    inner = un.find(exp.Column)
                    if inner is None or not isinstance(scope.sources.get(inner.table), exp.Table):
                        continue
                    ref, top = scope.sources[inner.table].name, inner.name
                else:
                    continue                                   # CTE / derived table column: its base columns appear in the inner scope
                key = (ref.lower(), top.lower())
                if key not in base:
                    base[key] = {(t["name"], c) for t in sch["tables"] if table_matches(t["name"], ref)
                                 for c in t["columns"] if c.lower() == top.lower()}
        req = [sorted(v) for v in base.values() if v]
        if req:
            stats["precise"] += 1
            it["requirements2"], it["gold_mode"] = req, "scoped"
        else:
            raise ValueError("no resolvable base columns")
    except Exception:
        stats["fallback"] += 1
        it["requirements2"], it["gold_mode"] = it["requirements"], "permissive"
    stats["req_before"] += len(it["requirements"]); stats["req_after"] += len(it["requirements2"])
json.dump(D, open(BOX + "/spider2_lite.json", "w"))
print(json.dumps(stats))
