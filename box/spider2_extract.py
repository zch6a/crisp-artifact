"""Spider 2.0-lite -> compact schemas + gold column requirements (for SRR / certification with the CRISP pipeline).

Schema: every <kind>/<db>/**/<table>.json of resource/databases (column_names, column_types, description), no sample
rows.  Gold: the public gold SQL (evaluation_suite/gold/sql/<id>.sql), parsed with sqlglot in the instance's dialect.
Requirements (same semantics as BIRD's GoldSet): one requirement per referenced column identifier = the set of
(table, column) pairs among the tables referenced in FROM/JOIN that carry a column of that name (nested BigQuery
fields `a.b` count as column `a`).  Sharded BigQuery tables (events_20210101, ...) referenced through a wildcard
(events_*) map to every matching shard.
Output: spider2_lite.json  {"instances": [...], "schemas": {db: {"tables": [{"name", "fullname", "columns",
        "types", "desc"}]}}, "stats": {...}}
"""
import glob, json, os, re, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
import sqlglot
from sqlglot import exp

ROOT = EXT + "/Spider2/spider2-lite"
KIND = {"bq": "bigquery", "ga": "bigquery", "sf": "snowflake", "local": "sqlite"}
DIALECT = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}

inst = [json.loads(l) for l in open(f"{ROOT}/spider2-lite.jsonl", encoding="utf-8")]
db_kind = {}
for kind in ("bigquery", "snowflake", "sqlite"):
    for db in os.listdir(f"{ROOT}/resource/databases/{kind}"):
        db_kind[db] = kind


def load_schema(db):
    kind = db_kind[db]
    tables = []
    for j in sorted(glob.glob(f"{ROOT}/resource/databases/{kind}/{db}/**/*.json", recursive=True)):
        try:
            d = json.load(open(j, encoding="utf-8"))
        except Exception:
            continue
        cols = d.get("column_names") or []
        if not cols:
            continue
        tables.append({"name": d.get("table_name") or os.path.basename(j)[:-5], "fullname": d.get("table_fullname") or "",
                       "columns": cols, "types": d.get("column_types") or [""] * len(cols),
                       "desc": [(x or "")[:200] for x in (d.get("description") or [""] * len(cols))]})
    return {"kind": kind, "tables": tables}


schemas, stats = {}, {"instances": len(inst), "with_gold": 0, "parsed": 0, "empty_req": 0}
out = []
for it in inst:
    db = it["db"]
    if db not in db_kind:
        continue
    if db not in schemas:
        schemas[db] = load_schema(db)
    sch = schemas[db]
    g = f"{ROOT}/evaluation_suite/gold/sql/{it['instance_id']}.sql"
    rec = {"id": it["instance_id"], "db": db, "kind": sch["kind"], "question": it["question"],
           "external_knowledge": it.get("external_knowledge") or "", "gold_sql": None, "requirements": None}
    if os.path.exists(g):
        stats["with_gold"] += 1
        sql = open(g, encoding="utf-8").read()
        rec["gold_sql"] = sql
        try:
            trees = sqlglot.parse(sql, read=DIALECT[sch["kind"]])
        except Exception:
            trees = []
        refs, idents = set(), set()
        for tr in trees:
            if tr is None:
                continue
            for t in tr.find_all(exp.Table):
                refs.add(t.name.lower())
            for c in tr.find_all(exp.Column):
                parts = [p.name for p in c.parts] if hasattr(c, "parts") else [c.name]
                for p in parts:
                    idents.add(p.lower())
        if trees:
            stats["parsed"] += 1
        tabs = []
        for tb in sch["tables"]:
            nm = tb["name"].lower()
            for r in refs:
                if nm == r or (r.endswith("*") and nm.startswith(r[:-1])) or (nm.startswith(r) and re.fullmatch(r"_?\d{6,8}", nm[len(r):] or "x")):
                    tabs.append(tb); break
        req = []
        for idf in sorted(idents):
            s = sorted({(tb["name"], c) for tb in tabs for c in tb["columns"] if c.lower() == idf})
            if s:
                req.append(s)
        rec["requirements"] = req
        rec["ref_tables"] = sorted({tb["name"] for tb in tabs})
        stats["empty_req"] += (not req)
    out.append(rec)
ncols = {db: sum(len(t["columns"]) for t in s["tables"]) for db, s in schemas.items()}
stats["cols_median_by_kind"] = {k: sorted(v)[len(v) // 2] for k in ("bigquery", "snowflake", "sqlite")
                                for v in [[ncols[d] for d in ncols if schemas[d]["kind"] == k]] if v}
json.dump({"instances": out, "schemas": schemas, "stats": stats}, open(BOX + "/spider2_lite.json", "w"))
print(json.dumps(stats, indent=1))
g = [r for r in out if r["requirements"]]
print("gold instances with requirements:", len(g), " by kind:",
      {k: sum(1 for r in g if r["kind"] == k) for k in ("bigquery", "snowflake", "sqlite")})
print("mean cols in schema (gold instances):", sum(ncols[r["db"]] for r in g) / max(1, len(g)))
