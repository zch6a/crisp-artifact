"""CRISP on new real-schema benchmarks: SNAILS (SIGMOD'25; real databases incl. a 90k-column ERP schema) and BEAVER
(real enterprise data-warehouse schemas).  Same modules, closure, calibration and prompt template as the paper; the
reader is the distilled local reader (no API) fused with BM25 exactly as `dist10f` on BIRD.

  build   --bench B             examples + requirements + reference-SQL execution check (SNAILS on SQLite)
  reader  --bench B             distilled block-reader scores (GPU)
  cert    --bench B             reader order, oracle depths, cross-fitted certificates, uncalibrated budget rules,
                                selector prompts (pool k=16) and generation prompts
  cert2   --bench B --sel F     certified selection from table-level selector outputs F ({h,text})
  eval    --bench B --gen G     EX of generator outputs G ({h,text}) for every system (SNAILS: SQLite)
Paths are relative to $NB_ROOT (default: the repository root).
"""
import argparse, glob, hashlib, json, math, os, re, sqlite3, sys, time
ROOT = os.environ.get("NB_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path[:0] = [f"{ROOT}/pylib", f"{ROOT}/code/src", f"{ROOT}/code/scripts"]
import numpy as np
import sqlglot
from jevschema.schema_graph import Column, ForeignKey, SchemaGraph, Table
from jevschema.data_loader import Example
from jevschema.gold_labels import extract_gold
from jevschema.certify import covers, kept_at, oracle_depth, order_from_scores
from jevschema.selection import closure, parse_closure
from jevschema.summary import bm25_scores
from jevschema.tokens import build_gen_prompt, n_tokens, schema_prompt_text

CFG = parse_closure("f1k1"); F1 = parse_closure("f1")
EPS = (0.05, 0.10)
POOLK = int(os.environ.get("POOLK", "16"))
FULL_MAX_TOK = 28000                      # full-schema prompts above this do not fit the generators' context
LINK_TMPL = """You are an expert in SQLite schema linking. Given the database schema, the question and the evidence,
list EVERY column that a correct SQL query needs (SELECT, WHERE, JOIN keys, GROUP BY, ORDER BY).
Output one column per line in the form table.column and nothing else.

Database: {db}
Schema:
{schema}

Evidence: {ev}
Question: {q}
Columns:"""
h = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()

# T-SQL date/string functions used by SNAILS reference queries, registered on every SQLite connection (gold and
# predicted SQL alike), so dialect gaps do not count as errors for either side.
_connect = sqlite3.connect


def _part(v, i):
    try:
        return int(str(v)[:10].split("-")[i]) if v is not None else None
    except Exception:
        return None


def _connect_with_tsql(*args, **kw):
    con = _connect(*args, **kw)
    con.create_function("YEAR", 1, lambda v: _part(v, 0), deterministic=True)
    con.create_function("MONTH", 1, lambda v: _part(v, 1), deterministic=True)
    con.create_function("DAY", 1, lambda v: _part(v, 2), deterministic=True)
    con.create_function("CONCAT", -1, lambda *xs: "".join("" if x is None else str(x) for x in xs), deterministic=True)
    return con


sqlite3.connect = _connect_with_tsql
OUT = lambda b, f="": os.path.join(ROOT, "nb", b, f)


# ------------------------------------------------------------------------------------------------ schemas
def sqlite_schema(db_id, path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tabs, fks = [], []
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
    for t in names:
        tabs.append(Table(t, [Column(r[1], (r[2] or "text").lower(), bool(r[5]), []) for r in con.execute(f'PRAGMA table_info("{t}")')]))
        for r in con.execute(f'PRAGMA foreign_key_list("{t}")'):
            if r[4] is not None:
                fks.append(ForeignKey(t, r[3], r[2], r[4]))
    con.close()
    tn = {t.name.lower() for t in tabs}
    return SchemaGraph(db_id=db_id, tables=tabs, fks=[f for f in fks if f.dst_table.lower() in tn])


def mysql_dump_schema(db_id, path):
    """CREATE TABLE blocks of a mysqldump file (INSERT lines skipped): columns, primary keys, foreign keys."""
    tabs, fks, cur, cols, pks = [], [], None, [], set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if cur is None:
                m = re.match(r"CREATE TABLE `([^`]+)` \(", line)
                if m:
                    cur, cols, pks = m.group(1), [], set()
                continue
            s = line.strip()
            if s.startswith(")"):
                tabs.append(Table(cur, [Column(n, t, n in pks, []) for n, t in cols])); cur = None; continue
            m = re.match(r"`([^`]+)` (\w+)", s)
            if m:
                cols.append((m.group(1), m.group(2).lower())); continue
            m = re.match(r"PRIMARY KEY \((.+?)\)", s)
            if m:
                pks |= set(re.findall(r"`([^`]+)`", m.group(1)))
                cols = [(n, t) for n, t in cols]
                continue
            m = re.search(r"FOREIGN KEY \((.+?)\) REFERENCES `([^`]+)` \((.+?)\)", s)
            if m:
                for a, b in zip(re.findall(r"`([^`]+)`", m.group(1)), re.findall(r"`([^`]+)`", m.group(3))):
                    fks.append(ForeignKey(cur, a, m.group(2), b))
    tabs = [Table(t.name, [Column(c.name, c.dtype, c.name in {x for x in []} or c.is_pk, []) for c in t.columns]) for t in tabs]
    tn = {t.name.lower() for t in tabs}
    return SchemaGraph(db_id=db_id, tables=tabs, fks=[f for f in fks if f.dst_table.lower() in tn])


# ------------------------------------------------------------------------------------------------ benchmarks
def load_snails():
    D = f"{ROOT}/data/snails"
    dbs = {os.path.basename(p)[:-7]: p for p in glob.glob(f"{D}/*.sqlite")}
    lut = {k.lower(): k for k in dbs}
    sch, out = {}, []
    for fn in sorted(glob.glob(f"{D}/pairs/*.sql")):
        stem = os.path.basename(fn)[:-4]; db = lut.get(stem.split("-")[0].lower())
        if db is None:
            continue
        if db not in sch:
            sch[db] = sqlite_schema(db, dbs[db])
        txt = open(fn, encoding="utf-8", errors="ignore").read()
        for blk in re.split(r"(?m)^--\s*(?=\d+\s*:)", txt)[1:]:
            m = re.match(r"(\d+)\s*:\s*(.*?)\n(.*)", blk, re.S)
            if not m:
                continue
            sql = m.group(3).strip().rstrip(";").strip()
            if not sql:
                continue
            out.append({"id": f"{stem}#{m.group(1)}", "db": db, "question": m.group(2).strip(), "evidence": "",
                        "sql_orig": sql, "dialect": "tsql", "path": dbs[db]})
    return out, sch


WH_SKIP = {"SBODemoUS"}                      # the SAP ERP schema is excluded everywhere in the paper


def _alias(db):
    return re.sub(r"[^A-Za-z0-9]", "", db)[:16]


def load_snails_wh():
    """SNAILS warehouse: the 8 real SNAILS databases merged into ONE schema (tables renamed <alias>__<table>, keys and
    foreign keys kept), as in a warehouse that hosts many unrelated applications.  Questions and reference queries are
    unchanged except for the table prefix; a query is executed on its source database after removing its own prefix
    (a query that touches another application's tables fails, as it would on the warehouse)."""
    rows, sch = load_snails()
    rows = [r for r in rows if r["db"] not in WH_SKIP]
    tabs, fks, amap = [], [], {}
    for d, g in sorted(sch.items()):
        if d in WH_SKIP:
            continue
        a = _alias(d); amap[d] = a
        tabs += [Table(f"{a}__{t.name}", t.columns) for t in g.tables]
        fks += [ForeignKey(f"{a}__{f.src_table}", f.src_col, f"{a}__{f.dst_table}", f.dst_col) for f in g.fks]
    for r in rows:
        r["src_db"], r["alias"], r["db"] = r["db"], amap[r["db"]], "snails_wh"
    _SRC.update({d: g for d, g in sch.items() if d not in WH_SKIP})
    return rows, {"snails_wh": SchemaGraph(db_id="snails_wh", tables=tabs, fks=fks)}


_SRC = {}


def wh_gold(r):
    """Requirement of a warehouse question: extracted on its source database, exactly as in the SNAILS experiment,
    with tables renamed <alias>__<table> (so requirements are identical to SNAILS by construction)."""
    from jevschema.gold_labels import GoldSet
    g = extract_gold(r["sql_exec"], _SRC[r["src_db"]]); a = r["alias"]
    if g is None:
        return None
    return GoldSet([{(f"{a}__{t}", c) for t, c in R} for R in g.requirements], {f"{a}__{t}" for t in g.tables},
                   g.ambiguous, g.unresolved, g.parse_ok)


def _ctes(tree):
    return {c.alias_or_name.lower() for c in tree.find_all(sqlglot.exp.CTE)}


def prefix_sql(sql, a):
    tree = sqlglot.parse_one(sql, read="sqlite"); ct = _ctes(tree)
    for t in tree.find_all(sqlglot.exp.Table):
        if t.name and t.name.lower() not in ct:
            t.set("this", sqlglot.exp.to_identifier(f"{a}__{t.name}"))
    return tree.sql(dialect="sqlite")


def unprefix_sql(sql, a):
    """Predicted warehouse SQL -> SQL on the source database; None if it reads a table of another application."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    ct, p = _ctes(tree), f"{a}__".lower()
    for t in tree.find_all(sqlglot.exp.Table):
        n = t.name or ""
        if n.lower() in ct:
            continue
        if not n.lower().startswith(p):
            return None
        t.set("this", sqlglot.exp.to_identifier(n[len(p):]))
    return tree.sql(dialect="sqlite")


LOADERS = {"snails": lambda: load_snails(), "snails_wh": lambda: load_snails_wh()}


def load_beaver():
    import pandas as pd
    D = f"{ROOT}/data/beaver"
    import pickle
    pk = f"{D}/schemas.pkl"                     # parsing the 2 GB nova dump takes minutes; cache the schemas once
    if os.path.exists(pk):
        sch = pickle.load(open(pk, "rb"))
    else:
        sch = {d: mysql_dump_schema(d, f"{D}/beaver_db/{d}.sql") for d in ("dw", "neutron", "nova")}
        pickle.dump(sch, open(pk, "wb"))
    out = []
    for split in ("dw_real", "dw", "neutron", "nova"):
        df = pd.read_parquet(glob.glob(f"{D}/q__{split}-*.parquet")[0])
        for r in df.itertuples():
            try:
                dk = json.loads(r.domain_knowledge) if r.domain_knowledge else []
            except Exception:
                dk = []
            out.append({"id": r.id, "db": r.db, "split": split, "question": r.question, "evidence": " ".join(dk),
                        "sql_orig": r.sql, "dialect": "mysql", "path": ""})
    return out, sch


def to_sqlite(sql, dialect):
    try:
        return sqlglot.transpile(sql, read=dialect, write="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)[0]
    except Exception:
        return ""


def examples(bench):
    if bench.startswith("scale"):
        # schema-scaling workload (scale_build.py): BIRD-Dev stride-3 subset merged with distractor tables; gold SQL
        # and execution on the original BIRD sqlite (a query that reads a distractor table fails)
        import pickle
        exs = pickle.load(open(f"{ROOT}/nb/scale/{bench}.pkl", "rb"))
        dbd = os.environ.get("BIRD_DB", f"{ROOT}/data/bird_dev/dev_databases")
        rows = [{"id": e.ex_id, "db": e.db_id, "question": e.question, "evidence": e.evidence or "", "sql": e.sql,
                 "path": f"{dbd}/{e.db_id}/{e.db_id}.sqlite"} for e in exs]
        return rows, exs
    rows = json.load(open(OUT(bench, "examples.json"), encoding="utf-8"))
    loader = LOADERS.get(bench, load_beaver)   # beaver_sub: the fixed EX subset of BEAVER
    _, sch = loader()
    exs = []
    for r in rows:
        e = Example(ex_id=r["id"], db_id=r["db"], question=r["question"], sql=r["sql"], schema=sch[r["db"]], evidence=r["evidence"])
        e.gold = wh_gold(r) if "alias" in r else extract_gold(r["sql"], e.schema)
        exs.append(e)
    return rows, exs


# ------------------------------------------------------------------------------------------------ build
def cmd_build(a):
    os.makedirs(OUT(a.bench), exist_ok=True)
    rows, sch = LOADERS.get(a.bench, load_beaver)()
    from jevschema.exec_eval import execute_sql
    keep, stat = [], {"loaded": len(rows)}
    for r in rows:
        r["sql"] = to_sqlite(r["sql_orig"], r["dialect"])
        if not r["sql"]:
            stat["transpile_fail"] = stat.get("transpile_fail", 0) + 1; continue
        if "alias" in r:                        # warehouse: requirements on prefixed tables, execution on the source DB
            r["sql_exec"] = r["sql"]
            try:
                r["sql"] = prefix_sql(r["sql"], r["alias"])
            except Exception:
                stat["prefix_fail"] = stat.get("prefix_fail", 0) + 1; continue
        g = wh_gold(r) if "alias" in r else extract_gold(r["sql"], sch[r["db"]])
        if g is None or not g.requirements:
            stat["no_requirement"] = stat.get("no_requirement", 0) + 1; continue
        r["n_req"] = len(g.requirements)
        if r["path"]:
            rows_, err = execute_sql(r["path"], r.get("sql_exec", r["sql"]), 30.0)
            r["exec_ok"] = rows_ is not None
            if rows_ is None:
                r["exec_err"] = str(err)[:200]
        keep.append(r)
    stat["kept"] = len(keep)
    if keep and "exec_ok" in keep[0]:
        stat["exec_ok"] = sum(r["exec_ok"] for r in keep)
    for d, g in sch.items():
        stat[f"cols_{d}"] = g.n_columns()
    json.dump(keep, open(OUT(a.bench, "examples.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(stat, open(OUT(a.bench, "build_stats.json"), "w"), indent=1)
    print(json.dumps(stat, indent=1))


# ------------------------------------------------------------------------------------------------ reader
def cmd_reader(a):
    import torch
    import v2_block_reader as br
    rows, exs = examples(a.bench)
    dev = torch.device("cuda")
    rl, cfg, tok, model = br.load_model(f"{ROOT}/models/laya_block_distill", dev); model.eval()
    part = os.environ.get("PART", "")            # extra parallel workers: PART=i/n takes every n-th remaining question from i
    ALLTAB = os.environ.get("ALLTAB") == "1"      # score every table (no BM25 table gate); files blockall*
    pre = "blockall" if ALLTAB else "block"
    if os.environ.get("SUBSET") == "1":
        S_ = ex_subset(rows); exs = [e for i, e in enumerate(exs) if i in S_]
    path = OUT(a.bench, f"{pre}{'.' + part.replace('/', 'of') if part else ''}.jsonl")
    done = set()
    for p in glob.glob(OUT(a.bench, f"{pre}[.j]*jsonl")):
        done |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}
    pad = tok.pad_token_id or 0; t0 = time.time()
    with open(path, "a", encoding="utf-8") as f, torch.no_grad():
        todo = [x for x in exs if x.ex_id not in done]
        if part:
            i_, n_ = map(int, part.split("/")); todo = todo[::-1][i_::n_]
        for n, e in enumerate(todo):
            t1 = time.time()
            its = br.make_items(rl, tok, [e], 1024, 320, all_tables=ALLTAB, top_t=10)
            s = {}
            for g in br.batches(its, 16384, False):
                sub = [its[i] for i in g]
                lg, _ = br.forward_logits(model, rl, sub, pad, dev)
                for row, it in enumerate(sub):
                    for (t, c), v in zip(it["cols"], torch.sigmoid(lg[row, :len(it["cols"])].cpu()).tolist()):
                        s[f"{t}|{c}"] = float(v)
            f.write(json.dumps({"id": e.ex_id, "s": s, "wall_s": round(time.time() - t1, 3)}) + "\n"); f.flush()
            if (n + 1) % 200 == 0:
                print(f"  {n+1} {(time.time()-t0)/60:.1f} min", flush=True)
    print("READER_DONE", flush=True)


_BM = {}


def _bm25_cached(db, schema, question, k1=1.5, b=0.75):
    """Same BM25 as jevschema.summary.bm25_scores (lower-case, snake-split, stemmed), with the column documents of a
    database tokenised once instead of once per question (the 90k-column ERP schema made this the bottleneck)."""
    from jevschema.summary import _stem
    tok = lambda s: [_stem(p) for raw in re.findall(r"[a-z0-9_]+", s.lower()) for p in raw.split("_") if p]
    if db not in _BM:
        docs = [tok(f"{t} {t.replace('_', ' ')} {c} {c.replace('_', ' ')}") for t, c in schema.columns()]
        tfs = []
        df = {}
        for d in docs:
            tf = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            tfs.append(tf)
            for w in tf:
                df[w] = df.get(w, 0) + 1
        dl = np.asarray([len(d) for d in docs], float)
        _BM[db] = (tfs, df, dl, max(dl.mean(), 1e-9) if len(dl) else 1.0, len(docs))
    tfs, df, dl, avgdl, N = _BM[db]
    sc = np.zeros(N)
    for w in tok(question):            # duplicates counted, as in the original
        if w not in df:
            continue
        idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
        for i, tf in enumerate(tfs):
            f = tf.get(w)
            if f:
                sc[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / avgdl))
    return sc


def reader_scores(bench, exs):
    blk = {}
    pre = os.environ.get("READER", "block")         # block (BM25-gated, as on BIRD) or blockall (all tables)
    for l in (l for p in glob.glob(OUT(bench, f"{pre}[.j]*jsonl")) for l in open(p, encoding="utf-8")):
        r = json.loads(l); blk[r["id"]] = {tuple(k.split("|", 1)): v for k, v in r["s"].items()}
    out = []
    for e in exs:
        pairs = e.schema.columns()
        key = e.ex_id if bench.startswith("scale") else e.db_id      # scaled schemas differ per question
        b = np.asarray(_bm25_cached(key, e.schema, e.question + (" " + e.evidence if e.evidence else "")), float)
        b = 0.5 * (b / b.max() if b.max() > 0 else b)                       # fused lexical tier (BM25 part)
        lo, hi = b.min(), b.max(); n0 = (b - lo) / max(1e-9, hi - lo)
        s1 = blk[e.ex_id]
        out.append({p: (s1[p] if p in s1 else -1.0) + 0.02 * x for p, x in zip(pairs, n0)})
    return out


# ------------------------------------------------------------------------------------------------ cert
def prompt_of(e, kept):
    txt = schema_prompt_text(e.schema) if kept is None else schema_prompt_text(e.schema, tables=sorted({t for t, _ in kept}), columns=kept)
    return build_gen_prompt(e.db_id, txt, e.question, e.evidence or "")


def qhat(v, eps):
    v = np.sort(v); j = math.ceil((len(v) + 1) * (1 - eps))
    return int(v[j - 1]) if j <= len(v) else 10 ** 9


def cmd_cert(a):
    rows, exs = examples(a.bench)
    sc = reader_scores(a.bench, exs)
    orders = [order_from_scores(e.schema.columns(), s) for e, s in zip(exs, sc)]
    ncol = np.asarray([len(e.schema.columns()) for e in exs])
    r = np.asarray([oracle_depth(e.schema, o, e.gold, CFG) for e, o in zip(exs, orders)])
    n = len(exs); rng = np.random.default_rng(0)
    _sz = {}

    def size(i, k):                       # memoised: the same (question, budget) recurs across the 200 splits
        key = (i, min(int(k), int(ncol[i])))
        if key not in _sz:
            _sz[key] = len(kept_at(exs[i].schema, orders[i], key[1], CFG))
        return _sz[key]
    res = {"n": n, "mean_cols": float(ncol.mean()), "median_r": float(np.median(r))}
    # MONDRIAN=1: calibrate separately within each database (group-conditional split conformal; the guarantee then
    # holds within every database).  Splits are stratified by database so every group has calibration questions.
    G = np.asarray([e.db_id if os.environ.get("MONDRIAN") == "1" else "all" for e in exs])
    groups = sorted(set(G)); gidx = {g: np.where(G == g)[0] for g in groups}
    res["mondrian"] = os.environ.get("MONDRIAN") == "1"

    def halves(seed_rng):
        c, t = [], []
        for g in groups:
            p = seed_rng.permutation(gidx[g]); c += list(p[: len(p) // 2]); t += list(p[len(p) // 2:])
        return np.asarray(c), np.asarray(t)

    def budgets(cal, eps):
        return {g: qhat(r[[i for i in cal if G[i] == g]], eps) for g in groups}
    A, B = halves(np.random.default_rng(0))
    sets = {}
    for eps in EPS:
        cov, sz = [], []
        for _ in range(200):
            c, t = halves(rng); kg = budgets(c, eps)
            cov.append(np.mean([r[i] <= kg[G[i]] for i in t])); sz.append(np.mean([size(i, kg[G[i]]) for i in rng.choice(t, min(300, len(t)), replace=False)]))
        kgA, kgB = budgets(B, eps), budgets(A, eps)
        ki = {int(i): kgA[G[i]] for i in A}; ki.update({int(i): kgB[G[i]] for i in B})
        kA, kB = (kgA, kgB) if len(groups) > 1 else (kgA["all"], kgB["all"])
        sets[f"cert_e{eps}"] = [kept_at(e.schema, orders[i], min(ki[i], ncol[i]), CFG) for i, e in enumerate(exs)]
        res[f"cert_e{eps}"] = {"coverage_mean": float(np.mean(cov)), "coverage_p05": float(np.percentile(cov, 5)),
                               "size_mean": float(np.mean(sz)), "k_hat": [kA, kB],
                               "crossfit_coverage": float(np.mean([r[i] <= ki[i] for i in range(n)]))}
    # uncalibrated budget rules: BIRD's calibrated budget transferred (k=10), and a score threshold of 0.5
    sets["topk10_uncal"] = [kept_at(e.schema, orders[i], min(10, ncol[i]), CFG) for i, e in enumerate(exs)]
    th = []
    for i, (e, s) in enumerate(zip(exs, sc)):
        k = max(1, sum(1 for v in s.values() if v >= 0.5)); th.append(kept_at(e.schema, orders[i], min(k, ncol[i]), CFG))
    sets["thresh05_uncal"] = th
    for name in ("topk10_uncal", "thresh05_uncal"):
        res[name] = {"coverage": float(np.mean([covers(e.gold, set(k)) for e, k in zip(exs, sets[name])])),
                     "size_mean": float(np.mean([len(k) for k in sets[name]]))}
    # per-database coverage of the eps=0.05 certificate
    res["per_db_cert_e0.05"] = {d: [float(np.mean([covers(e.gold, set(sets["cert_e0.05"][i])) for i, e in enumerate(exs) if e.db_id == d])), float(np.mean([len(sets["cert_e0.05"][i]) for i, e in enumerate(exs) if e.db_id == d])), int(ncol[[i for i, e in enumerate(exs) if e.db_id == d][0]])]
                                for d in sorted({e.db_id for e in exs})}
    full_tok = [n_tokens(prompt_of(e, None)) if len(e.schema.columns()) < 6000 else 10 ** 9 for e in exs]
    res["full_fits"] = int(sum(t <= FULL_MAX_TOK for t in full_tok))
    json.dump(res, open(OUT(a.bench, "cert.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    pools = [kept_at(e.schema, orders[i], min(POOLK, ncol[i]), CFG) for i, e in enumerate(exs)]
    json.dump({"orders": [[list(p) for p in o[:400]] for o in orders], "pools": [[list(p) for p in x] for x in pools],
               "r": r.tolist()}, open(OUT(a.bench, "orders.json"), "w"))
    S = ex_subset(rows)
    with open(OUT(a.bench, "link_prompts.jsonl"), "w", encoding="utf-8") as f:
        for i, (e, pool) in enumerate(zip(exs, pools)):
            if i not in S:
                continue
            lp = LINK_TMPL.format(db=e.db_id, schema=schema_prompt_text(e.schema, tables=sorted({t for t, _ in pool}), columns=pool),
                                  ev=e.evidence or "", q=e.question)
            f.write(json.dumps({"h": h(lp), "id": e.ex_id, "prompt": lp}, ensure_ascii=False) + "\n")
    systems = {**sets, "full": [None if t <= FULL_MAX_TOK else "SKIP" for t in full_tok]}
    write_gen(a.bench, exs, {k: [x if i in S else "SKIP" for i, x in enumerate(v)] for k, v in systems.items()})


def ex_subset(rows):
    """Questions used for selection and EX.  SNAILS: all.  BEAVER: all 121 real dw queries plus a fixed random sample
    of 500 per remaining split (seed 0); coverage of the single-stage certificate always uses every question."""
    if not any("split" in r for r in rows) or rows[0].get("sub"):
        return set(range(len(rows)))
    rng, S = np.random.default_rng(0), set()
    for sp in ("dw_real", "dw", "neutron", "nova"):
        idx = [i for i, r in enumerate(rows) if r.get("split") == sp]
        S |= set(idx) if sp == "dw_real" else set(int(x) for x in rng.choice(idx, min(500, len(idx)), replace=False))
    return S


def write_gen(bench, exs, systems):
    seen = set(); fp = OUT(bench, "gen_prompts.jsonl")
    if os.path.exists(fp):
        seen = {json.loads(l)["h"] for l in open(fp, encoding="utf-8")}
    tasks = json.load(open(OUT(bench, "tasks.json"))) if os.path.exists(OUT(bench, "tasks.json")) else {}
    with open(fp, "a", encoding="utf-8") as f:
        for name, sets in systems.items():
            tasks[name] = []
            for e, k in zip(exs, sets):
                if isinstance(k, str):
                    tasks[name].append(None); continue
                gp = prompt_of(e, k); hg = h(gp); tasks[name].append(hg)
                if hg not in seen:
                    seen.add(hg); f.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
    json.dump(tasks, open(OUT(bench, "tasks.json"), "w"))
    print("generation prompts", len(seen), "systems", list(tasks))


# ------------------------------------------------------------------------------------------------ cert2
def cmd_cert2(a):
    _ns = {"__file__": __file__}
    exec(open(f"{ROOT}/code/scripts/v2_llm_link.py", encoding="utf-8").read().split("\nap = argparse")[0].replace("from v2data import RES, load_examples", "").replace("from v2_e2e import prompt_of", ""), _ns)
    parse_cols = _ns["parse_cols"]
    rows, exs = examples(a.bench)
    O = json.load(open(OUT(a.bench, "orders.json")))
    links = {json.loads(l)["id"]: json.loads(l)["h"] for l in open(OUT(a.bench, "link_prompts.jsonl"), encoding="utf-8")}
    got = {}
    for l in open(a.sel, encoding="utf-8"):
        x = json.loads(l); got[x["h"]] = re.sub(r"<think>.*?</think>", "", x["text"], flags=re.S)
    S = sorted(ex_subset(rows)); allex = exs; exs = [allex[i] for i in S]
    orders, r2 = [], []
    for j, e in enumerate(exs):
        i = S[j]
        pool = [tuple(p) for p in O["pools"][i]]
        sel = [c for c in parse_cols(got.get(links[e.ex_id], ""), e.schema) if c in set(pool)]
        tabs = {t for t, _ in sel} or {t for t, _ in pool}
        Bs = [c for c in pool if c[0] in tabs]
        full_order = order_from_scores(e.schema.columns(), dict(zip(map(tuple, O["orders"][i]), range(len(O["orders"][i]), 0, -1))))
        o = Bs + [tuple(c) for c in O["orders"][i] if tuple(c) not in set(Bs)]
        o += [c for c in e.schema.columns() if c not in set(o)]
        orders.append((o, len(Bs)))
        lo, hi = 0, len(o) - len(Bs)
        while lo < hi:
            mid = (lo + hi) // 2
            if covers(e.gold, set(closure(e.schema, o, len(Bs) + mid, F1))):
                hi = mid
            else:
                lo = mid + 1
        r2.append(lo)
    r2 = np.asarray(r2); n = len(exs); rng = np.random.default_rng(0)
    _sz = {}

    def size(i, lam):
        key = (i, int(lam))
        if key not in _sz:
            _sz[key] = len(closure(exs[i].schema, orders[i][0], min(len(orders[i][0]), orders[i][1] + key[1]), F1))
        return _sz[key]
    res = {"selector_recall": float(np.mean(r2 == 0)), "mondrian": os.environ.get("MONDRIAN") == "1"}; sets = {}
    G = np.asarray([e.db_id if res["mondrian"] else "all" for e in exs]); groups = sorted(set(G))
    gidx = {g: np.where(G == g)[0] for g in groups}

    def halves(rg):
        c, t = [], []
        for g in groups:
            p = rg.permutation(gidx[g]); c += list(p[: len(p) // 2]); t += list(p[len(p) // 2:])
        return np.asarray(c), np.asarray(t)

    def lams(cal, eps):
        return {g: qhat(r2[[i for i in cal if G[i] == g]], eps) for g in groups}
    A, B = halves(np.random.default_rng(0))
    for eps in EPS:
        cov, sz = [], []
        for _ in range(200):
            c, t = halves(rng); lg = lams(c, eps)
            cov.append(np.mean([r2[i] <= lg[G[i]] for i in t]))
            sz.append(np.mean([size(i, lg[G[i]]) for i in rng.choice(t, min(300, len(t)), replace=False)]))
        lgA, lgB = lams(B, eps), lams(A, eps)
        li = {int(i): lgA[G[i]] for i in A}; li.update({int(i): lgB[G[i]] for i in B})
        sets[f"sel_e{eps}"] = [closure(e.schema, orders[i][0], min(len(orders[i][0]), orders[i][1] + li[i]), F1) for i, e in enumerate(exs)]
        res[f"sel_e{eps}"] = {"coverage_mean": float(np.mean(cov)), "coverage_p05": float(np.percentile(cov, 5)),
                              "size_mean": float(np.mean(sz)), "lam": [lgA, lgB],
                              "per_db": {g: [float(np.mean([r2[i] <= li[int(i)] for i in gidx[g]])),
                                             float(np.mean([len(sets[f"sel_e{eps}"][i]) for i in gidx[g]]))] for g in groups}}
    json.dump(res, open(OUT(a.bench, "cert2.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    pos = {i: j for j, i in enumerate(S)}
    write_gen(a.bench, allex, {k: [v[pos[i]] if i in pos else "SKIP" for i in range(len(allex))] for k, v in sets.items()})


# ------------------------------------------------------------------------------------------------ eval
def cmd_eval(a):
    from jevschema.exec_eval import execute_sql, result_match
    from jevschema.generator import extract_sql
    rows, exs = examples(a.bench)
    tasks = json.load(open(OUT(a.bench, "tasks.json")))
    gen = {}
    for l in open(a.gen, encoding="utf-8"):
        x = json.loads(l); gen[x["h"]] = extract_sql(re.sub(r"<think>.*?</think>", "", x["text"], flags=re.S))
    gcache, out = {}, {}
    for name, hs in tasks.items():
        fl = []
        for r, hg in zip(rows, hs):
            if not r.get("exec_ok", True):
                fl.append(None); continue
            if hg is None:
                fl.append(None); continue
            if r["id"] not in gcache:
                gcache[r["id"]] = execute_sql(r["path"], r.get("sql_exec", r["sql"]), 30.0)[0]
            psql = gen.get(hg, "") if hg in gen else None
            if psql is not None and "alias" in r:   # warehouse: run on the source DB, foreign tables -> failure
                psql = unprefix_sql(psql, r["alias"])
            pr = execute_sql(r["path"], psql, 30.0)[0] if psql else None
            fl.append(int(result_match(pr, gcache[r["id"]])) if gcache[r["id"]] is not None else None)
        out[name] = fl
        v = [x for x in fl if x is not None]
        print(f"{name:18s} EX={np.mean(v) if v else float('nan'):.4f} (n={len(v)})", flush=True)
    json.dump(out, open(OUT(a.bench, f"flags_{a.tag}.json"), "w"))


def cmd_mksub(a):
    """nb/beaver_sub/examples.json: the fixed EX subset of BEAVER (all dw_real + 500 per split, seed 0)."""
    rows = json.load(open(OUT("beaver", "examples.json"), encoding="utf-8"))
    S = ex_subset(rows)
    sub = [dict(r, sub=True) for i, r in enumerate(rows) if i in S]
    os.makedirs(OUT("beaver_sub"), exist_ok=True)
    json.dump(sub, open(OUT("beaver_sub", "examples.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("beaver_sub", len(sub))


def cmd_fullprompts(a):
    """Full-schema generation prompts (independent of the reader/calibration), so generators can start at once."""
    rows, exs = examples(a.bench)
    S = ex_subset(rows); n = 0
    with open(OUT(a.bench, "gen_prompts_full.jsonl"), "w", encoding="utf-8") as f:
        for i, e in enumerate(exs):
            if i in S and len(e.schema.columns()) < 6000:
                gp = prompt_of(e, None)
                if n_tokens(gp) <= FULL_MAX_TOK:
                    f.write(json.dumps({"h": h(gp), "prompt": gp}, ensure_ascii=False) + "\n"); n += 1
    print("full prompts", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "reader", "cert", "cert2", "eval", "fullprompts", "mksub"])
    ap.add_argument("--bench", choices=["snails", "snails_wh", "beaver", "beaver_sub", "scale0", "scale500", "scale2000",
                                        "scale5000", "scale10000"], required=True)
    ap.add_argument("--sel", default=""); ap.add_argument("--gen", default=""); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"build": cmd_build, "reader": cmd_reader, "cert": cmd_cert, "cert2": cmd_cert2, "eval": cmd_eval, "fullprompts": cmd_fullprompts, "mksub": cmd_mksub}[a.cmd](a)
