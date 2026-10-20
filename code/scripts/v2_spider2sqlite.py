"""Spider 2.0 SQLite subset (135 questions): execution accuracy of full schema vs CRISP vs two-stage CRISP.

Inputs are OmniSQL's official prompts (data/test_spider2_sqlite.json: DDL with example values); kept column sets
are applied verbatim with official_prune_lib.prune (as for BIRD).  Reader: Jev (cache/jev_t10_spider2sqlite.jsonl).
Budget k_hat: from results/spider2_shard_eval.json (--kfrom local: calibrated on the local Spider2-lite instances
with public gold, leave-one-out for those instances; --kfrom nonlocal: calibrated on all other Spider2-lite
instances).  Closure f1k1 (FK + PK; the SQLite files declare FKs).
  build : -> results/sp2s_genprompts.jsonl, sp2s_link_prompts.jsonl, sp2s_meta.json
  parse : --outputs <{h,text}> (selector) -> appends two-stage prompts to sp2s_genprompts.jsonl
  eval  : --gens tag1,tag2 (cache/gen_<tag>.jsonl {h, text|sql}) -> per-system EX with the official comparison
          (compare_multi_pandas_table, condition_cols / ignore_order from spider2_sqlite_eval.jsonl; last ```sql block)
"""
import argparse, hashlib, json, os, re, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, DATA, EXT, load_examples, read_scores
from jevschema.certify import kept_at, order_from_scores
from jevschema.selection import closure, parse_closure
import official_prune_lib as op

ROOT = os.path.join(EXT, "omnidata", "data")
MAX_ROWS = 100000   # largest gold result in spider2_sqlite/gold_exec_result is far below this (checked)
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["build", "parse", "eval", "selftest"])
ap.add_argument("--outputs", default=""); ap.add_argument("--gens", default=""); ap.add_argument("--kfrom", default="local")
ap.add_argument("--eps", default="0.05"); ap.add_argument("--tab", action="store_true", help="table-level second stage: keep every certificate column of the selected tables")
a = ap.parse_args()

exs = load_examples("spider2sqlite")
OFF = [d["input_seq"] for d in json.load(open(f"{ROOT}/test_spider2_sqlite.json", encoding="utf-8"))]
assert len(OFF) == len(exs) and all(e.question[:60] in o for e, o in zip(exs, OFF))
CFG = parse_closure("f1k1")
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_llm_link.py"), encoding="utf-8").read()
_ns = {"__file__": __file__}; exec(_src.split("\nap = argparse")[0], _ns)
TMPL, parse_cols = _ns["TMPL"], _ns["parse_cols"]
H = lambda s: hashlib.md5(s.encode()).hexdigest()


def blocks_of(cols):
    B = {}
    for t, c in cols:
        B.setdefault(t.lower(), []).append(f"  {c.lower()} ")
    return B


def pruned(i, e, cols):
    """official text with only cols kept; verbatim official text when nothing is removed"""
    if len(set(cols)) >= len(e.schema.columns()):
        return OFF[i]
    return op.prune(OFF[i], blocks_of(cols))[0]


def schema_text(p):
    s = p.index("Database Schema:\n") + len("Database Schema:\n")
    return p[s:p.index("\nThis schema describes")].strip()


def certificates():
    J = read_scores(os.path.join(CACHE, "jev_t10_spider2sqlite.jsonl"))
    E = json.load(open(os.path.join(RES, "spider2_shard_eval.json")))
    eps = a.eps
    out = []
    for e in exs:
        if a.kfrom == "local":
            ids, rs = E["local_ids"], E["local_r"]
            rr = sorted(r for i, r in zip(ids, rs) if i != e.ex_id)          # leave-one-out for calibration instances
            j = int(np.ceil((len(rr) + 1) * (1 - float(eps))))
            k = rr[j - 1] if j <= len(rr) else 10 ** 9
        else:
            k = E[f"k_hat_eps{eps}"] or 10 ** 9
        o = order_from_scores(e.schema.columns(), J[e.ex_id])
        out.append((k, kept_at(e.schema, o, min(k, len(o)), CFG)))
    return out


SYS = [f"full", f"crisp_{a.kfrom}_{a.eps}"]
if a.mode == "build":
    C = certificates()
    rows, link = [], []
    for i, (e, (k, kept)) in enumerate(zip(exs, C)):
        rows.append({"sys": "full", "i": i, "h": H(OFF[i]), "prompt": OFF[i]})
        p = pruned(i, e, kept)
        rows.append({"sys": SYS[1], "i": i, "h": H(p), "prompt": p, "k": int(min(k, 10 ** 6)), "kept": len(kept),
                     "n_cols": len(e.schema.columns())})
        lp = TMPL.format(db=e.db_id, schema=schema_text(p), ev=e.evidence or "", q=e.question)
        link.append({"h": H(lp), "prompt": lp, "i": i})
    gp = os.path.join(RES, "sp2s_genprompts.jsonl")
    if os.path.exists(gp):   # merge: keep systems built for other eps / two-stage runs
        names = {r["sys"] for r in rows}
        rows = [json.loads(l) for l in open(gp, encoding="utf-8") if json.loads(l)["sys"] not in names] + rows
    with open(gp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [r for r in rows if r["sys"] in (SYS[1], "full")]
    with open(os.path.join(RES, f"sp2s_link_prompts_{a.kfrom}_{a.eps}.jsonl"), "w", encoding="utf-8") as f:
        for r in link:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kk = [r for r in rows if r["sys"] != "full"]
    print(f"built {len(rows)} gen prompts; certificate: mean k={np.mean([min(r['k'], r['n_cols']) for r in kk]):.1f} "
          f"kept={np.mean([r['kept'] for r in kk]):.1f} of {np.mean([r['n_cols'] for r in kk]):.1f} cols; "
          f"full-schema questions (k>=n): {sum(r['k'] >= r['n_cols'] for r in kk)}")
elif a.mode == "parse":
    C = certificates()
    got = {}
    for fn in a.outputs.split(","):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); got[r["h"]] = r["text"]
    f1 = parse_closure("f1")
    rows, fb, miss = [], 0, 0
    for i, (e, (k, kept)) in enumerate(zip(exs, C)):
        p = pruned(i, e, kept)
        lp = TMPL.format(db=e.db_id, schema=schema_text(p), ev=e.evidence or "", q=e.question)
        t = got.get(H(lp))
        if t is None:
            miss += 1; t = ""
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
        sel = [c for c in parse_cols(t, e.schema) if c in set(kept)]
        if a.tab and sel:
            tabs = {tt for tt, _ in sel}; sel = [c for c in kept if c[0] in tabs]
        if not sel:
            sel, fb = list(kept), fb + 1
        fk = closure(e.schema, sel + [x for x in e.schema.columns() if x not in sel], len(sel), f1)
        q = pruned(i, e, fk)
        rows.append({"sys": f"{'tab' if a.tab else 'two'}_{a.kfrom}_{a.eps}", "i": i, "h": H(q), "prompt": q, "kept": len(fk)})
    old = [json.loads(l) for l in open(os.path.join(RES, "sp2s_genprompts.jsonl"), encoding="utf-8")]
    old = [r for r in old if r["sys"] != rows[0]["sys"]] + rows
    with open(os.path.join(RES, "sp2s_genprompts.jsonl"), "w", encoding="utf-8") as f:
        for r in old:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"two-stage: mean kept={np.mean([r['kept'] for r in rows]):.1f} fallback={fb} missing selector outputs={miss}")
else:
    sys.path.insert(0, os.path.join(EXT, "omnisql_repo"))
    import pandas as pd
    _ev = open(os.path.join(EXT, "omnisql_repo", "evaluate_spider2.py"), encoding="utf-8").read()
    _g = {"pd": pd, "math": __import__("math"), "List": list, "Union": object}
    exec(_ev[_ev.index("def compare_multi_pandas_table"):_ev.index("def get_sqlite_result")], _g)
    cmp_multi, cmp_one = _g["compare_multi_pandas_table"], _g["compare_pandas_table"]
    std = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{ROOT}/spider2_sqlite/spider2_sqlite_eval.jsonl", encoding="utf-8")}
    gdir = f"{ROOT}/spider2_sqlite/gold_exec_result"
    golds = {}
    for e in exs:
        fs = sorted(f for f in os.listdir(gdir) if re.match(rf"^{re.escape(e.ex_id)}(_[a-z])?\.csv$", f))
        golds[e.ex_id] = [pd.read_csv(os.path.join(gdir, f)) for f in fs]
    rows = [json.loads(l) for l in open(os.path.join(RES, "sp2s_genprompts.jsonl"), encoding="utf-8")] if a.mode == "eval" else []

    def last_sql(t):
        t = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S)
        b = re.findall(r"```sql\s*(.*?)```", t, re.S | re.I)
        return (b[-1] if b else t).strip()

    import io

    def run(db, sql):
        """official path: execute, write the result to CSV, read it back (dtypes as the official evaluator sees them)."""
        import time
        def go():
            con = sqlite3.connect(f"file:{ROOT}/spider2_sqlite/databases_original/{db}.sqlite?mode=ro", uri=True)
            deadline = time.time() + 120
            con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20000)
            try:
                buf = io.StringIO(); first = True; nrows = 0
                for ch in pd.read_sql_query(sql, con, chunksize=500):
                    nrows += len(ch)
                    if nrows > MAX_ROWS or time.time() > deadline:
                        return None   # no gold result has more than a few thousand rows: such a prediction cannot match
                    ch.to_csv(buf, header=first, index=False); first = False
                if first:
                    return None   # no result set at all (the official evaluator then fails on the missing CSV)
                buf.seek(0)
                return pd.read_csv(buf)
            finally:
                con.close()
        if not sql.strip():
            return None
        try:
            return go()
        except Exception:
            return None

    def score(ex_id, db, sql):
        df = run(db, sql)
        if df is None or not golds[ex_id]:
            return 0
        st = std[ex_id]
        try:
            if len(golds[ex_id]) == 1:
                return int(cmp_one(df, golds[ex_id][0], st["condition_cols"], st["ignore_order"]))
            io_ = st["ignore_order"]
            io_ = io_ if isinstance(io_, list) else [io_] * len(golds[ex_id])
            return int(cmp_multi(df, golds[ex_id], st["condition_cols"], io_))
        except Exception:
            return 0

    if a.mode == "selftest":
        # gold SQL of the local instances that have public gold in our Spider2-lite file must score 1
        D = json.load(open(os.path.join(DATA, "spider2_lite.json"), encoding="utf-8"))
        gs = {it["id"]: it.get("gold_sql") for it in D["instances"] if str(it["id"]).startswith("local") and it.get("gold_sql")}
        byid = {e.ex_id: e for e in exs}
        sc = {i: score(i, byid[i].db_id, s) for i, s in gs.items() if i in byid}
        print(f"selftest: gold SQL scores {sum(sc.values())}/{len(sc)}; failing: {[i for i, v in sc.items() if not v]}")
        sys.exit()
    report = {}
    for tag in a.gens.split(","):
        G = {}
        for l in open(os.path.join(CACHE, f"gen_{tag}.jsonl"), encoding="utf-8"):
            r = json.loads(l); G[r["h"]] = r.get("text", r.get("sql", ""))
        from concurrent.futures import ThreadPoolExecutor
        jobs = {}
        for r in rows:
            if r["h"] in G:
                e = exs[r["i"]]; jobs[(e.ex_id, last_sql(G[r["h"]]))] = e.db_id   # identical SQL is executed once
        with ThreadPoolExecutor(8) as pool:                                       # sqlite releases the GIL while executing
            res = dict(zip(jobs, pool.map(lambda kv: score(kv[0][0], kv[1], kv[0][1]), jobs.items())))
        per = {}
        for r in rows:
            e = exs[r["i"]]
            if r["h"] in G:
                per.setdefault(r["sys"], {})[e.ex_id] = res[(e.ex_id, last_sql(G[r["h"]]))]
        report[tag] = per
        print(tag, {k: f"{100*np.mean(list(v.values())):.2f} (n={len(v)})" for k, v in per.items()})
    json.dump(report, open(os.path.join(RES, f"sp2s_ex_{'_'.join(a.gens.split(','))}.json"), "w"), indent=1)
