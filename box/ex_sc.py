"""Self-consistency EX: for each question, execute every sampled SQL, group candidates by (normalized) execution result,
pick the largest group (ties -> earliest sample), and score it against the gold result.  Failed executions never win
unless every candidate fails.  Same gold semantics as ex_remote.py.
  python ex_sc.py <tasks.jsonl> <gens_with_sqls.jsonl> <out_flags.json> [procs]
"""
import json, os, sys
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exec_eval import execute_sql, result_match as _rm, _norm_rows
SETMATCH = bool(os.environ.get("SETMATCH"))   # BIRD/OmniSQL official metric: set(pred) == set(gold)


def result_match(p, g):
    if SETMATCH:
        return p is not None and g is not None and set(map(tuple, p)) == set(map(tuple, g))
    return _rm(p, g)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.environ.get("DBDIR", os.path.join(_REPO, "data", "bird_dev", "dev_databases"))
DBFMT = os.environ.get("DBFMT", "{DB}/{db}.sqlite")
tasks = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
G = {}
for l in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(l); G[r["h"]] = r.get("sqls") or [r["sql"]]


def key(rows):
    try:
        return json.dumps(sorted(map(repr, _norm_rows(rows))))
    except Exception:
        return repr(rows)


def run(t):
    path = DBFMT.format(DB=DB, db=t["db"])
    grows, gerr = execute_sql(path, t["gold"], 30.0)
    if gerr:
        return t["sys"], t["i"], None
    cands = G.get(t["h"])
    if not cands:
        return t["sys"], t["i"], None
    res = []
    for j, s in enumerate(cands):
        rows, err = execute_sql(path, s, 30.0)
        res.append((j, rows if not err else None))
    ok = [(j, r) for j, r in res if r is not None]
    if not ok:
        return t["sys"], t["i"], 0
    cnt = Counter(key(r) for _, r in ok)
    best = max(cnt.values())
    win = next(r for _, r in ok if cnt[key(r)] == best)
    return t["sys"], t["i"], (1 if result_match(win, grows) else 0)


if __name__ == "__main__":
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    out = {}
    with Pool(procs) as p:
        for s, i, v in p.imap_unordered(run, tasks, chunksize=8):
            out.setdefault(s, {})[i] = v
    res = {s: [d.get(i) for i in range(max(d) + 1)] for s, d in out.items()}
    json.dump(res, open(sys.argv[3], "w"))
    for s, fl in res.items():
        v = [x for x in fl if x is not None]
        print(f"{s:34s} SC-EX={sum(v)/max(1,len(v)):.4f} ({sum(v)}/{len(v)})")
