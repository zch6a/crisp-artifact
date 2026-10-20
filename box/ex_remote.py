"""Parallel execution accuracy on VGPU-A (same semantics as jevschema.exec_eval.evaluate_execution, after the
gold-cache fix: a failing gold query is excluded (None) for every system; a failing prediction counts as 0).
  python ex_remote.py <tasks.jsonl> <gen1.jsonl,gen2.jsonl> <out_flags.json> [procs]
Output: {system: [flag per question index]} with 1 / 0 / None; "missing" generations are reported and set to None.
"""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exec_eval import execute_sql, result_match as _rm
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
for fn in sys.argv[2].split(","):
    for l in open(fn, encoding="utf-8"):
        try:
            r = json.loads(l); G[r["h"]] = r["sql"]
        except Exception:
            pass
_gold = {}


def run(t):
    path = DBFMT.format(DB=DB, db=t["db"])
    key = (t["db"], t["gold"])
    if key not in _gold:
        _gold[key] = execute_sql(path, t["gold"], 30.0)
    grows, gerr = _gold[key]
    if gerr:
        return t["sys"], t["i"], None, "gold"
    sql = G.get(t["h"])
    if sql is None:
        return t["sys"], t["i"], None, "missing"
    prows, perr = execute_sql(path, sql, 30.0)
    if perr:
        return t["sys"], t["i"], 0, "pred_fail"
    return t["sys"], t["i"], (1 if result_match(prows, grows) else 0), ""


if __name__ == "__main__":
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 192
    out, stats = {}, {}
    tasks.sort(key=lambda t: (t["db"], t["gold"]))          # same gold in the same worker chunk -> cache hits
    with Pool(procs) as p:
        for s, i, v, why in p.imap_unordered(run, tasks, chunksize=16):
            out.setdefault(s, {})[i] = v
            if why:
                stats[(s, why)] = stats.get((s, why), 0) + 1
    res = {s: [d.get(i) for i in range(max(d) + 1)] for s, d in out.items()}
    json.dump(res, open(sys.argv[3], "w"))
    for s, fl in res.items():
        v = [x for x in fl if x is not None]
        miss = stats.get((s, "missing"), 0)
        print(f"{s:34s} EX={sum(v)/max(1,len(v)):.4f} ({sum(v)}/{len(v)}) gold_fail={stats.get((s,'gold'),0)} "
              f"pred_fail={stats.get((s,'pred_fail'),0)} missing={miss}")
