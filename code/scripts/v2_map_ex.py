"""EX of the systems in a prompt map (<dump>_map.json: {"ids", "systems": {name: [h...]}}) from generator caches,
paired vs reference systems, overall and per BIRD difficulty stratum.
  v2_map_ex.py <map.json> <cache1.jsonl,cache2.jsonl> <out_tag> [--refs full_ref.json:full] [--systems a,b]
The full-schema flags are taken from an existing e2e file of the same generator (--full-from).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, load_examples
from v2_e2e import paired
from jevschema.exec_eval import evaluate_execution

ap = argparse.ArgumentParser()
ap.add_argument("map"); ap.add_argument("caches"); ap.add_argument("tag")
ap.add_argument("--systems", default=""); ap.add_argument("--full-from", default="")
ap.add_argument("--refs", default="marginal_e0.05")
a = ap.parse_args()
M = json.load(open(a.map)); dev = load_examples("dev")
assert M["ids"] == [e.ex_id for e in dev]
gc = {}
for fn in a.caches.split(","):
    for l in open(fn, encoding="utf-8"):
        try:
            r = json.loads(l); gc[r["h"]] = r["sql"]
        except Exception:
            pass
flags, gold, out = {}, {}, {}
if a.full_from:
    flags["full"] = json.load(open(a.full_from))["flags"]["full"]
names = [s for s in (a.systems.split(",") if a.systems else M["systems"])]
strata = [e.difficulty for e in dev]
for name in names:
    hs = M["systems"][name]
    miss = sum(h not in gc for h in hs)
    if miss:
        print(f"{name}: {miss} generations missing, skipped", flush=True); continue
    fl = []
    rep = evaluate_execution([(e.db_id, e.sql, gc[h], name) for e, h in zip(dev, hs)], gold_cache=gold, flags=fl)
    flags[name] = fl
    out[name] = {"ex": rep.correct / max(1, rep.n - rep.gold_fail)}
    per = {}
    for s in ("simple", "moderate", "challenging"):
        v = [x for x, st in zip(fl, strata) if st == s and x is not None]
        per[s] = float(np.mean(v))
    out[name]["strata"] = per
    msg = f"{name:30s} EX={out[name]['ex']:.4f}  " + "  ".join(f"{s[:4]}={per[s]:.4f}" for s in per)
    for ref in ["full"] + a.refs.split(","):
        if ref in flags and ref != name:
            out[name][f"vs_{ref}"] = paired(flags[ref], fl)
            ch = [i for i, st in enumerate(strata) if st == "challenging"]
            out[name][f"vs_{ref}_challenging"] = paired([flags[ref][i] for i in ch], [fl[i] for i in ch])
            msg += f"\n      vs {ref}: all {out[name][f'vs_{ref}']}  challenging {out[name][f'vs_{ref}_challenging']}"
    print(msg, flush=True)
json.dump({"systems": out, "flags": flags}, open(os.path.join(RES, f"e2e_map_{a.tag}.json"), "w"))
