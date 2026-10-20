"""EX on the scaled benchmark from externally generated SQL (cache-only), paired vs full schema per size."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, load_examples
from v2_e2e import paired
from jevschema.exec_eval import evaluate_execution

gen_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CACHE, "gen_scale_qwen32b.jsonl")
gc = {}
for l in open(gen_path, encoding="utf-8"):
    r = json.loads(l); gc[r["h"]] = r["sql"]
MAP = sys.argv[2] if len(sys.argv) > 2 else "prompts_scale"
OUT = sys.argv[3] if len(sys.argv) > 3 else "e2e_scale_qwen32b"
M = json.load(open(os.path.join(RES, MAP + "_map.json")))
dev = {e.ex_id: e for e in load_examples("dev")}
out, flags, gold = {}, {}, {}
for key in sorted(M, key=lambda k: (int(k.split(":")[0][1:]), k.split(":")[1] != "full", k)):
    m = M[key]
    miss = sum(1 for h in m["h"] if h not in gc)
    if miss:
        print(f"{key}: {miss} generations missing, skip"); continue
    fl = []
    rep = evaluate_execution([(dev[i].db_id, dev[i].sql, gc[h], key) for i, h in zip(m["ids"], m["h"])], gold_cache=gold, flags=fl)
    flags[key] = fl
    ex = rep.correct / max(1, rep.n - rep.gold_fail)
    out[key] = {"ex": ex, "cbar": m["cbar"], "srr": m["srr"], "pred_fail": rep.pred_fail}
    msg = f"{key:18s} EX={ex:.4f} Cbar={m['cbar']:7.1f} SRR={m['srr']:.3f} pred_fail={rep.pred_fail}"
    fk = key.split(":")[0] + ":full"
    if key != fk and fk in flags:
        out[key]["vs_full"] = paired(flags[fk], fl)
        msg += f"  vs full {out[key]['vs_full']}"
    fc = key.split(":")[0] + ":full_collapse"
    if key not in (fk, fc) and fc in flags:
        out[key]["vs_full_collapse"] = paired(flags[fc], fl)
        msg += f"  vs full_collapse {out[key]['vs_full_collapse']}"
    print(msg, flush=True)
json.dump({"systems": out, "flags": flags}, open(os.path.join(RES, OUT + ".json"), "w"))
