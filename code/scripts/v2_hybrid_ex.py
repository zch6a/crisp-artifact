"""EX of hybrid linking (32B selects inside the CRISP certified set, then FK closure), Qwen2.5-32B generator,
paired vs full schema and vs the certified sets themselves."""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, load_examples
from v2_e2e import build_systems, prompt_of, paired
from jevschema.exec_eval import evaluate_execution
from jevschema.tokens import n_tokens

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
TIER1 = sys.argv[2] if len(sys.argv) > 2 else "jev_t10"
BASEGEN = sys.argv[3] if len(sys.argv) > 3 else "gen_qwen2.5-32b-awq.jsonl"
HYGEN = sys.argv[4] if len(sys.argv) > 4 else f"gen_hybrid{TAG}_qwen32b.jsonl"
MODEL = sys.argv[5] if len(sys.argv) > 5 else "qwen32b"
dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1=TIER1, beta=0.02, route_frac=0.4), dev, hold)
H = json.load(open(os.path.join(RES, f"hybrid{TAG}_sets.json")))
assert H["ids"] == [e.ex_id for e in dev]
gc = {}
for fn in (BASEGEN, HYGEN):
    for l in open(os.path.join(CACHE, fn), encoding="utf-8"):
        r = json.loads(l); gc[r["h"]] = r["sql"]
systems = {"full": [None] * len(dev), "jev_t1_e0.02": S["t1_e0.02"], "jev_t1_e0.1": S["t1_e0.1"],
           "hybrid": [[tuple(x) for x in s] for s in H["sets"]]}
out, flags, gold = {}, {}, {}
for name, kept in systems.items():
    prompts = [prompt_of(e, k) for e, k in zip(dev, kept)]
    hs = [hashlib.md5(p.encode("utf-8")).hexdigest() for p in prompts]
    miss = sum(h not in gc for h in hs)
    if miss:
        print(f"{name}: {miss} generations missing, skipped", flush=True); continue
    fl = []
    rep = evaluate_execution([(e.db_id, e.sql, gc[h], name) for e, h in zip(dev, hs)], gold_cache=gold, flags=fl)
    flags[name] = fl
    ex = rep.correct / max(1, rep.n - rep.gold_fail)
    cbar = float(np.mean([len(k) if k else len(e.schema.columns()) for e, k in zip(dev, kept)]))
    out[name] = {"ex": ex, "cbar": cbar, "prompt_tok": float(np.mean([n_tokens(p) for p in prompts]))}
    msg = f"{name:13s} EX={ex:.4f} Cbar={cbar:.1f} prompt_tok={out[name]['prompt_tok']:.0f}"
    if name != "full":
        out[name]["vs_full"] = paired(flags["full"], fl)
        msg += f"  vs full {out[name]['vs_full']}"
    print(msg, flush=True)
for ref in ("jev_t1_e0.02", "jev_t1_e0.1"):
    if ref in flags and "hybrid" in flags:
        out["hybrid"][f"vs_{ref}"] = paired(flags[ref], flags["hybrid"])
        print(f"hybrid vs {ref}: {out['hybrid'][f'vs_{ref}']}", flush=True)
json.dump({"model": MODEL, "tier1": "hybrid", "systems": out, "flags": flags},
          open(os.path.join(RES, f"e2e_hybrid{TAG}_{MODEL}.json"), "w"))
