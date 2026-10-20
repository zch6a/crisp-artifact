"""EX of the LLM-linking baseline sets (Qwen2.5-32B generator, plain prompt), paired vs full schema."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, load_examples
from v2_e2e import prompt_of, paired
from jevschema.exec_eval import evaluate_execution
from jevschema.tokens import n_tokens

dev = load_examples("dev")
S = json.load(open(os.path.join(RES, "llmlink_sets.json")))
gc = {}
for fn in ("gen_qwen2.5-32b-awq.jsonl", "gen_llmlink_qwen32b.jsonl"):
    for l in open(os.path.join(CACHE, fn), encoding="utf-8"):
        r = json.loads(l); gc[r["h"]] = r["sql"]
systems = {"full": [None] * len(dev)}
for k, v in S["sets"].items():
    systems[k] = [[tuple(x) for x in s] or None for s in v]
out, flags, gold = {}, {}, {}
for name, kept in systems.items():
    prompts = [prompt_of(e, k) for e, k in zip(dev, kept)]
    sqls = [gc[hashlib.md5(p.encode("utf-8")).hexdigest()] for p in prompts]
    fl = []
    rep = evaluate_execution([(e.db_id, e.sql, s, name) for e, s in zip(dev, sqls)], gold_cache=gold, flags=fl)
    flags[name] = fl
    ex = rep.correct / max(1, rep.n - rep.gold_fail)
    cbar = float(np.mean([len(k) if k else len(e.schema.columns()) for e, k in zip(dev, kept)]))
    out[name] = {"ex": ex, "cbar": cbar, "prompt_tok": float(np.mean([n_tokens(p) for p in prompts]))}
    msg = f"{name:8s} EX={ex:.4f} Cbar={cbar:.1f} prompt_tok={out[name]['prompt_tok']:.0f}"
    if name != "full":
        out[name]["vs_full"] = paired(flags["full"], fl)
        msg += f"  vs full {out[name]['vs_full']}"
    print(msg, flush=True)
json.dump({"systems": out, "flags": flags}, open(os.path.join(RES, "e2e_llmlink_qwen32b.json"), "w"))
