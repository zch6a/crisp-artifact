"""Hard benchmark with canonicalization BEFORE reading (the corrected pipeline), all 1534 questions.

After signature-based de-duplication the hard2000 schema is exactly the BIRD-Dev schema, so CRISP's sets, prompts and
generations are those of the BIRD pipeline (same Jev scores, same certificate).  The only new system is the raw
2000+-column schema as the no-pruning reference.
  dump : raw full prompts for every hard2000 question -> results/prompts_hardraw.jsonl
  ex   : --gen <raw generations> -> EX of raw full vs canonical full / CRISP eps / hybrid / oracle (Qwen2.5-32B)
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, load_examples
from v2_e2e import build_systems, prompt_of, paired

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "ex"]); ap.add_argument("--gen", default="")
a = ap.parse_args()
hard = load_examples("hard2000")
if a.mode == "dump":
    with open(os.path.join(RES, "prompts_hardraw.jsonl"), "w", encoding="utf-8") as f:
        for e in hard:
            p = prompt_of(e, None)
            f.write(json.dumps({"h": hashlib.md5(p.encode("utf-8")).hexdigest(), "prompt": p}, ensure_ascii=False) + "\n")
    print(len(hard), "raw prompts; mean cols", np.mean([len(e.schema.columns()) for e in hard]))
else:
    from jevschema.exec_eval import evaluate_execution
    dev, hold = load_examples("dev"), load_examples("holdout")
    assert [e.ex_id for e in dev] == [e.ex_id for e in hard]
    S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
    Hy = json.load(open(os.path.join(RES, "hybrid_sets.json")))
    gc = {}
    for fn in (os.path.join(CACHE, "gen_qwen2.5-32b-awq.jsonl"), os.path.join(CACHE, "gen_hybrid_qwen32b.jsonl"), a.gen):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); gc[r["h"]] = r["sql"]
    systems = {"raw_full": [prompt_of(e, None) for e in hard], "canon_full": [prompt_of(e, None) for e in dev]}
    for name in ("t1_e0.02", "t1_e0.05", "t1_e0.1", "oracle"):
        systems[f"crisp_{name}" if name != "oracle" else "oracle"] = [prompt_of(e, k) for e, k in zip(dev, S[name])]
    systems["hybrid"] = [prompt_of(e, [tuple(x) for x in s]) for e, s in zip(dev, Hy["sets"])]
    flags, gold, out = {}, {}, {}
    for name, ps in systems.items():
        hs = [hashlib.md5(p.encode("utf-8")).hexdigest() for p in ps]
        miss = sum(h not in gc for h in hs)
        if miss:
            print(f"{name}: {miss} missing, skipped"); continue
        fl = []
        rep = evaluate_execution([(e.db_id, e.sql, gc[h], name) for e, h in zip(dev, hs)], gold_cache=gold, flags=fl)
        flags[name] = fl; out[name] = {"ex": rep.correct / max(1, rep.n - rep.gold_fail)}
        msg = f"{name:16s} EX={out[name]['ex']:.4f}"
        for ref in ("raw_full", "canon_full"):
            if ref in flags and ref != name:
                out[name][f"vs_{ref}"] = paired(flags[ref], fl); msg += f"  vs {ref} {out[name][f'vs_{ref}']}"
        print(msg, flush=True)
    json.dump({"systems": out, "flags": flags}, open(os.path.join(RES, "e2e_hardraw_qwen32b.json"), "w"))
