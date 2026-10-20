"""Mechanism control: is it the *choice* of context columns that helps, or just a shorter prompt?

For every question, control set = (CRISP set  intersect  gold columns)  +  random non-gold columns, so that the
control has exactly the CRISP set's size and the CRISP set's gold coverage; only the non-gold context differs
(relevance-ranked vs random).  Three seeds.  Generator Qwen2.5-32B (plain prompt).
  dump : generation prompts -> results/mech_genprompts.jsonl (+ map)
  ex   : --gen <jsonl {h,sql}> -> EX paired vs CRISP and vs full
"""
import argparse, hashlib, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, load_examples
from v2_e2e import build_systems, prompt_of, paired

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "ex"]); ap.add_argument("--gen", default="")
ap.add_argument("--system", default="t1_e0.05"); ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args()
dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
crisp = S[a.system]


def control(e, kept, seed):
    gold = set(e.gold.flat_pairs())
    keep_gold = [p for p in kept if p in gold]
    pool = [p for p in e.schema.columns() if p not in gold]
    r = random.Random(f"{seed}:{e.ex_id}")
    need = max(0, len(kept) - len(keep_gold))
    return keep_gold + r.sample(pool, min(need, len(pool)))


systems = {"full": [None] * len(dev), "crisp": crisp}
for s in a.seeds.split(","):
    systems[f"random_ctx_s{s}"] = [control(e, k, int(s)) for e, k in zip(dev, crisp)]
H = {name: [hashlib.md5(prompt_of(e, k).encode("utf-8")).hexdigest() for e, k in zip(dev, kept)] for name, kept in systems.items()}
if a.mode == "dump":
    seen = set(); n = 0
    with open(os.path.join(RES, "mech_genprompts.jsonl"), "w", encoding="utf-8") as f:
        for name, kept in systems.items():
            if name in ("full", "crisp"):
                continue
            for e, k, h in zip(dev, kept, H[name]):
                if h not in seen:
                    seen.add(h); n += 1
                    f.write(json.dumps({"h": h, "prompt": prompt_of(e, k)}, ensure_ascii=False) + "\n")
    json.dump({"ids": [e.ex_id for e in dev], "systems": H}, open(os.path.join(RES, "mech_map.json"), "w"))
    print(f"{n} control prompts; mean size crisp={np.mean([len(k) for k in crisp]):.1f} "
          f"control={np.mean([len(k) for k in systems['random_ctx_s0']]):.1f}")
else:
    from jevschema.exec_eval import evaluate_execution
    gc = {}
    for fn in (os.path.join(CACHE, "gen_qwen2.5-32b-awq.jsonl"), a.gen):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); gc[r["h"]] = r["sql"]
    flags, gold, out = {}, {}, {}
    for name in systems:
        fl = []
        rep = evaluate_execution([(e.db_id, e.sql, gc[h], name) for e, h in zip(dev, H[name])], gold_cache=gold, flags=fl)
        flags[name] = fl; out[name] = {"ex": rep.correct / max(1, rep.n - rep.gold_fail)}
        msg = f"{name:16s} EX={out[name]['ex']:.4f}"
        for ref in ("full", "crisp"):
            if ref in flags and ref != name:
                out[name][f"vs_{ref}"] = paired(flags[ref], fl); msg += f"  vs {ref} {out[name][f'vs_{ref}']}"
        print(msg, flush=True)
    ctrl = [k for k in flags if k.startswith("random_ctx")]
    avg = np.nanmean([[np.nan if x is None else x for x in flags[k]] for k in ctrl], axis=0)
    cr = np.asarray([np.nan if x is None else x for x in flags["crisp"]], float)
    d = cr - avg; d = d[~np.isnan(d)]
    rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    print(f"crisp - mean(random context over {len(ctrl)} seeds) = {d.mean():+.4f} 95%CI[{np.quantile(bs,.025):+.4f},{np.quantile(bs,.975):+.4f}]")
    json.dump({"systems": out, "flags": flags}, open(os.path.join(RES, "e2e_mechanism_qwen32b.json"), "w"))
