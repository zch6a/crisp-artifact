"""Relevance-tiered certified prompts: the certified set is rendered unchanged (coverage guarantee untouched) and one
extra line lists its K highest-scoring columns ("most likely needed"), giving the generator a ranked focus inside the
certified context without dropping anything.  Applied uniformly to CRISP eps=0.05 and Mondrian(3, hard 0.02).
  dump -> results/prompts_tiered.jsonl + prompts_tiered_map.json  (K from --k, default 8)
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from v2_e2e import build_systems
from jevschema.tokens import build_gen_prompt, schema_prompt_text

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=8); a = ap.parse_args()
dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
bm, em = lexical("dev"); jd = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl"))
sc = combine(dev, [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev], [jd.get(e.ex_id, {}) for e in dev], 0.02)
# Mondrian(3, hard 0.02) sets: recovered from the dumped prompts' map is not possible, so rebuild via the Mondrian script's logic
import runpy
mon = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_mondrian.py"), run_name="mondrian_import")
sets = {"t1_e0.05": S["t1_e0.05"], "mondrian3_hard0.02": mon["S"]["mondrian3_hard0.02"]}


def tiered_prompt(e, kept, s):
    txt = schema_prompt_text(e.schema, tables=sorted({t for t, _ in kept}), columns=kept)
    top = sorted(kept, key=lambda p: -s.get(p, -9))[: a.k]
    txt += "\n-- Most likely needed columns (ranked): " + ", ".join(f"{t}.{c}" for t, c in top)
    return build_gen_prompt(e.db_id, txt, e.question, e.evidence or "")


rows, mp, seen = [], {}, set()
for name, kept in sets.items():
    hs = []
    for i, (e, k) in enumerate(zip(dev, kept)):
        p = tiered_prompt(e, k, sc[i]); h = hashlib.md5(p.encode("utf-8")).hexdigest(); hs.append(h)
        if h not in seen:
            seen.add(h); rows.append({"h": h, "prompt": p})
    mp[f"tiered{a.k}:" + name] = hs
with open(os.path.join(RES, "prompts_tiered.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"ids": [e.ex_id for e in dev], "systems": mp}, open(os.path.join(RES, "prompts_tiered_map.json"), "w"))
print(f"{len(rows)} tiered prompts")
