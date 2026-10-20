"""Hybrid linking: an LLM selects columns *inside the CRISP certified set* (Jev, eps .02), then FK closure.

  dump  : linking prompts restricted to the certified set -> results/hybrid_link_prompts.jsonl
  parse : --outputs <jsonl with {h,text}> -> sets, SRR/Cbar/linking tokens, generation prompts
"""
import argparse, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, load_examples
from v2_e2e import build_systems, prompt_of
from jevschema.certify import covers
from jevschema.selection import closure, parse_closure
from jevschema.tokens import n_tokens, schema_prompt_text

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_llm_link.py"), encoding="utf-8").read()
_ns = {"__file__": __file__}
exec(_src.split("\nap = argparse")[0], _ns)
TMPL, parse_cols = _ns["TMPL"], _ns["parse_cols"]

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "parse"]); ap.add_argument("--outputs", default="")
ap.add_argument("--pool", default="t1_e0.02")
ap.add_argument("--tier1", default="jev_t10")
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1=a.tier1, beta=0.02, route_frac=0.4), dev, hold)
pool = S[a.pool]


def hprompt(e, cols):
    txt = schema_prompt_text(e.schema, tables=sorted({t for t, _ in cols}), columns=cols)
    return TMPL.format(db=e.db_id, schema=txt, ev=e.evidence or "", q=e.question)


if a.mode == "dump":
    with open(os.path.join(RES, f"hybrid{a.tag}_link_prompts.jsonl"), "w", encoding="utf-8") as f:
        for e, cols in zip(dev, pool):
            p = hprompt(e, cols)
            f.write(json.dumps({"h": hashlib.md5(p.encode()).hexdigest(), "prompt": p}, ensure_ascii=False) + "\n")
    print("dumped", len(dev), "mean pool cols", np.mean([len(c) for c in pool]))
else:
    got = {json.loads(l)["h"]: json.loads(l)["text"] for l in open(a.outputs, encoding="utf-8")}
    f1 = parse_closure("f1")
    sets, rows = [], []
    for e, cols in zip(dev, pool):
        p = hprompt(e, cols); t = got.get(hashlib.md5(p.encode()).hexdigest(), "")
        sel = [c for c in parse_cols(t, e.schema) if c in set(cols)] or list(cols)
        fk = closure(e.schema, sel + [x for x in e.schema.columns() if x not in sel], len(sel), f1)
        sets.append(fk)
        rows.append((covers(e.gold, set(fk)), len(fk), n_tokens(p) + n_tokens(t)))
    R = np.asarray(rows, float)
    print(f"hybrid (LLM inside certified set + FK): SRR={R[:,0].mean():.4f} Cbar={R[:,1].mean():.1f} linking LLM tok/q={R[:,2].mean():.0f}")
    json.dump({"ids": [e.ex_id for e in dev], "sets": [[list(x) for x in s] for s in sets]}, open(os.path.join(RES, f"hybrid{a.tag}_sets.json"), "w"))
    with open(os.path.join(RES, f"hybrid{a.tag}_genprompts.jsonl"), "w", encoding="utf-8") as f:
        seen = set()
        for e, s in zip(dev, sets):
            gp = prompt_of(e, s); h = hashlib.md5(gp.encode()).hexdigest()
            if h not in seen:
                seen.add(h); f.write(json.dumps({"h": h, "prompt": gp}, ensure_ascii=False) + "\n")
    print("gen prompts", len(seen))
