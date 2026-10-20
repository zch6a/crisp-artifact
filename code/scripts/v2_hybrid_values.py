"""Value-aware second stage: the linker reads the certified pool rendered from OmniSQL's OFFICIAL schema text
(question-matched example values + column descriptions, data/dev_bird.json), i.e. the same evidence the generator
sees; it selects columns inside the pool, then FK closure (as v2_hybrid.py).  Post-hoc improvement, reported as such.
  dump  : -> results/hybridv{tag}_link_prompts.jsonl
  parse : --outputs <{h,text}> -> hybridv{tag}_sets.json, hybridv{tag}_genprompts.jsonl, tasks_hybridv{tag}.jsonl"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, DATA, load_examples
from v2_e2e import build_systems, prompt_of
from jevschema.certify import covers
from jevschema.selection import closure, parse_closure
from jevschema.tokens import n_tokens
import official_prune_lib as op

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_llm_link.py"), encoding="utf-8").read()
_ns = {"__file__": __file__}
exec(_src.split("\nap = argparse")[0], _ns)
TMPL, parse_cols = _ns["TMPL"], _ns["parse_cols"]

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "parse"]); ap.add_argument("--outputs", default="")
ap.add_argument("--pool", default="t1_e0.02"); ap.add_argument("--tier1", default="jev_t10"); ap.add_argument("--tag", default="_ds")
a = ap.parse_args()
dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1=a.tier1, beta=0.02, route_frac=0.4), dev, hold)
pool = [list(e.schema.columns()) for e in dev] if a.pool == "full" else S[a.pool]   # "full": no certificate (ablation)
OFF = [d["input_seq"] for d in json.load(open(os.path.join(DATA, "omnisql_dev_bird.json"), encoding="utf-8"))]
assert len(OFF) == len(dev)


def official_schema(i, cols):
    blocks = {}
    for t, c in cols:
        blocks.setdefault(t.lower(), []).append(f"  {c.lower()} ")
    p, _ = op.prune(OFF[i], blocks)
    s = p.index("Database Schema:\n") + len("Database Schema:\n")
    return p[s:p.index("\nThis schema describes")].strip()


def hprompt(i, e, cols):
    return TMPL.format(db=e.db_id, schema=official_schema(i, cols), ev=e.evidence or "", q=e.question)


if a.mode == "dump":
    with open(os.path.join(RES, f"hybridv{a.tag}_link_prompts.jsonl"), "w", encoding="utf-8") as f:
        for i, (e, cols) in enumerate(zip(dev, pool)):
            p = hprompt(i, e, cols)
            f.write(json.dumps({"h": hashlib.md5(p.encode()).hexdigest(), "prompt": p}, ensure_ascii=False) + "\n")
    print("dumped", len(dev), "mean pool cols", np.mean([len(c) for c in pool]))
else:
    got = {json.loads(l)["h"]: json.loads(l)["text"] for l in open(a.outputs, encoding="utf-8")}
    f1 = parse_closure("f1")
    sets, rows, fallback = [], [], 0
    for i, (e, cols) in enumerate(zip(dev, pool)):
        p = hprompt(i, e, cols); t = got.get(hashlib.md5(p.encode()).hexdigest(), "")
        sel = [c for c in parse_cols(t, e.schema) if c in set(cols)]
        if not sel:
            sel, fallback = list(cols), fallback + 1
        fk = closure(e.schema, sel + [x for x in e.schema.columns() if x not in sel], len(sel), f1)
        sets.append(fk)
        rows.append((covers(e.gold, set(fk)), len(fk), n_tokens(p) + n_tokens(t)))
    R = np.asarray(rows, float)
    print(f"value-aware hybrid: SRR={R[:,0].mean():.4f} Cbar={R[:,1].mean():.1f} linking tok/q={R[:,2].mean():.0f} "
          f"fallback={fallback} missing_outputs={sum(1 for i,(e,c) in enumerate(zip(dev,pool)) if hashlib.md5(hprompt(i,e,c).encode()).hexdigest() not in got)}")
    json.dump({"ids": [e.ex_id for e in dev], "sets": [[list(x) for x in s] for s in sets]},
              open(os.path.join(RES, f"hybridv{a.tag}_sets.json"), "w"))
    seen = set()
    with open(os.path.join(RES, f"hybridv{a.tag}_genprompts.jsonl"), "w", encoding="utf-8") as f, \
            open(os.path.join(RES, f"tasks_hybridv{a.tag}.jsonl"), "w", encoding="utf-8") as ft:
        for i, (e, s) in enumerate(zip(dev, sets)):
            gp = prompt_of(e, s); h = hashlib.md5(gp.encode()).hexdigest()
            ft.write(json.dumps({"sys": f"hybridv{a.tag}", "i": i, "h": h, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
            if h not in seen:
                seen.add(h); f.write(json.dumps({"h": h, "prompt": gp}, ensure_ascii=False) + "\n")
    print("gen prompts", len(seen))
