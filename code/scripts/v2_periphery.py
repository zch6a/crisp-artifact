"""'Certified core + periphery' prompts: the certified set is rendered in full (types, keys, FKs); every pruned column is
still listed by name only, one compact line per table, so the model sees the whole schema structure (as in its
training distribution) while the detailed context stays focused.  Coverage is unchanged (a superset of names only).
Systems: t1_e0.05, mondrian3_hard0.02 (from prompts_mondrian map sets), hybrid (32B-linked sets).
  dump -> results/prompts_periph.jsonl + prompts_periph_map.json
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, load_examples
from v2_e2e import build_systems
from jevschema.tokens import build_gen_prompt, schema_prompt_text

dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
Hy = json.load(open(os.path.join(RES, "hybrid_sets.json")))
sets = {"t1_e0.05": S["t1_e0.05"], "hybrid": [[tuple(x) for x in s] for s in Hy["sets"]]}


def periph_prompt(e, kept):
    ks = {(t.lower(), c.lower()) for t, c in kept}
    core = schema_prompt_text(e.schema, tables=sorted({t for t, _ in kept}), columns=kept)
    lines = []
    for t in e.schema.tables:
        rest = [c.name for c in t.columns if (t.name.lower(), c.name.lower()) not in ks]
        if rest:
            lines.append(f"--   {t.name}({', '.join(rest)})")
    txt = core + ("\n-- Other columns in this database (names only, not detailed):\n" + "\n".join(lines) if lines else "")
    return build_gen_prompt(e.db_id, txt, e.question, e.evidence or "")


rows, mp, seen = [], {}, set()
for name, kept in sets.items():
    hs = []
    for e, k in zip(dev, kept):
        p = periph_prompt(e, k); h = hashlib.md5(p.encode("utf-8")).hexdigest(); hs.append(h)
        if h not in seen:
            seen.add(h); rows.append({"h": h, "prompt": p})
    mp["periph:" + name] = hs
with open(os.path.join(RES, "prompts_periph.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"ids": [e.ex_id for e in dev], "systems": mp}, open(os.path.join(RES, "prompts_periph_map.json"), "w"))
from jevschema.tokens import n_tokens
print(f"{len(rows)} prompts; mean tokens periph:t1_e0.05 = {np.mean([n_tokens(r['prompt']) for r in rows[:1534]]):.0f}")
