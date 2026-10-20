"""Hard benchmark, duplicate-aware certification done properly: canonicalize BEFORE ranking and calibration.

Canonical schema = one representative per group of tables with identical column-name signatures (shortest name),
FKs restricted to representatives.  Gold lives on representatives, so this is a pure schema-normalization step
(an assumption we state: the canonical copy has the shortest name).  CRISP (Jev tier-1, closure f1k1, fixed family,
2-fold cross-fitted split conformal, the same eps grid as BIRD) is then run on the canonical view.
Systems (all 1534 questions that Jev scored on hard2000):
  full            the raw 2000+-column schema (reference; long prompts)
  full_canon      the whole canonical schema
  crisp_e{eps}    certified sets on the canonical view, eps in 0.02/0.05/0.10  (eps=0.05 is the primary setting)
  crisp_e0.05_dupnote  same set + one line naming the copy tables of every kept table (tells the model copies exist)
  oracle
  dump : prompts -> results/prompts_hardcanon.jsonl + _map.json      ex : --gen <jsonl> -> EX, paired tests
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, load_examples, read_scores
from v2_frontier import combine
from v2_e2e import crossfit, prompt_of, paired
from jevschema.certify import fixed_family
from jevschema.data_loader import Example
from jevschema.schema_graph import SchemaGraph
from jevschema.tokens import build_gen_prompt, schema_prompt_text

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "ex"]); ap.add_argument("--gen", default="")
a = ap.parse_args()
n = 2000
J = read_scores(os.path.join(CACHE, f"jev_fk_hard{n}.jsonl"))
exs = [e for e in load_examples(f"hard{n}") if e.ex_id in J]
bm = read_scores(os.path.join(CACHE, f"hard{n}_bm25.jsonl")); em = read_scores(os.path.join(CACHE, f"hard{n}_emb.jsonl"))


def groups(e):
    sig = {}
    for t in e.schema.tables:
        sig.setdefault(tuple(sorted(c.name.lower() for c in t.columns)), []).append(t.name)
    return {sorted(v, key=lambda x: (len(x), x))[0]: sorted(x for x in v if x != sorted(v, key=lambda y: (len(y), y))[0])
            for v in sig.values()}


def canon(e):
    g = groups(e); reps = set(g)
    sch = SchemaGraph(db_id=e.schema.db_id, tables=[t for t in e.schema.tables if t.name in reps],
                      fks=[f for f in e.schema.fks if f.src_table in reps and f.dst_table in reps])
    e2 = Example(ex_id=e.ex_id, db_id=e.db_id, question=e.question, sql=e.sql, schema=sch, evidence=e.evidence,
                 difficulty=e.difficulty)
    e2.gold = e.gold
    return e2, g


C = [canon(e) for e in exs]
cex = [c for c, _ in C]; G = [g for _, g in C]
reps = [set(g) for g in G]
f0 = [{p: v for p, v in fused_scores(e, bm[e.ex_id], em[e.ex_id]).items() if p[0] in r} for e, r in zip(exs, reps)]
j1 = [{p: v for p, v in J[e.ex_id].items() if (p[0] if isinstance(p, tuple) else p.split("|")[0]) in r} for e, r in zip(exs, reps)]
v = View(cex, combine(cex, f0, j1, 0.02), "f1k1")
fam = fixed_family(v.r, v.n_cols)
fold = np.random.default_rng(0).permutation(len(cex)) % 2
S = {"full": [None] * len(exs), "full_canon": [None] * len(cex)}
for eps in (0.02, 0.05, 0.10):
    kk = crossfit(v, fam, eps, list(range(len(cex))), fold)
    S[f"crisp_e{eps}"] = [kk[i] for i in range(len(cex))]
orc = []
for e in cex:
    gg = set(e.gold.flat_pairs()); tabs = {t for t, _ in gg}
    for fk in e.schema.fks:
        if fk.src_table in tabs and fk.dst_table in tabs:
            gg.add((fk.src_table, fk.src_col)); gg.add((fk.dst_table, fk.dst_col))
    orc.append(sorted(gg))
S["oracle"] = orc


def dupnote_prompt(e, k, g):
    tabs = sorted({t for t, _ in k})
    txt = schema_prompt_text(e.schema, tables=tabs, columns=k)
    notes = [f"-- {t} also exists as identical copies: {', '.join(g[t][:6])}{' ...' if len(g[t]) > 6 else ''} (same columns; use {t})"
             for t in tabs if g.get(t)]
    return build_gen_prompt(e.db_id, txt + ("\n" + "\n".join(notes) if notes else ""), e.question, e.evidence or "")


P = {}
for name, kept in S.items():
    base = exs if name == "full" else cex
    P[name] = [prompt_of(e, k) for e, k in zip(base, kept)]
P["crisp_e0.05_dupnote"] = [dupnote_prompt(e, k, g) for e, k, g in zip(cex, S["crisp_e0.05"], G)]
H = {name: [hashlib.md5(p.encode("utf-8")).hexdigest() for p in ps] for name, ps in P.items()}
srr = {name: float(np.mean([k is None or all(req & set(k) for req in e.gold.requirements) for e, k in zip(cex, S.get(name, S["crisp_e0.05"]))]))
       for name in P}
cbar = {name: float(np.mean([len(k) if k is not None else len(e.schema.columns()) for e, k in
                             zip(exs if name == "full" else cex, S.get(name, S["crisp_e0.05"]))])) for name in P}
if a.mode == "dump":
    seen = set()
    with open(os.path.join(RES, "prompts_hardcanon.jsonl"), "w", encoding="utf-8") as f:
        for name, ps in P.items():
            for p, h in zip(ps, H[name]):
                if h not in seen:
                    seen.add(h); f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
    json.dump({"ids": [e.ex_id for e in exs], "h": H, "srr": srr, "cbar": cbar}, open(os.path.join(RES, "prompts_hardcanon_map.json"), "w"))
    print(f"{len(exs)} questions, {len(seen)} unique prompts")
    for name in P:
        print(f"  {name:22s} Cbar={cbar[name]:7.1f} SRR={srr[name]:.3f}")
else:
    from jevschema.exec_eval import evaluate_execution
    gc = {}
    for fn in [x for x in a.gen.split(",") if x]:
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); gc[r["h"]] = r["sql"]
    dev = {e.ex_id: e for e in load_examples("dev")}
    flags, gold, out = {}, {}, {}
    for name in P:
        miss = sum(h not in gc for h in H[name])
        if miss:
            print(f"{name}: {miss} generations missing, skipped"); continue
        fl = []
        rep = evaluate_execution([(dev[e.ex_id].db_id, dev[e.ex_id].sql, gc[h], name) for e, h in zip(exs, H[name])], gold_cache=gold, flags=fl)
        flags[name] = fl
        out[name] = {"ex": rep.correct / max(1, rep.n - rep.gold_fail), "cbar": cbar[name], "srr": srr[name]}
        msg = f"{name:22s} EX={out[name]['ex']:.4f} Cbar={cbar[name]:7.1f} SRR={srr[name]:.3f}"
        for ref in ("full", "full_canon"):
            if ref in flags and ref != name:
                out[name][f"vs_{ref}"] = paired(flags[ref], fl); msg += f"  vs {ref} {out[name][f'vs_{ref}']}"
        print(msg, flush=True)
    json.dump({"systems": out, "flags": flags, "ids": [e.ex_id for e in exs]}, open(os.path.join(RES, "e2e_hardcanon_qwen32b.json"), "w"))
