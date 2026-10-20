"""Hybrid linking on the hard benchmark: 32B selects columns inside the collapsed CRISP certified set, then FK closure.

  dump  : link prompts -> results/hard_hybrid_link_prompts.jsonl
  parse : --outputs <jsonl {h,text}> -> sets, SRR, generation prompts results/hard_hybrid_genprompts.jsonl
  ex    : --gen <jsonl {h,sql}> -> EX paired vs full, full_collapse and the certified set itself
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, load_examples, read_scores
from v2_frontier import combine
from v2_e2e import crossfit, prompt_of, paired
from jevschema.certify import fixed_family, covers
from jevschema.selection import closure, parse_closure
from jevschema.tokens import n_tokens, schema_prompt_text

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_llm_link.py"), encoding="utf-8").read()
_ns = {"__file__": __file__}
exec(_src.split("\nap = argparse")[0], _ns)
TMPL, parse_cols = _ns["TMPL"], _ns["parse_cols"]

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "parse", "ex"])
ap.add_argument("--outputs", default=""); ap.add_argument("--gen", default=""); ap.add_argument("--eps", type=float, default=0.02)
a = ap.parse_args()


def dedup(e, s):
    sig = {}
    for t in e.schema.tables:
        sig.setdefault(tuple(sorted(c.name.lower() for c in t.columns)), []).append(t.name)
    rep = {}
    for names in sig.values():
        r = sorted(names, key=lambda x: (len(x), x))[0]
        for nm in names:
            rep[nm] = (nm == r)
    lo = min(s.values()) if s else 0.0
    return {p: (v if rep.get(p[0], True) else v - 10.0 - abs(lo)) for p, v in s.items()}


def reps(e):
    sig = {}
    for t in e.schema.tables:
        sig.setdefault(tuple(sorted(c.name.lower() for c in t.columns)), []).append(t.name)
    return {sorted(v, key=lambda x: (len(x), x))[0] for v in sig.values()}


n = 2000
J = read_scores(os.path.join(CACHE, f"jev_fk_hard{n}.jsonl"))
exs = [e for e in load_examples(f"hard{n}") if e.ex_id in J]
bm = read_scores(os.path.join(CACHE, f"hard{n}_bm25.jsonl")); em = read_scores(os.path.join(CACHE, f"hard{n}_emb.jsonl"))
f0 = [fused_scores(e, bm[e.ex_id], em[e.ex_id]) for e in exs]
sc = [dedup(e, s) for e, s in zip(exs, combine(exs, f0, [J[e.ex_id] for e in exs], 0.02))]
v = View(exs, sc, "f1k1")
fam = fixed_family(v.r, v.n_cols)
fold = np.random.default_rng(0).permutation(len(exs)) % 2
kk = crossfit(v, fam, a.eps, list(range(len(exs))), fold)
R = [reps(e) for e in exs]
pool = [[p for p in kk[i] if p[0] in R[i]] for i in range(len(exs))]
full_c = [[(t.name, c.name) for t in e.schema.tables if t.name in r for c in t.columns] for e, r in zip(exs, R)]
M = json.load(open(os.path.join(RES, "prompts_hardc_map.json")))
tag = f"N{n}:jev_e{a.eps}_collapse"
assert [hashlib.md5(prompt_of(e, k).encode("utf-8")).hexdigest() for e, k in zip(exs, pool)] == M[tag]["h"], "pool mismatch"


def hprompt(e, cols):
    txt = schema_prompt_text(e.schema, tables=sorted({t for t, _ in cols}), columns=cols)
    return TMPL.format(db=e.db_id, schema=txt, ev=e.evidence or "", q=e.question)


def hybrid_sets(got):
    f1 = parse_closure("f1"); sets, toks = [], []
    for e, cols in zip(exs, pool):
        p = hprompt(e, cols); t = got.get(hashlib.md5(p.encode()).hexdigest(), "")
        sel = [c for c in parse_cols(t, e.schema) if c in set(cols)] or list(cols)
        rest = [x for x in cols if x not in sel]
        sets.append(closure(e.schema, sel + rest, len(sel), f1)); toks.append(n_tokens(p) + n_tokens(t))
    return sets, toks


if a.mode == "dump":
    with open(os.path.join(RES, "hard_hybrid_link_prompts.jsonl"), "w", encoding="utf-8") as f:
        for e, cols in zip(exs, pool):
            p = hprompt(e, cols)
            f.write(json.dumps({"h": hashlib.md5(p.encode()).hexdigest(), "prompt": p}, ensure_ascii=False) + "\n")
    print("dumped", len(exs), "mean pool cols", np.mean([len(c) for c in pool]))
else:
    got = {json.loads(l)["h"]: json.loads(l)["text"] for l in open(a.outputs, encoding="utf-8")}
    sets, toks = hybrid_sets(got)
    sets = [[p for p in s if p[0] in r] for s, r in zip(sets, R)]
    srr = np.mean([covers(e.gold, set(s)) for e, s in zip(exs, sets)])
    print(f"hard hybrid: SRR={srr:.4f} Cbar={np.mean([len(s) for s in sets]):.1f} link tok/q={np.mean(toks):.0f}")
    if a.mode == "parse":
        with open(os.path.join(RES, "hard_hybrid_genprompts.jsonl"), "w", encoding="utf-8") as f:
            seen = set()
            for e, s in zip(exs, sets):
                gp = prompt_of(e, s); h = hashlib.md5(gp.encode()).hexdigest()
                if h not in seen:
                    seen.add(h); f.write(json.dumps({"h": h, "prompt": gp}, ensure_ascii=False) + "\n")
        print("gen prompts", len(seen))
    else:
        from jevschema.exec_eval import evaluate_execution
        gc = {}
        for fn in (os.path.join(CACHE, "gen_hardc_qwen32b.jsonl"), a.gen):
            for l in open(fn, encoding="utf-8"):
                r = json.loads(l); gc[r["h"]] = r["sql"]
        dev = {e.ex_id: e for e in load_examples("dev")}
        systems = {"full": [None] * len(exs), "full_collapse": full_c, f"jev_e{a.eps}_collapse": pool, "hybrid": sets}
        out, flags, gold = {}, {}, {}
        for name, kept in systems.items():
            hs = [hashlib.md5(prompt_of(e, k).encode("utf-8")).hexdigest() for e, k in zip(exs, kept)]
            fl = []
            rep = evaluate_execution([(dev[e.ex_id].db_id, dev[e.ex_id].sql, gc[h], name) for e, h in zip(exs, hs)], gold_cache=gold, flags=fl)
            flags[name] = fl
            out[name] = {"ex": rep.correct / max(1, rep.n - rep.gold_fail),
                         "cbar": float(np.mean([len(k) if k else len(e.schema.columns()) for e, k in zip(exs, kept)]))}
            msg = f"{name:18s} EX={out[name]['ex']:.4f} Cbar={out[name]['cbar']:.1f}"
            for ref in ("full", "full_collapse", f"jev_e{a.eps}_collapse"):
                if ref in flags and ref != name:
                    out[name][f"vs_{ref}"] = paired(flags[ref], fl); msg += f"  vs {ref} {out[name][f'vs_{ref}']}"
            print(msg, flush=True)
        json.dump({"systems": out, "flags": flags}, open(os.path.join(RES, "e2e_hard_hybrid_qwen32b.json"), "w"))
