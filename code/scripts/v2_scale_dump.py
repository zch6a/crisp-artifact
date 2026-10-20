"""Prompts for end-to-end EX on the scaled benchmark (BIRD-Dev + distractor tables, stride-3 subset scored by Jev).

Systems per size N: full (entire merged schema), Jev+FK-gate certified sets (eps .02/.05, 2-fold cross-fitted
split conformal on this subset), oracle.  Gold SQL still executes on the original sqlite DB (distractor tables are
prompt-only), so a query that touches a distractor table simply fails.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, load_examples, read_scores
from v2_frontier import combine
from v2_e2e import crossfit, prompt_of
from jevschema.certify import fixed_family

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="500,2000")
ap.add_argument("--src", default="jev_fk2_scale{n}")
ap.add_argument("--kind", default="scale", choices=["scale", "hard"])
ap.add_argument("--out", default="prompts_scale")
ap.add_argument("--dedup", action="store_true", help="duplicate-aware ranking (see v2_hard_dedup.py)")
ap.add_argument("--collapse", action="store_true", help="drop non-representative copy tables from every kept set "
                "(coverage unchanged: gold lives on representatives); adds full_collapse = all representative tables")
ap.add_argument("--eps", default="0.02,0.05")
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


def representatives(e):
    sig = {}
    for t in e.schema.tables:
        sig.setdefault(tuple(sorted(c.name.lower() for c in t.columns)), []).append(t.name)
    return {sorted(v, key=lambda x: (len(x), x))[0] for v in sig.values()}


def collapse(e, k):
    reps = representatives(e)
    if k is None:
        return [(t.name, c.name) for t in e.schema.tables if t.name in reps for c in t.columns]
    return [p for p in k if p[0] in reps]


rows, seen, sysmap = [], set(), {}
for n in [int(x) for x in a.sizes.split(",")]:
    J = read_scores(os.path.join(CACHE, a.src.format(n=n) + ".jsonl"))
    if not J:
        J = read_scores(os.path.join(CACHE, f"jev_fk_{a.kind}{n}.jsonl"))
    exs = [e for e in load_examples(f"{a.kind}{n}") if e.ex_id in J]
    bm = read_scores(os.path.join(CACHE, f"{a.kind}{n}_bm25.jsonl")); em = read_scores(os.path.join(CACHE, f"{a.kind}{n}_emb.jsonl"))
    f0 = [fused_scores(e, bm[e.ex_id], em[e.ex_id]) for e in exs]
    sc = combine(exs, f0, [J[e.ex_id] for e in exs], 0.02)
    if a.dedup:
        sc = [dedup(e, s) for e, s in zip(exs, sc)]
    v = View(exs, sc, "f1k1")
    fam = fixed_family(v.r, v.n_cols)
    fold = np.random.default_rng(0).permutation(len(exs)) % 2
    S = {"full": [None] * len(exs)}
    for eps in [float(x) for x in a.eps.split(",")]:
        kk = crossfit(v, fam, eps, list(range(len(exs))), fold)
        S[f"jev_e{eps}"] = [kk[i] for i in range(len(exs))]
    if a.collapse:
        S = {("full_collapse" if nm == "full" else nm + "_collapse"): ([collapse(e, k) for e, k in zip(exs, kept)] if nm != "oracle" else kept)
             for nm, kept in S.items()} | {"full": S["full"]}
    orc = []
    for e in exs:
        g = set(e.gold.flat_pairs()); tabs = {t for t, _ in g}
        for fk in e.schema.fks:
            if fk.src_table in tabs and fk.dst_table in tabs:
                g.add((fk.src_table, fk.src_col)); g.add((fk.dst_table, fk.dst_col))
        orc.append(sorted(g))
    S["oracle"] = orc
    for name, kept in S.items():
        hs = []
        for e, k in zip(exs, kept):
            p = prompt_of(e, k); h = hashlib.md5(p.encode("utf-8")).hexdigest(); hs.append(h)
            if h not in seen:
                seen.add(h); rows.append({"h": h, "prompt": p})
        sysmap[f"N{n}:{name}"] = {"ids": [e.ex_id for e in exs], "h": hs,
                                  "cbar": float(np.mean([len(k) if k else len(e.schema.columns()) for e, k in zip(exs, kept)])),
                                  "srr": float(np.mean([1.0 if k is None or all(req & set(k) for req in e.gold.requirements) else 0.0 for e, k in zip(exs, kept)]))}
    print(f"N={n}: {len(exs)} questions; " + ", ".join(f"{k}: Cbar {sysmap[f'N{n}:{k}']['cbar']:.1f} SRR {sysmap[f'N{n}:{k}']['srr']:.3f}" for k in S), flush=True)
with open(os.path.join(RES, a.out + ".jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(sysmap, open(os.path.join(RES, a.out + "_map.json"), "w"))
print(f"{len(rows)} unique prompts")
