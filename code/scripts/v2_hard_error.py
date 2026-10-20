"""Error analysis of end-to-end EX on the hard benchmark (self-copy distractors), per question vs full schema.

Rebuilds the exact sets of v2_scale_dump.py --kind hard --dedup (same fold seed), loads the generated SQL and the
per-question flags of e2e_hard_qwen32b.json, and classifies every win/loss:
  miss      : certified set lost a required gold column (coverage failure)
  copy      : set covers the gold, but the SQL references a copy table (<base>_<suffix>), which does not exist in the DB
  exec_err  : SQL failed for another reason
  wrong     : SQL executes, result differs
Also reports which base tables are shown / hidden and how often the kept set contains copies at all.
"""
import argparse, hashlib, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, load_examples, read_scores
from v2_frontier import combine
from v2_e2e import crossfit, prompt_of
from jevschema.certify import fixed_family
from jevschema.scale_hard import SUFFIXES

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--gen", default="gen_hard_qwen32b.jsonl")
ap.add_argument("--res", default="e2e_hard_qwen32b.json")
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


n = a.n
J = read_scores(os.path.join(CACHE, f"jev_fk_hard{n}.jsonl"))
exs = [e for e in load_examples(f"hard{n}") if e.ex_id in J]
bm = read_scores(os.path.join(CACHE, f"hard{n}_bm25.jsonl")); em = read_scores(os.path.join(CACHE, f"hard{n}_emb.jsonl"))
f0 = [fused_scores(e, bm[e.ex_id], em[e.ex_id]) for e in exs]
sc = [dedup(e, s) for e, s in zip(exs, combine(exs, f0, [J[e.ex_id] for e in exs], 0.02))]
v = View(exs, sc, "f1k1")
fam = fixed_family(v.r, v.n_cols)
fold = np.random.default_rng(0).permutation(len(exs)) % 2
S = {"full": [None] * len(exs)}
for eps in (0.02, 0.05):
    kk = crossfit(v, fam, eps, list(range(len(exs))), fold)
    S[f"jev_e{eps}"] = [kk[i] for i in range(len(exs))]

M = json.load(open(os.path.join(RES, "prompts_hard_map.json")))
for name in S:
    hs = [hashlib.md5(prompt_of(e, k).encode("utf-8")).hexdigest() for e, k in zip(exs, S[name])]
    assert hs == M[f"N{n}:{name}"]["h"], f"set reconstruction mismatch for {name}"
print("sets reconstructed exactly (hashes match the dump)", flush=True)

gc = {}
for l in open(os.path.join(CACHE, a.gen), encoding="utf-8"):
    r = json.loads(l); gc[r["h"]] = r["sql"]
R = json.load(open(os.path.join(RES, a.res)))
F = R["flags"]
base_db = load_examples("dev")
base_tables = {e.ex_id: {t.name.lower() for t in e.schema.tables} for e in base_db}
sfx_re = re.compile(r"\b([A-Za-z0-9_]+?)_(" + "|".join(SUFFIXES) + r")\d*\b", re.I)


def uses_copy(sql, eid):
    for m in sfx_re.finditer(sql or ""):
        if m.group(0).lower() not in base_tables[eid]:
            return True
    return False


def covered(e, k):
    return k is None or all(req & set(k) for req in e.gold.requirements)


def n_copy_tables(e, k):
    if k is None:
        return None
    tabs = {t for t, _ in k}
    return sum(1 for t in tabs if t.lower() not in base_tables[e.ex_id])


full_h = M[f"N{n}:full"]["h"]
out = {}
for name in ("jev_e0.02", "jev_e0.05"):
    hs = M[f"N{n}:{name}"]["h"]
    fa, fb = F[f"N{n}:full"], F[f"N{n}:{name}"]
    cat = {"loss": Counter(), "win": Counter()}
    examples = {"loss": [], "win": []}
    for i, e in enumerate(exs):
        x, y = fa[i], fb[i]
        if x is None or y is None or x == y:
            continue
        kind = "loss" if x > y else "win"
        bad_h = hs[i] if kind == "loss" else full_h[i]      # the side that got it wrong
        bad_set = S[name][i] if kind == "loss" else None
        sql = gc[bad_h]
        if kind == "loss" and not covered(e, bad_set):
            c = "miss"
        elif uses_copy(sql, e.ex_id):
            c = "copy"
        elif not sql.strip():
            c = "empty"
        else:
            c = "wrong_or_err"
        cat[kind][c] += 1
        if len(examples[kind]) < 12:
            examples[kind].append({"id": e.ex_id, "cat": c, "q": e.question[:120], "gold": e.sql[:200], "pred": sql[:200],
                                   "set_size": len(bad_set) if bad_set else None})
    sizes = [len(k) for k in S[name]]
    ncopy = [n_copy_tables(e, k) for e, k in zip(exs, S[name])]
    cov = [covered(e, k) for e, k in zip(exs, S[name])]
    ex_cov = np.mean([fb[i] for i in range(len(exs)) if cov[i] and fb[i] is not None])
    ex_cov_full = np.mean([fa[i] for i in range(len(exs)) if cov[i] and fa[i] is not None])
    print(f"\n== {name}: Cbar={np.mean(sizes):.1f}  SRR={np.mean(cov):.3f}  sets with copy tables: "
          f"{np.mean([c > 0 for c in ncopy]):.3f} (mean {np.mean(ncopy):.2f} copy tables)")
    print(f"   losses vs full: {dict(cat['loss'])}   wins vs full: {dict(cat['win'])}")
    print(f"   EX on covered questions: {name}={ex_cov:.4f} full={ex_cov_full:.4f} (n={sum(cov)})")
    out[name] = {"loss": dict(cat["loss"]), "win": dict(cat["win"]), "examples": examples,
                 "ex_covered": float(ex_cov), "ex_covered_full": float(ex_cov_full), "srr": float(np.mean(cov)),
                 "copy_set_frac": float(np.mean([c > 0 for c in ncopy]))}
full_copy = sum(uses_copy(gc[h], e.ex_id) for h, e in zip(full_h, exs))
print(f"\nfull schema: SQL referencing a copy table in {full_copy}/{len(exs)} questions")
out["full_copy_refs"] = int(full_copy)
json.dump(out, open(os.path.join(RES, "hard_error.json"), "w"), indent=1)
