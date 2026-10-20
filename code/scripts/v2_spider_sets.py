"""Spider-Dev certified kept sets, same construction as BIRD (v2_e2e.build_systems): tier-1 = Jev reader fused with
the lexical scorer (beta=0.02), closure f1k1, fixed family, 2-fold cross-fitted split conformal, eps in {0.05, 0.1};
lexical-only control t0 (closure f1p5s15k1).  -> results/spider_sets.json {ids, db, question, sets{sys: [[t,c],..]}}"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_cascade import tier1_scores
from v2_e2e import crossfit
from jevschema.certify import fixed_family

exs = load_examples("spider")
bm, em = lexical("spider")
Jv = read_scores(os.path.join(CACHE, "jev_t10_spider.jsonl"))
exs = [e for e in exs if e.ex_id in Jv]
s0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in exs]
s1 = tier1_scores(exs, s0, [Jv[e.ex_id] for e in exs], 0.02)
v0, v1 = View(exs, s0, "f1p5s15k1"), View(exs, s1, "f1k1")
f0, f1 = fixed_family(v0.r, v0.n_cols), fixed_family(v1.r, v1.n_cols)
fold = np.random.default_rng(0).permutation(len(exs)) % 2
all_i = list(range(len(exs)))
S = {}
for eps in (0.05, 0.1):
    k1 = crossfit(v1, f1, eps, all_i, fold); S[f"jev_t10:t1_e{eps}"] = [sorted(map(list, k1[i])) for i in all_i]
    k0 = crossfit(v0, f0, eps, all_i, fold); S[f"t0_e{eps}"] = [sorted(map(list, k0[i])) for i in all_i]
for k, v in S.items():
    cov = np.mean([set(e.gold.flat_pairs()) <= {tuple(x) for x in s} for e, s in zip(exs, v)])
    print(f"{k:22s} cbar={np.mean([len(s) for s in v]):.1f} / full {np.mean([len(e.schema.columns()) for e in exs]):.1f}  gold-cov={cov:.3f}")
json.dump({"ids": [e.ex_id for e in exs], "db": [e.db_id for e in exs], "question": [e.question for e in exs],
           "sql": [e.sql for e in exs], "sets": S}, open(os.path.join(RES, "spider_sets.json"), "w"))
