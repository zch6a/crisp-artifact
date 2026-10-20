"""Difficulty-conditional (Mondrian) certification to protect hard questions.

Marginal split conformal guarantees E[coverage] >= 1-eps over all questions, so hard questions (BIRD "challenging":
coverage 0.931 at eps=0.05) are averaged away.  Mondrian conformal calibrates separately inside bins of a *predicted*
difficulty (label-free at test time), giving a coverage guarantee inside every bin.
  difficulty predictor: heteroscedastic ridge on reader-score shape features, fitted on the BIRD-train holdout only
  bins: quantiles of the predicted difficulty over the test questions (no labels used)
Variants (eps = 0.05 unless stated): marginal | mondrian K=2,3,4 | mondrian K=3 with eps 0.02 in the hardest bin |
BIRD-label Mondrian (analysis only, not deployable).  Reports overall and per-BIRD-stratum SRR / Cbar and dumps
generation prompts for EX.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from v2_e2e import crossfit, prompt_of
from jevschema.certify import fixed_family
from jevschema.difficulty import Difficulty, shape_features

dev, hold = load_examples("dev"), load_examples("holdout")
bm, em = lexical("dev"); hbm, hem = lexical("holdout")
jd = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl")); jh = read_scores(os.path.join(CACHE, "jev_t10_holdout.jsonl"))
hold = [e for e in hold if e.ex_id in jh]
f_dev = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
f_hold = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
v = View(dev, combine(dev, f_dev, [jd.get(e.ex_id, {}) for e in dev], 0.02), "f1k1")
vh = View(hold, combine(hold, f_hold, [jh.get(e.ex_id, {}) for e in hold], 0.02), "f1k1")


def feats(view):
    return np.asarray([shape_features(view.sorted_scores(i), len(e.schema.columns()), len(e.schema.tables),
                                      len(e.question.split()), bool(e.evidence)) for i, e in enumerate(view.exs)])


D = Difficulty().fit(feats(vh), np.log(np.maximum(1, vh.r)))
mu = D.predict(feats(v))[0]
print(f"difficulty model R2(holdout)={D.r2:.3f}; corr(pred, log r_dev)={np.corrcoef(mu, np.log(np.maximum(1, v.r)))[0,1]:.3f}")
fam = fixed_family(v.r, v.n_cols)
n = len(dev); all_i = list(range(n))
fold = np.random.default_rng(0).permutation(n) % 2
strata = [e.difficulty for e in dev]


def mondrian(bins, eps_of_bin):
    out = {}
    for b in sorted(set(bins)):
        idx = [i for i in all_i if bins[i] == b]
        out.update(crossfit(v, fam, eps_of_bin(b), idx, fold))
    return [out[i] for i in all_i]


def qbins(K):
    edges = np.quantile(mu, np.linspace(0, 1, K + 1)[1:-1])
    return [int(np.searchsorted(edges, m, side="right")) for m in mu]


S = {"marginal_e0.05": [crossfit(v, fam, 0.05, all_i, fold)[i] for i in all_i],
     "marginal_e0.02": [crossfit(v, fam, 0.02, all_i, fold)[i] for i in all_i]}
for K in (2, 3, 4):
    S[f"mondrian{K}_e0.05"] = mondrian(qbins(K), lambda b: 0.05)
b3 = qbins(3)
S["mondrian3_hard0.02"] = mondrian(b3, lambda b: 0.02 if b == 2 else 0.05)
S["mondrian3_hard0.01"] = mondrian(b3, lambda b: 0.01 if b == 2 else 0.05)
S["label_mondrian_e0.05 (analysis)"] = mondrian(strata, lambda b: 0.05)

cov = lambda i, k: all(req & set(k) for req in dev[i].gold.requirements)
res = {}
print(f"\n{'system':34s} {'SRR':>6s} {'Cbar':>6s} | " + " | ".join(f"{s[:5]} SRR/Cbar" for s in ("simple", "moderate", "challenging")))
for name, kept in S.items():
    row = {"srr": float(np.mean([cov(i, k) for i, k in enumerate(kept)])), "cbar": float(np.mean([len(k) for k in kept]))}
    parts = []
    for s in ("simple", "moderate", "challenging"):
        ii = [i for i in all_i if strata[i] == s]
        row[s] = [float(np.mean([cov(i, kept[i]) for i in ii])), float(np.mean([len(kept[i]) for i in ii]))]
        parts.append(f"{row[s][0]:.3f}/{row[s][1]:5.1f}")
    res[name] = row
    print(f"{name:34s} {row['srr']:6.3f} {row['cbar']:6.1f} | " + " | ".join(parts))
json.dump(res, open(os.path.join(RES, "mondrian.json"), "w"), indent=1)
seen = set()
with open(os.path.join(RES, "prompts_mondrian.jsonl"), "w", encoding="utf-8") as f:
    mp = {}
    for name, kept in S.items():
        hs = []
        for e, k in zip(dev, kept):
            p = prompt_of(e, k); h = hashlib.md5(p.encode("utf-8")).hexdigest(); hs.append(h)
            if h not in seen:
                seen.add(h); f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
        mp[name] = hs
json.dump({"ids": [e.ex_id for e in dev], "systems": mp}, open(os.path.join(RES, "prompts_mondrian_map.json"), "w"))
print(f"{len(seen)} unique prompts dumped")
