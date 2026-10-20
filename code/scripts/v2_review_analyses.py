"""Analyses requested by the reviewers (all from existing per-question results; no new generation).
  1. requirement-set construction statistics + a 100-question audit sample
  2. database-clustered bootstrap (11 BIRD-Dev databases) for the headline claims
  3. single-stage pooled test including OmniSQL (6 generators)
  4. matched-size comparison: certified budget vs fixed top-k (same reader, same closure), coverage and size
  5. end-to-end token cost per question (selector input+output, generator input)
  6. rerun replication of the CRISP-vs-full difference (two runs of the same API generators)
  7. corrected two-level (per-database) guarantee: PAC per-database thresholds, leave-one-database-out check
-> results/review_analyses.json, results/requirement_audit_sample.jsonl
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import binom
from v2data import RES, CACHE, BOX, View, load_examples
from v2_e2e import build_systems
from jevschema.certify import covers, kept_at

B = os.path.join(BOX, "")
dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
N = len(dev)
db = [e.db_id for e in dev]
out = {}
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. requirement sets
g = [e.gold for e in dev]
stats = {"n": N, "parse_ok_sqlglot": float(np.mean([x.parse_ok for x in g])),
         "avg_requirements": float(np.mean([len(x.requirements) for x in g])),
         "frac_with_ambiguous_requirement": float(np.mean([x.ambiguous > 0 for x in g])),
         "frac_with_unresolved_reference": float(np.mean([x.unresolved > 0 for x in g])),
         "avg_gold_columns": float(np.mean([len(x.flat_pairs()) for x in g])),
         "avg_gold_tables": float(np.mean([len(x.tables) for x in g]))}
out["requirements"] = stats
print("requirements:", stats)
idx = rng.choice(N, 100, replace=False)
with open(os.path.join(RES, "requirement_audit_sample.jsonl"), "w", encoding="utf-8") as f:
    for i in sorted(idx):
        e = dev[i]
        f.write(json.dumps({"i": int(i), "db": e.db_id, "question": e.question, "evidence": e.evidence, "gold_sql": e.sql,
                            "requirements": [sorted(map(list, r)) for r in e.gold.requirements],
                            "ambiguous": e.gold.ambiguous, "unresolved": e.gold.unresolved}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- helpers
def load(fn):
    return json.load(open(B + fn))


def pooled(entries, cluster="question", n_boot=10000):
    rows = {}
    for F, fk, sk in entries:
        for i in range(N):
            if F[fk][i] is not None and i < len(F[sk]) and F[sk][i] is not None:
                rows.setdefault(i, []).append(F[sk][i] - F[fk][i])
    qs = sorted(rows)
    if cluster == "question":
        Sv = np.array([sum(rows[q]) for q in qs]); Cv = np.array([len(rows[q]) for q in qs])
    else:
        dbs = sorted({db[q] for q in qs})
        Sv = np.array([sum(sum(rows[q]) for q in qs if db[q] == d) for d in dbs])
        Cv = np.array([sum(len(rows[q]) for q in qs if db[q] == d) for d in dbs])
    ix = rng.integers(0, len(Sv), (n_boot, len(Sv))); bs = Sv[ix].sum(1) / Cv[ix].sum(1)
    d = Sv.sum() / Cv.sum()
    return {"diff": float(d), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "p": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))), "units": int(len(Sv))}


GENS5 = ["dsv4flash", "gpt56luna", "llama1530", "qwen2.5-32b-awq", "qwen2.5-coder_7b"]
single5 = [(load(f"flags_all2_{x}.json"), "full", "jev_t10:t1_e0.05") for x in GENS5]
R1 = load("flags_r1.json")
single6 = single5 + [(R1, "tpl:full", "tpl:jev_t10:t1_e0.05")]
two5 = [(R1, "tpl:full", "tpl:hybrid_ds"), (load("flags_ds_hyb.json"), "full", "hybrid_ds"),
        (load("flags_q7b_hyb.json"), "full", "hybrid_ds"), (load("flags_gpt_hyb2.json"), "full", "hybrid_ds"),
        (load("flags_l8b_hyb.json"), "full", "hybrid_ds")]
OFF = load("flags_offgreedy_all_set.json"); SC = load("flags_scoff_m_set.json")
off6 = [(OFF, "off:full", k) for k in ("off:hybrid_ds", "off:hybridv_ds", "off:hybridv_g6l")] + \
       [(SC, "off:full", k) for k in ("off:hybrid_ds", "off:hybridv_ds", "off:hybridv_g6l")]
claims = {"single-stage CRISP eps=.05, 5 generators": single5,
          "single-stage CRISP eps=.05, 6 generators incl. OmniSQL": single6,
          "two-stage uniform DS selector, 5 generators": two5,
          "OmniSQL official greedy value-aware DS": [(OFF, "off:full", "off:hybridv_ds")],
          "OmniSQL official omnibus (6 configs, set metric)": off6}
out["clustered"] = {}
for name, E in claims.items():
    q = pooled(E, "question"); d = pooled(E, "database")
    out["clustered"][name] = {"question_clustered": q, "database_clustered": d}
    print(f"{name:55s} q-clust {q['diff']*100:+.2f} [{q['ci'][0]*100:+.2f},{q['ci'][1]*100:+.2f}] p={q['p']:.4f} | "
          f"DB-clust (11) [{d['ci'][0]*100:+.2f},{d['ci'][1]*100:+.2f}] p={d['p']:.4f}")

# ---------------------------------------------------------------- 4. matched size: certified vs fixed top-k
from jevschema.selection import closure, parse_closure
v1 = None
cert_sets = {e_: S[f"t1_e{e_}"] for e_ in (0.02, 0.05, 0.1)}
from v2data import fused_scores, lexical, read_scores
from v2_cascade import tier1_scores
bm, em = lexical("dev")
s0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
ld = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl"))
s1 = tier1_scores(dev, s0, [ld[e.ex_id] for e in dev], 0.02)
V = View(dev, s1, "f1k1")
match = {}
for e_, sets in cert_sets.items():
    size = float(np.mean([len(s) for s in sets])); cov = float(np.mean([covers(x.gold, set(map(tuple, s))) for x, s in zip(dev, sets)]))
    best = None
    for k in range(1, 80):
        sz = float(np.mean([V.size(i, k) for i in range(N)]))
        if best is None or abs(sz - size) < abs(best[1] - size):
            best = (k, sz, float(np.mean(V.r <= k)))
    match[str(e_)] = {"certified": {"mean_cols": size, "coverage": cov, "target": 1 - e_},
                      "fixed_topk_matched_mean_size": {"k": best[0], "mean_cols": best[1], "coverage": best[2]}}
    print(f"eps={e_}: certified {size:.1f} cols cov {cov:.4f} (target {1-e_}) | fixed top-{best[0]} {best[1]:.1f} cols cov {best[2]:.4f}")
out["matched_size"] = match

# ---------------------------------------------------------------- 5. end-to-end tokens
def usage(tag):
    p = os.path.join(CACHE, f"gen_{tag}_usage.json")
    return json.load(open(p)) if os.path.exists(p) else None
cost = {}
for name, tags in {"DeepSeek selector (plain)": ["hybridlink_ds"], "DeepSeek selector (value-aware)": ["hybridv_ds_link"],
                   "GPT-6-Luna selector (value-aware)": ["hybridv_g6l_link", "hybridv_g6l_linkb"],
                   "GPT-6-Luna selector on full schema": ["hybridv_fullpool_g6l_link"]}.items():
    us = [usage(t) for t in tags]; us = [u for u in us if u]
    if not us:
        continue
    calls = sum(u["calls"] for u in us)
    cost[name] = {"input_per_q": sum(u["prompt_tokens"] for u in us) / calls, "output_per_q": sum(u["completion_tokens"] for u in us) / calls,
                  "reasoning_per_q": sum(u.get("reasoning_tokens", 0) for u in us) / calls, "calls": calls}
    print(name, {k: round(v, 1) for k, v in cost[name].items()})
out["selector_tokens"] = cost

# ---------------------------------------------------------------- 6. rerun replication
rep = {}
sel = json.load(open(B + "tasks_noise_sel.json"))
for gname, fn in (("dsv4flash", "flags_noise_dsv4flash.json"), ("gpt56luna", "flags_noise_gpt56luna.json")):
    A = load(f"flags_all2_{gname}.json"); Bn = load(fn)
    ii = [i for i in sel["idx"] if None not in (A["full"][i], A["jev_t10:t1_e0.05"][i], Bn["full"][i], Bn["jev_t10:t1_e0.05"][i])]
    d1 = np.mean([A["jev_t10:t1_e0.05"][i] - A["full"][i] for i in ii]); d2 = np.mean([Bn["jev_t10:t1_e0.05"][i] - Bn["full"][i] for i in ii])
    dm = np.mean([(A["jev_t10:t1_e0.05"][i] + Bn["jev_t10:t1_e0.05"][i] - A["full"][i] - Bn["full"][i]) / 2 for i in ii])
    rep[gname] = {"n": len(ii), "run1": float(d1), "run2": float(d2), "mean_of_runs": float(dm)}
    print("rerun", gname, rep[gname])
out["rerun"] = rep

# ---------------------------------------------------------------- 7. corrected two-level guarantee (PAC per database)
def pac_threshold(scores, eps, delta):
    """smallest order statistic s_(j) with P(true coverage of threshold s_(j) >= 1-eps) >= 1-delta (binomial bound)."""
    s = np.sort(scores); n = len(s)
    for j in range(1, n + 1):
        if binom.cdf(j - 1, n, 1 - eps) >= 1 - delta:   # P(#{s_i <= q_{1-eps}} <= j-1) >= 1-delta
            return s[j - 1]
    return np.inf
fam_v = View(dev, s1, "f1k1")
r = fam_v.r.astype(float)          # fixed family: conformity score = oracle depth
dbs = sorted(set(db))
two = {}
for eps, delta in ((0.05, 0.01), (0.05, 0.05), (0.1, 0.01)):
    per = {d: r[[i for i in range(N) if db[i] == d]] for d in dbs}
    res = []
    for d_new in dbs:
        others = [d for d in dbs if d != d_new]
        U = sorted([pac_threshold(per[d], eps, delta) for d in others], reverse=True)
        for m in (1, 2):
            lam = U[m - 1]
            covn = float(np.mean(per[d_new] <= lam)); sz = float(np.mean([fam_v.size(i, int(min(lam, 10**6))) if np.isfinite(lam) else fam_v.n_cols[i]
                                                                        for i in range(N) if db[i] == d_new]))
            res.append({"db": d_new, "m": m, "lam": float(lam), "coverage": covn, "cols": sz})
    for m in (1, 2):
        rr = [x for x in res if x["m"] == m]
        K = len(dbs) - 1
        bound = (K - m + 1) / (K + 1) - K * delta
        two[f"eps{eps}_delta{delta}_m{m}"] = {"dbs_meeting_target": sum(x["coverage"] >= 1 - eps for x in rr), "n_dbs": len(rr),
                                              "mean_cols": float(np.mean([x["cols"] for x in rr])), "theory_lower_bound": bound,
                                              "per_db": rr}
        print(f"two-level PAC eps={eps} delta={delta} m={m}: {sum(x['coverage'] >= 1-eps for x in rr)}/{len(rr)} DBs meet 1-eps; "
              f"mean cols {np.mean([x['cols'] for x in rr]):.1f}; bound {bound:.3f}")
out["two_level_pac"] = two
json.dump(out, open(os.path.join(RES, "review_analyses.json"), "w"), indent=1, default=float)
