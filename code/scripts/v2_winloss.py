"""Question-level win/loss analysis of CRISP eps=0.05 vs full schema on BIRD-Dev, all generators (recomputed flags).

loss (full right, CRISP wrong): 'miss' = the certified set lost a required gold column; 'covered' = all gold kept.
win (full wrong, CRISP right): characterized by how many *confusable* columns the full schema carried, i.e.
non-gold columns whose Jev score is at least the lowest gold score (they rank among the gold columns).
Also: share of wins/losses shared across generators (a question that flips for >=3 generators is systematic).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
from v2_e2e import build_systems

dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
kept = S["t1_e0.05"]
bm, em = lexical("dev"); jd = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl"))
f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
sc = combine(dev, f0, [jd.get(e.ex_id, {}) for e in dev], 0.02)


def confusable(i):
    e, s = dev[i], sc[i]
    gold = set(e.gold.flat_pairs())
    gmin = min((s.get(p, -9) for p in gold), default=-9)
    return sum(1 for p in e.schema.columns() if p not in gold and s.get(p, -9) >= gmin)


cov = [all(req & set(k) for req in e.gold.requirements) for e, k in zip(dev, kept)]
conf = [confusable(i) for i in range(len(dev))]
flip = {}
tot = {"loss_miss": 0, "loss_covered": 0, "win": 0}
win_conf, loss_conf = [], []
for g in ("qwen2.5-coder_7b", "llama3.1_8b", "qwen2.5-32b-awq", "gpt56luna", "dsv4flash"):
    F = json.load(open(os.path.join(RES, f"flags_all2_{g}.json")))
    for i, (a, b) in enumerate(zip(F["full"], F["jev_t10:t1_e0.05"])):
        if a is None or b is None or a == b:
            continue
        if a > b:
            k = "loss_miss" if not cov[i] else "loss_covered"; tot[k] += 1; loss_conf.append(conf[i])
            flip.setdefault(i, []).append(-1)
        else:
            tot["win"] += 1; win_conf.append(conf[i]); flip.setdefault(i, []).append(+1)
print("pooled over 5 generators:", tot)
print(f"losses from coverage misses: {tot['loss_miss']/(tot['loss_miss']+tot['loss_covered']):.2%}")
print(f"confusable non-gold columns in the full schema: wins mean {np.mean(win_conf):.1f} (median {np.median(win_conf):.0f}) "
      f"vs losses mean {np.mean(loss_conf):.1f} vs all questions mean {np.mean(conf):.1f}")
sys_w = sum(1 for v in flip.values() if v.count(1) >= 3); sys_l = sum(1 for v in flip.values() if v.count(-1) >= 3)
print(f"systematic questions (same direction for >=3 generators): wins {sys_w}, losses {sys_l}")
json.dump({"totals": tot, "win_conf_mean": float(np.mean(win_conf)), "loss_conf_mean": float(np.mean(loss_conf)),
           "all_conf_mean": float(np.mean(conf)), "systematic_wins": sys_w, "systematic_losses": sys_l},
          open(os.path.join(RES, "winloss.json"), "w"), indent=1)
