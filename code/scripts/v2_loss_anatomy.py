"""Anatomy of wins/losses vs the full schema: for each pruned system, split questions by whether the kept set covers
all gold columns ("covered") or not ("miss"), and report wins / losses in each part, by BIRD difficulty and by schema
size.  Flags: GPU-box ex_remote outputs copied to results/.
  v2_loss_anatomy.py <flags.json> <full_key> <sys_key>:<sets_source> ...   sets_source in {crisp05, crisp02, crisp10,
  hybrid_ds, hybridv_ds, hybridv_g6l}"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, load_examples
from v2_e2e import build_systems

dev, hold = load_examples("dev"), load_examples("holdout")
S, _ = build_systems(argparse.Namespace(tier1="jev_t10", beta=0.02, route_frac=0.4), dev, hold)
SETS = {"crisp05": S["t1_e0.05"], "crisp02": S["t1_e0.02"], "crisp10": S["t1_e0.1"]}
for tag in ("hybrid_ds", "hybridv_ds", "hybridv_g6l"):
    p = os.path.join(RES, f"{tag}_sets.json")
    if os.path.exists(p):
        SETS[tag] = [[tuple(x) for x in s] for s in json.load(open(p))["sets"]]
gold = [set((t.lower(), c.lower()) for t, c in e.gold.flat_pairs()) for e in dev]
ncol = np.array([len(e.schema.columns()) for e in dev])
diff = [e.difficulty or "unknown" for e in dev]
F = json.load(open(sys.argv[1])); fk = sys.argv[2]
size_bins = [(0, 30, "<=30 cols"), (31, 80, "31-80"), (81, 10**6, ">80")]
for arg in sys.argv[3:]:
    sk, src = arg.rsplit(":", 1)
    sets = SETS[src]
    cov = [gold[i] <= {(t.lower(), c.lower()) for t, c in sets[i]} for i in range(len(dev))]
    idx = [i for i in range(len(dev)) if F[fk][i] is not None and F[sk][i] is not None]
    def wl(sel):
        w = sum(1 for i in sel if F[sk][i] > F[fk][i]); l = sum(1 for i in sel if F[sk][i] < F[fk][i])
        return f"{len(sel):4d} q  w/l {w:3d}/{l:3d}  net {w-l:+4d}"
    print(f"\n=== {sk} (sets={src}, mean {np.mean([len(s) for s in sets]):.1f} cols, coverage {np.mean(cov):.3f})")
    print("  covered:", wl([i for i in idx if cov[i]]))
    print("  miss   :", wl([i for i in idx if not cov[i]]),
          f" | full correct on miss: {sum(1 for i in idx if not cov[i] and F[fk][i]==1)}")
    for d in ("simple", "moderate", "challenging"):
        print(f"  {d:11s}:", wl([i for i in idx if diff[i] == d]), " miss-part:", wl([i for i in idx if diff[i] == d and not cov[i]]))
    for lo, hi, nm in size_bins:
        print(f"  {nm:11s}:", wl([i for i in idx if lo <= ncol[i] <= hi]), " miss-part:", wl([i for i in idx if lo <= ncol[i] <= hi and not cov[i]]))
