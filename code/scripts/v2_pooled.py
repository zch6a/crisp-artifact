"""Pooled significance of EX gains across generators (paired per question, stratified bootstrap + sign test).
Reads e2e JSONs that contain per-question flags (1 correct / 0 wrong / None gold-failure)."""
import glob, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import RESULTS as RES
files = sys.argv[1:] or sorted(glob.glob(os.path.join(RES, "e2e_*.json")))
by_sys = {}
for p in files:
    d = json.load(open(p))
    fl = d.get("flags")
    if not fl or "full" not in fl:
        continue
    tag = os.path.basename(p)
    for name, f in fl.items():
        if name == "full":
            continue
        pairs = [(x, y) for x, y in zip(fl["full"], f) if x is not None and y is not None]
        by_sys.setdefault(f"{d.get('tier1')}:{name}", []).append((tag, np.asarray(pairs, float)))

rng = np.random.default_rng(0)
for s, groups in sorted(by_sys.items()):
    diffs = [g[:, 1] - g[:, 0] for _, g in groups]
    mean = float(np.mean(np.concatenate(diffs)))
    bs = np.asarray([np.mean(np.concatenate([d[rng.integers(0, len(d), len(d))] for d in diffs])) for _ in range(5000)])
    wins = int(sum((g[:, 1] > g[:, 0]).sum() for _, g in groups))
    losses = int(sum((g[:, 1] < g[:, 0]).sum() for _, g in groups))
    n = wins + losses
    p_sign = min(1.0, 2 * sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n) if wins > losses else 1.0
    per = ", ".join(f"{t.replace('e2e_', '').replace('.json', '')}: {np.mean(g[:, 1] - g[:, 0]):+.4f}" for t, g in groups)
    print(f"{s:28s} pooled diff={mean:+.4f} 95%CI[{np.quantile(bs, .025):+.4f},{np.quantile(bs, .975):+.4f}] "
          f"p_boot={2 * min((bs <= 0).mean(), (bs >= 0).mean()):.4f} wins/losses={wins}/{losses} p_sign={p_sign:.4f}\n      ({per})")
