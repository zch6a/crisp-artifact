"""Pooled EX significance across generators with a question-clustered bootstrap.

The same BIRD-Dev questions appear under every generator, so per-(generator, question) differences are not
independent.  v2_pooled.py resamples each generator separately (too narrow a CI when gains are correlated across
generators); here whole questions are resampled, carrying all of their generator rows with them.
Also reports a question-level sign test (mean difference per question) and which configs contribute to each system.

  v2_pooled_cluster.py [--only substr,substr] [--systems jev_t10:t1_e0.05,...]
"""
import argparse, glob, json, math, os, re
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import RESULTS as RES
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="comma list of substrings; keep result files whose name contains one")
ap.add_argument("--systems", default="jev_t10:t1_e0.02,jev_t10:t1_e0.05,jev_t10:t1_e0.1,jev_t10:t0_e0.05,laya_t10:t1_e0.05")
ap.add_argument("--B", type=int, default=10000)
a = ap.parse_args()
only = [s for s in a.only.split(",") if s]

rows = {}
for p in sorted(glob.glob(os.path.join(RES, "e2e_*.json"))):
    tag = os.path.basename(p).replace("e2e_", "").replace(".json", "")
    if only and not any(s in tag for s in only):
        continue
    d = json.load(open(p))
    fl = d.get("flags")
    if not isinstance(fl, dict) or "full" not in fl or d.get("tier1") in (None, "hybrid"):
        continue
    m = re.search(r"_s(\d+)", tag)
    stride = int(m.group(1)) if m else 1
    for name, f in fl.items():
        if name == "full":
            continue
        key = f"{d.get('tier1')}:{name}"
        for i, (x, y) in enumerate(zip(fl["full"], f)):
            if x is not None and y is not None:
                rows.setdefault(key, []).append((tag, i * stride, float(y) - float(x)))


def binom_two_sided(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n)


rng = np.random.default_rng(0)
for key in a.systems.split(","):
    items = rows.get(key, [])
    if not items:
        print(f"{key:24s} no data"); continue
    by_q = {}
    for tag, q, dl in items:
        by_q.setdefault(q, []).append(dl)
    qs = sorted(by_q)
    s = np.array([sum(by_q[q]) for q in qs]); c = np.array([len(by_q[q]) for q in qs])
    mean = s.sum() / c.sum()
    idx = rng.integers(0, len(qs), (a.B, len(qs)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    qm = s / c
    w, l = int((qm > 0).sum()), int((qm < 0).sum())
    # naive (per-config independent) bootstrap for comparison
    by_t = {}
    for tag, q, dl in items:
        by_t.setdefault(tag, []).append(dl)
    diffs = [np.array(v) for v in by_t.values()]
    nb = np.array([np.mean(np.concatenate([d[rng.integers(0, len(d), len(d))] for d in diffs])) for _ in range(2000)])
    nlo, nhi = np.quantile(nb, [0.025, 0.975])
    cfg = ", ".join(f"{t}:{np.mean(v):+.4f}(n={len(v)})" for t, v in sorted(by_t.items()))
    print(f"{key:24s} diff={mean:+.4f}  clustered95%CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} | naiveCI[{nlo:+.4f},{nhi:+.4f}] | "
          f"question-level sign {w}/{l} p={binom_two_sided(w, l):.4f} | {len(qs)} questions, {len(items)} rows\n      configs: {cfg}")
