"""Three-way protocol for certified selection (Theorem 2 with a pool size that does not depend on the fallback labels).

BIRD-Dev is split into three parts. Part P chooses the pool size from the fixed grid {10, 16}: the smallest k whose closed
top-k prefix covers at least ceil((|P|+1)(1-eps_pool)) questions of P (16 if none). Part F calibrates the fallback depth
mu_hat for that pool (split-conformal quantile of r2). Part T is tested. The pool depends only on P, the fallback only on
F, so Theorem 2 holds on T.  Coverage/size: 200 random three-way splits.  EX: one fixed split, rotated so that every
question is tested once; pooled over three open generators, question-level bootstrap (10,000).
  python three_way.py <res_dir> <flags_dir>
"""
import json, math, os, sys
import numpy as np

RES, FL = sys.argv[1], sys.argv[2]
G = ["q7b", "l8b", "q3"]
T16, T10 = os.environ.get("TAGS", "32,32p10").split(",")      # TAGS=d32,d32p10 for the fully open pipeline
D = {k: json.load(open(os.path.join(RES, f"cert2{t}_r2_k{k}.json"))) for k, t in ((16, T16), (10, T10))}
n = len(D[16]["r2"])
r2 = {k: np.asarray(D[k]["r2"]) for k in D}
cover = {k: np.asarray(D[16]["pool_cover"][str(k)]) for k in (10, 16)}
size = {k: {int(l): np.asarray(v) for l, v in D[k]["size"].items()} for k in D}


def flags(k, lam, g):
    if T16 != "32":                                              # fully open: per-depth flags under flags_cert2{tag}_lam{lam}_{g}.json
        p = os.path.join(FL, f"flags_cert2{T16 if k == 16 else T10}_lam{lam}_{g}.json")
        if not os.path.exists(p):
            return None
        f = json.load(open(p)); return f if isinstance(f, list) else next(iter(f.values()))
    if lam == 0:
        f = json.load(open(os.path.join(FL, f"flags_{g}.json")))[f"tab32_top{k}"]
    else:
        p = os.path.join(FL, f"flags_cert2seed32{'p10' if k == 10 else ''}_lam{lam}_{g}.json")
        if not os.path.exists(p):
            return None
        f = json.load(open(p)); f = f if isinstance(f, list) else next(iter(f.values()))
    return f


FULL = {g: json.load(open(os.path.join(FL, f"flags_{g}.json")))["full"] for g in G}


def qhat(vals, eps):
    v = np.sort(vals); j = math.ceil((len(v) + 1) * (1 - eps))
    return int(v[j - 1]) if j <= len(v) else 10 ** 9


def pool_of(P, eps_pool):
    need = math.ceil((len(P) + 1) * (1 - eps_pool))
    for k in (10, 16):
        if cover[k][P].sum() >= need:
            return k
    return 16


rng = np.random.default_rng(0)
for eps_pool in (0.05, 0.02):
    for eps in (0.05, 0.10):
        cov, sz, ks, lams = [], [], [], []
        for _ in range(200):
            p = rng.permutation(n); P, F, T = p[: n // 3], p[n // 3: 2 * n // 3], p[2 * n // 3:]
            k = pool_of(P, eps_pool); lam = qhat(r2[k][F], eps)
            cov.append(np.mean(r2[k][T] <= lam)); sz.append(np.mean(size[k][min(lam, 30)][T])); ks.append(k); lams.append(lam)
        # EX: fixed split, rotated
        p = np.random.default_rng(1).permutation(n); parts = [p[: n // 3], p[n // 3: 2 * n // 3], p[2 * n // 3:]]
        assign = {}
        for r in range(3):
            P, F, T = parts[r], parts[(r + 1) % 3], parts[(r + 2) % 3]
            k = pool_of(P, eps_pool); lam = qhat(r2[k][F], eps)
            for i in T:
                assign[int(i)] = (k, lam)
        used = sorted(set(assign.values()))
        Fx = {}
        ok = True
        for (k, lam) in used:
            for g in G:
                f = flags(k, lam, g)
                if f is None:
                    ok = False
                Fx[(k, lam, g)] = f
        line = (f"pool@{1-eps_pool:.2f} eps={eps:.2f}: coverage {np.mean(cov):.4f} (p5 {np.percentile(cov, 5):.4f}) "
                f"size {np.mean(sz):.1f}  pools {dict(zip(*np.unique(ks, return_counts=True)))} "
                f"lams {dict(zip(*np.unique(lams, return_counts=True)))}  EX-split uses {used}")
        if not ok:
            print(line, " [missing flags]"); continue
        Dm = np.full((len(G), n), np.nan)
        for j, g in enumerate(G):
            for i in range(n):
                k, lam = assign[i]; x, y = Fx[(k, lam, g)][i], FULL[g][i]
                if x is not None and y is not None:
                    Dm[j, i] = x - y
        q = np.nanmean(Dm, 0); q = q[np.isfinite(q)]
        obs = 100 * q.mean()
        bs = np.array([q[np.random.default_rng(7).integers(0, len(q), len(q))].mean() for _ in range(1)])
        brng = np.random.default_rng(2); B = np.array([q[brng.integers(0, len(q), len(q))].mean() for _ in range(10000)]) * 100
        pval = max(1e-4, 2 * min((B <= 0).mean(), (B >= 0).mean()))
        ex_cov = np.mean([r2[assign[i][0]][i] <= assign[i][1] for i in range(n)])
        ex_sz = np.mean([size[assign[i][0]][min(assign[i][1], 30)][i] for i in range(n)])
        print(line + f"\n    EX vs full {obs:+.2f} [{np.percentile(B, 2.5):+.2f}, {np.percentile(B, 97.5):+.2f}] p={pval:.4f}"
              f"  (rotated split: coverage {ex_cov:.4f}, size {ex_sz:.1f})")
