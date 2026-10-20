"""Certified schema linking (v2 core).

1. Nested closure family.  For a scorer order pi over all columns and the closure
   operator Gamma (selection.closure: FK / intra-table pad / PK / small-table), the kept
   set S(k) = Gamma(pi[:k]) is monotone in k: every rule depends only on the per-table
   prefix of pi[:k] or on the set of touched tables.  Hence the oracle depth
        r(x) = min k such that gold(x) is covered by S(k)
   is well defined, SRR@k = P(r(x) <= k), and coverage at depth k  <=>  r(x) <= k.
   check_nested verifies monotonicity empirically.

2. Budget families k(x, lam), monotone in lam.  Any such family is a conformal
   predictor: nonconformity s(x) = min lam with k(x, lam) >= r(x), and lam_hat = the
   ceil((n+1)(1-eps))-th smallest calibration score gives
        P(gold(x_test) covered by S(k(x_test, lam_hat))) >= 1 - eps
   for exchangeable calibration/test questions (split conformal).
     fixed      : k = floor(lam)                        s = r
     normalized : k = floor(exp(mu(x) + lam*sig(x)))    s = (log r - mu)/sig
     mass       : k = first k with M_x(k) >= lam        s = M_x(r)
   mu, sig come from a difficulty model fitted on data disjoint from calibration.

3. Route-conditional (Mondrian) calibration for a cascade.  If the route R(x) is a
   fixed function of x, calibrating lam inside each route keeps
   P(cover | R = r) >= 1 - eps for every r, hence overall.

Coverage uses requirement semantics (metrics.strict_recall).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .selection import ClosureConfig, closure

Pair = Tuple[str, str]


def covers(gold, kept) -> bool:
    if gold is None or not gold.requirements:
        return True
    ks = kept if isinstance(kept, set) else set(kept)
    return all(bool(req & ks) for req in gold.requirements)


def order_from_scores(pairs: Sequence[Pair], scores: Dict[Pair, float]) -> List[Pair]:
    """Full descending order over all columns (closure needs the tail for padding)."""
    v = np.asarray([float(scores.get(p, 0.0)) for p in pairs], dtype=float)
    idx = np.argsort(-v, kind="stable")
    return [pairs[j] for j in idx]


def kept_at(schema, order, k, cfg):
    return closure(schema, order, max(1, int(k)), cfg)


def oracle_depth(schema, order, gold, cfg) -> int:
    """Smallest k with gold covered by closure(order[:k]); binary search over nested sets.
    Returns len(order)+1 if even the full schema does not cover."""
    n = len(order)
    if n == 0:
        return 1
    if not covers(gold, set(kept_at(schema, order, n, cfg))):
        return n + 1
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if covers(gold, set(kept_at(schema, order, mid, cfg))):
            hi = mid
        else:
            lo = mid + 1
    return lo


def check_nested(schema, order, cfg, ks) -> bool:
    prev = set()
    for k in sorted(ks):
        cur = set(kept_at(schema, order, k, cfg))
        if not prev <= cur:
            return False
        prev = cur
    return True


def conformal_quantile(scores, eps: float) -> float:
    """ceil((n+1)(1-eps))-th smallest score, +inf if that exceeds n."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    r = int(math.ceil((n + 1) * (1.0 - eps)))
    return float("inf") if r > n else float(s[max(0, r - 1)])


@dataclass
class Family:
    name: str
    nonconf: Callable
    budget: Callable


def fixed_family(r, n_cols) -> Family:
    def budget(i, lam):
        if not math.isfinite(lam):
            return int(n_cols[i])
        return int(min(n_cols[i], max(1, math.floor(lam))))
    return Family("fixed", lambda i: float(r[i]), budget)


def normalized_family(r, n_cols, mu, sig) -> Family:
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-3)

    def nonconf(i):
        return float((math.log(max(1, r[i])) - mu[i]) / sig[i])

    def budget(i, lam):
        if not math.isfinite(lam):
            return int(n_cols[i])
        k = math.floor(math.exp(min(50.0, mu[i] + lam * sig[i])) * (1 + 1e-9))
        return int(min(n_cols[i], max(1, k)))
    return Family("normalized", nonconf, budget)


def mass_family(r, n_cols, cum_mass) -> Family:
    """cum_mass[i][k-1] = share of non-negative score mass inside the top-k columns."""
    def nonconf(i):
        cm = cum_mass[i]
        return float(cm[int(min(len(cm), max(1, r[i]))) - 1])

    def budget(i, lam):
        if not math.isfinite(lam) or lam >= 1.0:
            return int(n_cols[i])
        k = int(np.searchsorted(cum_mass[i], lam - 1e-12, side="left")) + 1
        return int(min(n_cols[i], max(1, k)))
    return Family("mass", nonconf, budget)


def calibrate(fam: Family, cal_idx, eps: float) -> float:
    return conformal_quantile([fam.nonconf(i) for i in cal_idx], eps)


def evaluate_budgets(fam: Family, lam: float, idx, r):
    ks = np.asarray([fam.budget(i, lam) for i in idx], dtype=int)
    cov = np.asarray([1.0 if r[i] <= k else 0.0 for i, k in zip(idx, ks)])
    return ks, cov


def thresh_family(r, n_cols, sorted_scores) -> Family:
    """k(x, lam) = number of columns with score >= -lam  (conformal score threshold).
    Covered <=> the r-th best score >= -lam <=> lam >= s(x) = -score_(r).  This is the
    classic conformal prediction-set rule; a threshold fitted on another dataset does not
    transfer (Spider -> BIRD), but one calibrated on the target distribution is valid."""
    def nonconf(i):
        ss = sorted_scores[i]
        if r[i] > len(ss):
            return float("inf")
        return float(-ss[r[i] - 1])

    def budget(i, lam):
        if not math.isfinite(lam):
            return int(n_cols[i])
        k = int(np.sum(sorted_scores[i] >= -lam - 1e-12))
        return int(min(n_cols[i], max(1, k)))
    return Family("thresh", nonconf, budget)


def hybrid_family(r, n_cols, sorted_scores, k0) -> Family:
    """max(k0, thresh budget): a fixed floor k0 plus a score threshold for the tail."""
    th = thresh_family(r, n_cols, sorted_scores)

    def nonconf(i):
        return float("-inf") if r[i] <= k0 else th.nonconf(i)

    def budget(i, lam):
        if lam == float("-inf"):
            return int(min(n_cols[i], k0))
        return int(min(n_cols[i], max(k0, th.budget(i, lam))))
    return Family(f"hybrid{k0}", nonconf, budget)
