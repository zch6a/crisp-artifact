"""Difficulty model for adaptive certified budgets.

mu(x) ~ E[log r | x], sig(x) ~ sd[log r | x] from DB-agnostic shape features of the
scorer output (reused from scripts/budget_predictor.feats).  Fitted on BIRD-train
holdout only.  Conformal coverage never depends on how good this model is; only the
average budget does.
"""
from __future__ import annotations

import math

import numpy as np


def shape_features(s_sorted, n_cols, n_tables, q_len, has_ev):
    n = len(s_sorted)
    hi = float(s_sorted[0]) if n else 0.0
    v = s_sorted / hi if hi > 0 else np.zeros_like(s_sorted)
    tot = float(v.sum()) or 1.0

    def at(k):
        return float(v[min(k, n - 1)]) if n else 0.0

    pv = v[v > 1e-9]
    if len(pv):
        q = pv / pv.sum()
        ent = float(-(q * np.log(q)).sum())
    else:
        ent = 0.0
    gaps = -np.diff(v[:30]) if n > 2 else np.zeros(1)
    return np.asarray([
        math.log1p(n_cols), math.log1p(n_tables), math.log1p(q_len), 1.0 if has_ev else 0.0,
        at(0), at(1), at(2), at(4), at(9), at(14), at(19), at(29),
        float(np.mean(v[:10])), float(np.std(v[:20])),
        float(v[:10].sum() / tot), float(v[:20].sum() / tot),
        float((v > 0.5).sum()) / max(1, n), float((v > 0.2).sum()) / max(1, n),
        ent / math.log(max(2, n)),
        float(np.argmax(gaps)) / 30.0 if len(gaps) else 0.0,
        float(gaps.max()) if len(gaps) else 0.0,
    ], dtype=float)


class Difficulty:
    """Heteroscedastic ridge regression on log r."""

    def __init__(self, ridge=1.0, sig_floor=0.15):
        self.ridge, self.sig_floor = ridge, sig_floor

    def _z(self, X):
        Z = (X - self.m) / self.sd
        return np.hstack([Z, np.ones((len(Z), 1))])

    def fit(self, X, logr):
        self.m = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd < 1e-6, 1.0, sd)
        Z = self._z(X)
        A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ logr)
        res = logr - Z @ self.w
        lv = np.log(np.maximum(res ** 2, 1e-4))
        self.wv = np.linalg.solve(A, Z.T @ lv)
        self.v_shift = float(np.log(np.mean(res ** 2) + 1e-9) - np.log(np.mean(np.exp(Z @ self.wv)) + 1e-9))
        yhat = Z @ self.w
        self.r2 = 1 - float(((logr - yhat) ** 2).mean()) / max(1e-9, float(((logr - logr.mean()) ** 2).mean()))
        return self

    def predict(self, X):
        Z = self._z(X)
        mu = Z @ self.w
        sig = np.sqrt(np.exp(np.clip(Z @ self.wv + self.v_shift, -10, 10)))
        return mu, np.maximum(sig, self.sig_floor)
