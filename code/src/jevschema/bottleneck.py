"""Bottleneck-aware ranking: an SRR-aligned training objective for column scorers.

Strict recall is all-or-nothing, so for a nested closure family the only thing that
matters per question is the oracle depth r(x), i.e. the rank of the worst-ranked gold
column that the closure cannot add by itself (the bottleneck).  Pointwise log-loss
spends capacity on easy positives and on the 90% negative mass (AUROC up, top-k down).

Loss per question (linear scorer s = X w):
    s_b  = softmin_tau over non-implied gold columns
    L    = mean over negatives j of softplus((s_j - s_b) / T)
Gold columns that the closure adds deterministically (PKs, FK endpoints, members of
small tables) are "implied" and excluded from the bottleneck.
"""
from __future__ import annotations

import re

import numpy as np

from .summary import bm25_scores

FEATS = ["bm25_col", "bm25_rel", "bm25_hit", "bm25_rank", "emb_col", "emb_rel", "emb_rank",
         "bm25_tbl", "tbl_rank", "is_fk", "is_pk", "is_num", "has_digit", "n_tok", "tbl_size",
         "pos_in_tbl", "bm25_x_emb", "tbl_max_emb", "extra"]


def _tok(s):
    out = []
    for raw in re.findall(r"[a-z0-9_]+", s.lower()):
        out.extend(p for p in raw.split("_") if p)
    return out


def col_features(ex, bm, em, extra=None):
    """Per-column features from cached bm25/emb dicts (same templates as the caches).
    extra: optional dict pair->float (e.g. laya probability) appended as the last feature."""
    pairs = list(ex.schema.columns())
    n = len(pairs)
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    b = np.asarray([bm.get(p, 0.0) for p in pairs], dtype=float)
    bn = b / b.max() if b.max() > 0 else np.zeros(n)
    e = np.asarray([em.get(p, 0.0) for p in pairs], dtype=float) if em else np.zeros(n)
    en = (e - e.min()) / max(1e-9, e.max() - e.min()) if n else e
    tnames = [t.name for t in ex.schema.tables]
    tdocs = [f"{t.name} " + " ".join(c.name for c in t.columns) for t in ex.schema.tables]
    tb = np.asarray(bm25_scores(q, tdocs), dtype=float)
    tbn = {t: v / (tb.max() if tb.max() > 0 else 1.0) for t, v in zip(tnames, tb)}
    torder = {tnames[j]: r for r, j in enumerate(np.argsort(-tb, kind="stable"))}
    fk = set()
    for f in ex.schema.fks:
        fk.add((f.src_table, f.src_col)); fk.add((f.dst_table, f.dst_col))
    pk = {(t.name, c.name) for t in ex.schema.tables for c in t.columns if c.is_pk}
    dtype = {(t.name, c.name): (c.dtype or "text") for t in ex.schema.tables for c in t.columns}
    by_t = {}
    for i, (t, c) in enumerate(pairs):
        by_t.setdefault(t, []).append(i)
    tbl_size, pos_in, brel, erel, tmax_e = (np.zeros(n) for _ in range(5))
    for t, idxs in by_t.items():
        sb, se = b[idxs], en[idxs]
        mb = sb.max() if sb.max() > 0 else 1.0
        me = se.max() if se.max() > 1e-9 else 1.0
        for r, i in enumerate(idxs):
            tbl_size[i] = len(idxs); pos_in[i] = r / max(1, len(idxs) - 1)
            brel[i] = b[i] / mb; erel[i] = en[i] / me; tmax_e[i] = se.max()
    bm_rank = np.empty(n); bm_rank[np.argsort(-b, kind="stable")] = np.arange(n) / max(1, n - 1)
    em_rank = np.empty(n); em_rank[np.argsort(-en, kind="stable")] = np.arange(n) / max(1, n - 1)
    ex_col = np.asarray([extra.get(p, 0.0) for p in pairs]) if extra else np.zeros(n)
    X = np.column_stack([
        bn, brel, (b > 0).astype(float), bm_rank, en, erel, em_rank,
        np.asarray([tbn.get(t, 0.0) for t, _ in pairs]),
        np.asarray([torder.get(t, 0) / max(1, len(tnames) - 1) for t, _ in pairs]),
        np.asarray([1.0 if p in fk else 0.0 for p in pairs]),
        np.asarray([1.0 if p in pk else 0.0 for p in pairs]),
        np.asarray([1.0 if re.search(r"int|real|doub|floa|dec|num", dtype.get(p, "")) else 0.0 for p in pairs]),
        np.asarray([1.0 if re.search(r"\d", c) else 0.0 for _, c in pairs]),
        np.asarray([min(len(_tok(c)), 12) / 12.0 for _, c in pairs]),
        np.minimum(tbl_size, 200.0) / 200.0, pos_in, bn * en, tmax_e, ex_col,
    ])
    return pairs, X


def implied_mask(ex, pairs, small=15):
    """Columns the closure adds by itself once their table is touched (PK, FK endpoint, small table)."""
    fk = set()
    for f in ex.schema.fks:
        fk.add((f.src_table, f.src_col)); fk.add((f.dst_table, f.dst_col))
    pk = {(t.name, c.name) for t in ex.schema.tables for c in t.columns if c.is_pk}
    tsize = {t.name: len(t.columns) for t in ex.schema.tables}
    return np.asarray([(p in pk) or (p in fk) or (tsize.get(p[0], 99) <= small) for p in pairs])


class Packed:
    """Padded tensors for vectorized training: X [Q,N,F], valid/pos/imp masks [Q,N]."""

    def __init__(self, rows):
        Q = len(rows); N = max(len(r[0]) for r in rows); F = rows[0][0].shape[1]
        self.X = np.zeros((Q, N, F)); self.valid = np.zeros((Q, N), bool)
        self.pos = np.zeros((Q, N), bool); self.bott = np.zeros((Q, N), bool)
        for q, (X, pos, imp) in enumerate(rows):
            n = len(X)
            self.X[q, :n] = X; self.valid[q, :n] = True; self.pos[q, :n] = pos
            bt = pos & ~imp
            self.bott[q, :n] = bt if bt.any() else pos


def train_linear(P, loss="bottleneck", iters=400, lr=0.05, l2=1e-3, T=0.1, tau=0.05, seed=0):
    Q, N, F = P.X.shape
    flat = P.X[P.valid]
    mu = flat.mean(0); sd = flat.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    Z = np.where(P.valid[..., None], (P.X - mu) / sd, 0.0)
    w = np.zeros(F); b0 = 0.0
    m = np.zeros(F); v = np.zeros(F)
    neg = P.valid & ~P.pos
    nneg = np.maximum(1, neg.sum(1))
    for it in range(1, iters + 1):
        s = Z @ w + b0
        if loss == "logloss":
            p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
            g_s = np.where(P.valid, p - P.pos, 0.0) / P.valid.sum()
            gb = float(g_s.sum())
        else:
            sb_in = np.where(P.bott, -s / tau, -np.inf)
            mx = sb_in.max(1, keepdims=True)
            wts = np.exp(sb_in - mx); wts = wts / wts.sum(1, keepdims=True)
            s_b = -tau * (np.log(np.exp(sb_in - mx).sum(1)) + mx[:, 0])
            d = (s - s_b[:, None]) / T
            sig = 1 / (1 + np.exp(-np.clip(d, -30, 30)))
            g_neg = np.where(neg, sig, 0.0) / (T * nneg[:, None])
            g_sb = -g_neg.sum(1)
            g_s = g_neg + wts * g_sb[:, None]
            g_s = np.where(P.valid, g_s, 0.0) / Q
            gb = 0.0
        gw = np.einsum("qn,qnf->f", g_s, Z) + l2 * w
        m = 0.9 * m + 0.1 * gw; v = 0.999 * v + 0.001 * gw * gw
        w -= lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        b0 -= lr * gb
    return {"w": w, "b": b0, "mu": mu, "sd": sd}


def apply_linear(M, X):
    return ((X - M["mu"]) / M["sd"]) @ M["w"] + M["b"]
