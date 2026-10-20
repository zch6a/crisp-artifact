"""Structural table gate: the closure idea applied at the table level.

gate = top-m tables by max-column BM25, closed under FK adjacency (h hops), then topped up with the
next best tables until `cap`.  On the 2040-column scaling benchmark it lifts gate recall from 0.910
(BM25 top-10) to 0.986 (m8 h2 cap20): relevant tables are FK-connected, distractors are not.
"""
from __future__ import annotations

from .summary import bm25_scores


def bm25_cols(e):
    pairs = e.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = e.question + (" " + e.evidence if e.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def ranked_tables(scores):
    best = {}
    for (t, c), v in scores.items():
        best[t] = max(best.get(t, -1e9), v)
    return [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])]


def fk_gate(e, scores=None, m=8, hops=2, cap=20):
    ranked = ranked_tables(scores if scores is not None else bm25_cols(e))
    adj = {}
    for f in e.schema.fks:
        adj.setdefault(f.src_table, set()).add(f.dst_table)
        adj.setdefault(f.dst_table, set()).add(f.src_table)
    g = set(ranked[:m]); frontier = set(g)
    for _ in range(hops):
        nxt = set()
        for t in frontier:
            nxt |= adj.get(t, set())
        nxt -= g; g |= nxt; frontier = nxt
    for t in ranked:
        if len(g) >= cap:
            break
        g.add(t)
    return g


def bm25_gate(e, top_t=10):
    return set(ranked_tables(bm25_cols(e))[:top_t])
