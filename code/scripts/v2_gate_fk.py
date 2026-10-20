"""Structural table gate: top-m tables by BM25 plus their FK neighbourhood (1 or 2 hops)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, load_examples, read_scores


def fk_gate(e, s, m, hops, cap):
    best = {}
    for (t, c), v in s.items():
        best[t] = max(best.get(t, -1e9), v)
    ranked = [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    adj = {}
    for f in e.schema.fks:
        adj.setdefault(f.src_table, set()).add(f.dst_table); adj.setdefault(f.dst_table, set()).add(f.src_table)
    g = set(ranked[:m]); frontier = set(g)
    for _ in range(hops):
        nxt = set()
        for t in frontier:
            nxt |= adj.get(t, set())
        nxt -= g; g |= nxt; frontier = nxt
    extra = [t for t in ranked if t not in g]
    g |= set(extra[: max(0, cap - len(g))])
    return g


for n in (250, 500, 1000, 2000):
    exs = load_examples(f"scale{n}")
    bm = read_scores(os.path.join(CACHE, f"scale{n}_bm25.jsonl"))
    out = []
    for m, hops, cap in ((5, 1, 10), (5, 2, 10), (5, 1, 15), (5, 2, 15), (10, 1, 20), (8, 2, 20)):
        ok, sz = [], []
        for e in exs:
            g = fk_gate(e, bm[e.ex_id], m, hops, cap)
            ok.append(all(any(t in g for (t, c) in req) for req in e.gold.requirements)); sz.append(len(g))
        out.append(f"m{m}h{hops}cap{cap}: rec={np.mean(ok):.4f} tables={np.mean(sz):.1f}")
    print(f"N={n}: " + " | ".join(out), flush=True)
