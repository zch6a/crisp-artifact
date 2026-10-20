"""Wall-clock of tier-0 (BM25 + MiniLM on CPU) and of the closure + budget step, per question."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(4)
from sentence_transformers import SentenceTransformer
from v2data import MODELS, load_examples, View, fused_scores
from jevschema.summary import bm25_scores
from jevschema.certify import kept_at
dev = load_examples("dev")[::8][:192]
m = SentenceTransformer(os.path.join(MODELS, "miniLM"), device="cpu")
tb, te, sc = [], [], []
for e in dev:
    pairs = e.schema.columns(); q = e.question + " " + (e.evidence or "")
    t0 = time.time(); b = bm25_scores(q, [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]); tb.append(time.time() - t0)
    t0 = time.time(); v = m.encode([q] + [f"{t.replace('_',' ')} {c.replace('_',' ')}" for t, c in pairs], normalize_embeddings=True, show_progress_bar=False); te.append(time.time() - t0)
    sc.append(fused_scores(e, dict(zip(pairs, b)), dict(zip(pairs, v[1:] @ v[0]))))
vw = View(dev, sc, "f1p5s15k1")
t0 = time.time()
for i, e in enumerate(dev):
    kept_at(e.schema, vw.orders[i], 20, vw.cfg)
tc = (time.time() - t0) / len(dev)
print(f"n={len(dev)} bm25 {np.mean(tb)*1000:.1f} ms/q  MiniLM(CPU) {np.mean(te)*1000:.1f} ms/q  closure {tc*1000:.2f} ms/q")
