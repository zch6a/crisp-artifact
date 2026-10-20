"""Tier-0 signals (BM25 + MiniLM) for the hard scaling benchmark (self-copy distractors)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, MODELS, load_examples
from jevschema.summary import bm25_scores

ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="500,2000"); a = ap.parse_args()
import torch
from sentence_transformers import SentenceTransformer
torch.cuda.set_per_process_memory_fraction(0.25, 0)
emb = SentenceTransformer(os.path.join(MODELS, "miniLM"), device="cuda")
for n in [int(x) for x in a.sizes.split(",")]:
    exs = load_examples(f"hard{n}")
    print(f"hard{n}: cols={np.mean([len(e.schema.columns()) for e in exs]):.0f} tables={np.mean([len(e.schema.tables) for e in exs]):.0f}", flush=True)
    pb, pe = (os.path.join(CACHE, f"hard{n}_{k}.jsonl") for k in ("bm25", "emb"))
    done = {json.loads(l)["id"] for l in open(pe, encoding="utf-8")} if os.path.exists(pe) else set()
    t0 = time.time()
    with open(pb, "a", encoding="utf-8") as fb, open(pe, "a", encoding="utf-8") as fe:
        for i, e in enumerate(exs):
            if e.ex_id in done:
                continue
            pairs = e.schema.columns(); q = e.question + (" " + e.evidence if e.evidence else "")
            docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
            fb.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": float(v) for (t, c), v in zip(pairs, bm25_scores(q, docs))}}) + "\n")
            docs2 = [f"{t.replace('_',' ')} {c.replace('_',' ')}" for t, c in pairs]
            v = emb.encode([q] + docs2, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
            fe.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": float(x) for (t, c), x in zip(pairs, v[1:] @ v[0])}}) + "\n")
            if (i + 1) % 300 == 0:
                fb.flush(); fe.flush(); print(f"  {i+1}/{len(exs)} {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"hard{n} done", flush=True)
