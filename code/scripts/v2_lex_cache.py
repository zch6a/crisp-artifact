"""Tier-0 lexical signals for BIRD-train (all 6048 kept examples): BM25 + MiniLM cosine.

Same document templates as pareto_closure.py / fusion.py so dev and train scores are
exchangeable in distribution (the old budget predictor broke exactly because train had
bm25 only while dev had bm25+emb).  CPU only -- never touches the GPU.
Outputs: <CRISP_CACHE>/train_bm25.jsonl, train_emb.jsonl
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from jevschema.bird_train import load_bird_train
from jevschema.summary import bm25_scores

from jevschema.paths import CACHE as C, MODELS


def main():
    exs, st = load_bird_train(limit=None, log_every=0)
    print("train examples", len(exs), flush=True)
    pb, pe = os.path.join(C, "train_bm25.jsonl"), os.path.join(C, "train_emb.jsonl")
    done_b = {json.loads(l)["id"] for l in open(pb, encoding="utf-8")} if os.path.exists(pb) else set()
    done_e = {json.loads(l)["id"] for l in open(pe, encoding="utf-8")} if os.path.exists(pe) else set()
    t0 = time.time()
    with open(pb, "a", encoding="utf-8") as f:
        for e in exs:
            if e.ex_id in done_b:
                continue
            pairs = e.schema.columns()
            docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
            q = e.question + (" " + e.evidence if e.evidence else "")
            sc = bm25_scores(q, docs)
            f.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": float(v) for (t, c), v in zip(pairs, sc)}}) + "\n")
    print(f"bm25 done {time.time()-t0:.1f}s", flush=True)
    import torch
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(os.path.join(MODELS, "miniLM"), device="cpu")
    t0 = time.time()
    with open(pe, "a", encoding="utf-8") as f:
        for i, e in enumerate(exs):
            if e.ex_id in done_e:
                continue
            pairs = e.schema.columns()
            docs = [f"{t.replace('_',' ')} {c.replace('_',' ')}" for t, c in pairs]
            q = e.question + (" " + e.evidence if e.evidence else "")
            emb = m.encode([q] + docs, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
            sims = emb[1:] @ emb[0]
            f.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": float(v) for (t, c), v in zip(pairs, sims)}}) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  emb {i+1}/{len(exs)} {time.time()-t0:.0f}s", flush=True)
    print(f"emb done {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
