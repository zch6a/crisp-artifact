"""Score the schema-scaling benchmark: bm25 + MiniLM (tier-0) and block reader (tier-1, top-10 tables)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from v2data import CACHE, MODELS, load_examples
from jevschema.scale import Scaler
from jevschema.summary import bm25_scores


def scaled(n):
    sc = Scaler(load_examples("holdout"))
    return [sc.merge(e, n) for e in load_examples("dev")]


def done_ids(p):
    return {json.loads(l)["id"] for l in open(p, encoding="utf-8")} if os.path.exists(p) else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="250,500,1000,2000")
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(os.path.join(MODELS, "miniLM"), device="cuda")
    import v2_block_reader as B
    rl, cfg, tok, model = B.load_model(B.OUT, torch.device("cuda"))
    model.eval()
    pad = tok.pad_token_id or 0
    for n in [int(x) for x in a.sizes.split(",")]:
        exs = scaled(n)
        print(f"N={n}: mean cols={np.mean([len(e.schema.columns()) for e in exs]):.0f} "
              f"tables={np.mean([len(e.schema.tables) for e in exs]):.0f}", flush=True)
        pb, pe, pk = (os.path.join(CACHE, f"scale{n}_{k}.jsonl") for k in ("bm25", "emb", "block"))
        db, de, dk = done_ids(pb), done_ids(pe), done_ids(pk)
        t0 = time.time()
        with open(pb, "a", encoding="utf-8") as fb, open(pe, "a", encoding="utf-8") as fe, \
                open(pk, "a", encoding="utf-8") as fk, torch.no_grad():
            for i, e in enumerate(exs):
                pairs = e.schema.columns()
                q = e.question + (" " + e.evidence if e.evidence else "")
                if e.ex_id not in db:
                    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
                    fb.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": float(v) for (t, c), v in zip(pairs, bm25_scores(q, docs))}}) + "\n")
                if e.ex_id not in de:
                    docs = [f"{t.replace('_',' ')} {c.replace('_',' ')}" for t, c in pairs]
                    v = emb.encode([q] + docs, normalize_embeddings=True, show_progress_bar=False, batch_size=512)
                    fe.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": float(x) for (t, c), x in zip(pairs, v[1:] @ v[0])}}) + "\n")
                if e.ex_id not in dk:
                    t1 = time.time()
                    its = B.make_items(rl, tok, [e], 1024, 320, all_tables=False, top_t=10)
                    s = {}
                    for g in B.batches(its, 8192, False):
                        sub = [its[j] for j in g]
                        lg, _ = B.forward_logits(model, rl, sub, pad, torch.device("cuda"))
                        lg = lg.cpu()
                        for r, it in enumerate(sub):
                            for (t, c), pv in zip(it["cols"], torch.sigmoid(lg[r, :len(it["cols"])]).tolist()):
                                s[f"{t}|{c}"] = float(pv)
                    torch.cuda.synchronize()
                    fk.write(json.dumps({"id": e.ex_id, "s": s, "n_seq": len(its),
                                         "enc_tokens": int(sum(len(x["ids"]) for x in its)),
                                         "wall_s": round(time.time() - t1, 3)}) + "\n")
                if (i + 1) % 300 == 0:
                    for f in (fb, fe, fk):
                        f.flush()
                    print(f"  N={n} {i+1}/{len(exs)} {(time.time()-t0)/60:.1f} min", flush=True)
        print(f"N={n} done {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
