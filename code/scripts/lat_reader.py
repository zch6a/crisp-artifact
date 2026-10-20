"""Reader latency on the latency-benchmark questions (same block reader and settings as the BIRD/SNAILS runs):
  python lat_reader.py <model_dir> <examples.pkl> <out.json>
Per question: build the table blocks (BM25 table gate, top 10 tables) and score them in one batched forward pass;
5 warm-up questions, then wall-clock per question on the GPU (synchronised)."""
import json, pickle, sys, time
import numpy as np
import torch
import v2_block_reader as br
M, P, out = sys.argv[1:4]
exs = pickle.load(open(P, "rb"))
dev = torch.device("cuda")
rl, cfg, tok, model = br.load_model(M, dev); model.eval()
pad = tok.pad_token_id or 0


def score(e):
    its = br.make_items(rl, tok, [e], 1024, 320, all_tables=False, top_t=10)
    for g in br.batches(its, 16384, False):
        sub = [its[i] for i in g]
        br.forward_logits(model, rl, sub, pad, dev)
    torch.cuda.synchronize()


with torch.no_grad():
    for e in exs[:5]:
        score(e)
    lat = []
    for e in exs:
        t0 = time.perf_counter(); score(e); lat.append(time.perf_counter() - t0)
res = {"n": len(lat), "p50_s": float(np.median(lat)), "p95_s": float(np.percentile(lat, 95)), "mean_s": float(np.mean(lat)),
       "gpu": torch.cuda.get_device_name(0)}
json.dump(res, open(out, "w"), indent=1)
print(res, flush=True)
