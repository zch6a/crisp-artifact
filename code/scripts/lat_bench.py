"""End-to-end latency on one GPU (vLLM, greedy, same chat template as the EX runs).
  python lat_bench.py <model_dir> <inputs.jsonl> <field> <out.json>   field in {full, cert, sel}
Per-request latency: requests are issued one at a time (batch size 1, the interactive setting), after 5 warm-up
requests; throughput: all requests submitted at once. Env MAXLEN, MAXTOK, UTIL, QUANT."""
import json, os, sys, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import numpy as np
from vllm import LLM, SamplingParams
M, inp, field, out = sys.argv[1:5]
rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
llm = LLM(model=M, max_model_len=int(os.environ.get("MAXLEN", "16384")), gpu_memory_utilization=float(os.environ.get("UTIL", "0.9")),
          quantization=os.environ.get("QUANT") or None, enable_prefix_caching=False)
sp = SamplingParams(temperature=0.0, max_tokens=int(os.environ.get("MAXTOK", "512")))
tok = llm.get_tokenizer()
msgs = [[{"role": "user", "content": r[field]}] for r in rows]
ok = [i for i, m in enumerate(msgs) if len(tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True)) + 512 <= int(os.environ.get("MAXLEN", "16384"))]
for m in msgs[:5]:
    llm.chat([m], sp, use_tqdm=False)                                   # warm-up
lat, ntok_in, ntok_out = [], [], []
for i in ok:
    t0 = time.perf_counter(); o = llm.chat([msgs[i]], sp, use_tqdm=False)[0]; lat.append(time.perf_counter() - t0)
    ntok_in.append(len(o.prompt_token_ids)); ntok_out.append(len(o.outputs[0].token_ids))
t0 = time.perf_counter(); llm.chat([msgs[i] for i in ok], sp, use_tqdm=False); tp = time.perf_counter() - t0
res = {"model": os.path.basename(M.rstrip("/")), "field": field, "n": len(ok), "skipped_too_long": len(rows) - len(ok),
       "p50_s": float(np.median(lat)), "p95_s": float(np.percentile(lat, 95)), "mean_s": float(np.mean(lat)),
       "in_tok": float(np.mean(ntok_in)), "out_tok": float(np.mean(ntok_out)),
       "throughput_q_per_s": len(ok) / tp, "per_q": {str(rows[i]["i"]): l for i, l in zip(ok, lat)}}
json.dump(res, open(out, "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_q"}, flush=True)
