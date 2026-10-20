"""Generic vLLM text generation: python gen_text.py <prompts.jsonl> <out.jsonl>. Env TP, MAXLEN, UTIL, MAXTOK."""
import json, os, sys, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
inp, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
done = {json.loads(l)["h"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
todo = [r for r in rows if r["h"] not in done]
from vllm import LLM, SamplingParams
llm = LLM(model=os.environ.get("MODEL", MODELS + "/Qwen2.5-32B-Instruct-AWQ"), quantization="awq",
          max_model_len=int(os.environ.get("MAXLEN", "4096")), tensor_parallel_size=int(os.environ.get("TP", "1")),
          gpu_memory_utilization=float(os.environ.get("UTIL", "0.90")), enforce_eager=True)
sp = SamplingParams(temperature=0.0, max_tokens=int(os.environ.get("MAXTOK", "384")))
t0 = time.time()
for i in range(0, len(todo), 300):
    part = todo[i:i + 300]
    res = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in part], sp, use_tqdm=False)
    with open(out, "a", encoding="utf-8") as f:
        for r, o in zip(part, res):
            f.write(json.dumps({"h": r["h"], "text": o.outputs[0].text}, ensure_ascii=False) + "\n")
    print(f"  {i+len(part)}/{len(todo)} {(time.time()-t0)/60:.1f} min", flush=True)
print("DONE", flush=True)
