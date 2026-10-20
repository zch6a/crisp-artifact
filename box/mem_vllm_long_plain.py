"""vLLM SQL generation for long prompts (large-schema EX): python mem_vllm_long.py <model> <in.jsonl> <out.jsonl>.
Env TP, UTIL, MAXLEN. Prompts longer than MAXLEN-400 tokens cannot be answered by the model: recorded with sql=""
and overflow=true (counted as a wrong answer, i.e. the honest outcome of feeding the full schema)."""
import json, os, re, sys, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem_vllm_extract import extract_sql

model, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
maxlen = int(os.environ.get("MAXLEN", "32768"))
rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
done = {json.loads(l)["h"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
todo = [r for r in rows if r["h"] not in done]
tok = AutoTokenizer.from_pretrained(model)
if os.environ.get("SKIPLEN"):          # lengths already verified for this prompt set (skip the slow CPU pass)
    lens = [0 for r in todo]
else:
    lens = [len(tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)).input_ids) for r in todo]
ok = [r for r, n in zip(todo, lens) if n <= maxlen - 400]
over = [r for r, n in zip(todo, lens) if n > maxlen - 400]
print(f"{len(todo)} to generate, {len(over)} over {maxlen - 400} tokens (max {max(lens) if lens else 0})", flush=True)
with open(out, "a", encoding="utf-8") as f:
    for r in over:
        f.write(json.dumps({"h": r["h"], "sql": "", "overflow": True}) + "\n")
llm = LLM(model=model, max_model_len=maxlen, gpu_memory_utilization=float(os.environ.get("UTIL", "0.90")),
          tensor_parallel_size=int(os.environ.get("TP", "2")), enforce_eager=bool(os.environ.get("EAGER")))
NS = int(os.environ.get("NSAMP", "1"))
sp = SamplingParams(temperature=float(os.environ.get("TEMP", "0.0")), n=NS, seed=0,
                    max_tokens=int(os.environ.get("MAXTOK", "384")),
                    stop=(None if os.environ.get("NOSTOP") else ["\n\n\n", ";END"]))
t0 = time.time()
ok.sort(key=lambda r: len(r["prompt"]))
for i in range(0, len(ok), 1024):
    part = ok[i:i + 1024]
    res = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in part], sp, use_tqdm=False)
    with open(out, "a", encoding="utf-8") as f:
        for r, o in zip(part, res):
            txt = o.outputs[0].text
            rec = {"h": r["h"], "text": txt} if os.environ.get("RAW") else {"h": r["h"], "sql": extract_sql(txt)}
            if NS > 1:
                rec["sqls"] = [extract_sql(x.text) for x in o.outputs]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {i + len(part)}/{len(ok)}  {(time.time() - t0) / 60:.1f} min", flush=True)
print("DONE", flush=True)
