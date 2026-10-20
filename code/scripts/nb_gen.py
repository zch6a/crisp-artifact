"""vLLM generation for the new-benchmark prompts: python nb_gen.py <model_dir> <in.jsonl {h,prompt}> <out.jsonl {h,text}>.
Greedy, user-only chat prompt (same template as every generator in the paper); Qwen3 thinking disabled; resumable.
Env: UTIL (gpu_memory_utilization), MAXLEN, MAXTOK, TP, KVGB, MAXSEQS, REVERSE."""
import json, os, sys, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
M, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
lock = out + ".lock"                            # one writer per output file (several GPU queues may reach the same model)
try:
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
except FileExistsError:
    try:
        os.kill(int(open(lock).read().strip() or 0), 0); print("locked by another writer; skip", flush=True); sys.exit(0)
    except (ProcessLookupError, ValueError):
        open(lock, "w").write(str(os.getpid()))   # stale lock
import atexit
atexit.register(lambda: os.path.exists(lock) and open(lock).read().strip() == str(os.getpid()) and os.remove(lock))
rows, seen = [], set()
for l in open(inp, encoding="utf-8"):
    r = json.loads(l)
    if r["h"] not in seen:
        seen.add(r["h"]); rows.append(r)
done = {json.loads(l)["h"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
todo = [r for r in rows if r["h"] not in done]
if os.environ.get("REVERSE") == "1":   # second worker from the other end (own output file; merged afterwards)
    todo = todo[::-1]
print(len(todo), "to do", flush=True)
if todo and os.environ.get("WAITMEM", "1") == "1":   # shared GPUs: wait until every visible GPU has the memory we will take
    import subprocess
    need = float(os.environ.get("UTIL", "0.6")); vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    tw = time.time() + float(os.environ.get("WAITMAX", "1e9"))   # give up (exit 3, lock released) so a scheduler can try another GPU
    while True:
        if time.time() > tw:
            print("WAIT_TIMEOUT", flush=True); sys.exit(3)
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits", "-i", vis],
                           capture_output=True, text=True).stdout.strip().splitlines()
        free = [(float(t) - float(u)) / float(t) for _, u, t in (x.split(", ") for x in q)]
        if free and min(free) >= need + 0.02:
            break
        print(f"waiting for GPU memory: free {[round(f, 2) for f in free]} < need {need}", flush=True); time.sleep(30)
from vllm import LLM, SamplingParams
ML = int(os.environ.get("MAXLEN", "32768")); MT = int(os.environ.get("MAXTOK", "512"))
llm = LLM(model=M, max_model_len=ML, tensor_parallel_size=int(os.environ.get("TP", "1")),
          gpu_memory_utilization=float(os.environ.get("UTIL", "0.6")), enable_prefix_caching=True,
          **({"max_num_seqs": int(os.environ["MAXSEQS"])} if os.environ.get("MAXSEQS") else {}),   # smaller sampler warm-up on nearly full GPUs
          **({"kv_cache_memory_bytes": int(float(os.environ["KVGB"]) * 2**30)} if os.environ.get("KVGB") else {}))  # fixed KV budget: skips memory profiling (neighbours on shared GPUs change their usage)
tok = llm.get_tokenizer()
keep = [r for r in todo if len(tok.encode(r["prompt"])) + MT + 64 <= ML]
with open(out, "a", encoding="utf-8") as f:
    for r in todo:
        if r not in keep:
            f.write(json.dumps({"h": r["h"], "text": "", "too_long": 1}) + "\n")
sp = SamplingParams(temperature=0.0, max_tokens=MT)
kw = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in M.lower() else {}
t0 = time.time()
for i in range(0, len(keep), 512):
    part = keep[i:i + 512]
    res = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in part], sp, use_tqdm=False, **kw)
    with open(out, "a", encoding="utf-8") as f:
        for r, o in zip(part, res):
            f.write(json.dumps({"h": r["h"], "text": o.outputs[0].text}, ensure_ascii=False) + "\n")
    print(f"  {i+len(part)}/{len(keep)} {(time.time()-t0)/60:.1f} min", flush=True)
print("GEN_DONE", flush=True)
