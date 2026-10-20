"""Offline batch SQL generation with vLLM (temperature 0), resumable, single-instance (lock file).
Env: TP (tensor parallel size, default 1), UTIL (gpu_memory_utilization, default 0.88)."""
import json, os, re, sys, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "src", "jevschema"))
from paths import BOX, EXT, MODELS  # env CRISP_BOX / CRISP_EXT / CRISP_MODELS
lock = BOX + "/gen.lock"
if os.path.exists(lock):
    try:
        pid = int(open(lock).read().strip()); os.kill(pid, 0)
        print(f"another generator (pid {pid}) is running; exit", flush=True); sys.exit(0)
    except Exception:
        pass
open(lock, "w").write(str(os.getpid()))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
from vllm import LLM, SamplingParams

model, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text):
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    lines = text.strip().split("\n")
    start = 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b", ln, re.I):
            start = i
            break
    body = "\n".join(lines[start:]).strip()
    if ";" in body:
        body = body[: body.index(";") + 1]
    return body.strip().rstrip(";").strip()


try:
    rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
    done = {json.loads(l)["h"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [r for r in rows if r["h"] not in done]
    tp, util = int(os.environ.get("TP", "1")), float(os.environ.get("UTIL", "0.93"))
    print(f"{len(todo)} to generate ({len(done)} done)  TP={tp} UTIL={util}", flush=True)
    llm = LLM(model=model, max_model_len=int(os.environ.get("MAXLEN", "2304")), gpu_memory_utilization=util,
              tensor_parallel_size=tp, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=384, stop=["\n\n\n", ";END"])
    t0 = time.time()
    for i in range(0, len(todo), 2000):
        part = todo[i:i + 2000]
        res = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in part], sp, use_tqdm=False)
        with open(out, "a", encoding="utf-8") as f:
            for r, o in zip(part, res):
                f.write(json.dumps({"h": r["h"], "sql": extract_sql(o.outputs[0].text)}, ensure_ascii=False) + "\n")
        print(f"  {i + len(part)}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("DONE", flush=True)
finally:
    try:
        os.remove(lock)
    except Exception:
        pass
