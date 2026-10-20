"""Local ollama generation that keeps the full reply (for official-format prompts whose parser takes the last ```sql
block).  Long context (num_ctx), no stop sequences, thinking disabled for Qwen3 (think=false).  Resumable.
  v2_ollama_raw.py <prompts.jsonl> <model> <tag> [num_ctx=16384] [num_predict=2048]  -> cache/gen_<tag>.jsonl {h, text}
"""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import CACHE

pf, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
num_ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
num_predict = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
out = os.path.join(CACHE, f"gen_{tag}.jsonl")
done = set()
if os.path.exists(out):
    for l in open(out, encoding="utf-8"):
        try:
            done.add(json.loads(l)["h"])
        except Exception:
            pass
rows, seen = [], set()
for l in open(pf, encoding="utf-8"):
    r = json.loads(l)
    if r["h"] not in done and r["h"] not in seen:
        seen.add(r["h"]); rows.append(r)
print(f"{model}: {len(rows)} to generate ({len(done)} cached), num_ctx={num_ctx}", flush=True)


def post(prompt):
    body = {"model": model, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.0, "num_ctx": num_ctx, "num_predict": num_predict}}
    if model.startswith("qwen3"):
        body["think"] = False
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        p = json.loads(r.read().decode("utf-8"))
    if p.get("error"):
        raise RuntimeError(str(p["error"])[:300])
    return p["message"]["content"], p.get("prompt_eval_count"), p.get("eval_count")


t0 = time.time()
with open(out, "a", encoding="utf-8") as f:
    for n, r in enumerate(rows):
        for attempt in range(3):
            try:
                text, pin, pout = post(r["prompt"]); break
            except Exception as ex:
                print("  retry", attempt, str(ex)[:120], flush=True); time.sleep(10)
        else:
            continue
        f.write(json.dumps({"h": r["h"], "text": text, "in_tok": pin, "out_tok": pout}, ensure_ascii=False) + "\n"); f.flush()
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(rows)} {(time.time() - t0) / 60:.1f} min", flush=True)
print("done", flush=True)
