"""Generate SQL for a prompt file with a local ollama model (resumable).
  v2_ollama_gen.py <prompts.jsonl> <model> <tag>  -> cache/gen_<tag>.jsonl {h, sql}
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import CACHE
from jevschema.generator import OllamaGenerator

pf, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
out = os.path.join(CACHE, f"gen_{tag}.jsonl")
done = set()
if os.path.exists(out):
    for l in open(out, encoding="utf-8"):
        try:
            done.add(json.loads(l)["h"])
        except Exception:
            pass
rows = [json.loads(l) for l in open(pf, encoding="utf-8")]
todo = [r for r in rows if r["h"] not in done]
print(f"{model}: {len(todo)} to generate ({len(done)} cached)", flush=True)
gen = OllamaGenerator(model=model, timeout_s=300)
with open(out, "a", encoding="utf-8") as f:
    for n, r in enumerate(todo):
        s = gen.generate_sql(r["prompt"])
        f.write(json.dumps({"h": r["h"], "sql": s}, ensure_ascii=False) + "\n"); f.flush()
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(todo)}", flush=True)
print("done", flush=True)
