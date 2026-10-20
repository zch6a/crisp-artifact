"""Prompts (from prompts_all2) that Llama-3.1-8B has not generated yet; its first pass covered 764/1534 questions.
  llama_fullfill.py <tasks.jsonl> <out.jsonl>   (task order = priority order)"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jevschema.paths import CACHE, RESULTS
need = [json.loads(l)["h"] for l in open(sys.argv[1], encoding="utf-8")]
done = set()
for l in open(os.path.join(CACHE, "gen_llama3.1_8b.jsonl"), encoding="utf-8"):
    try:
        r = json.loads(l)
        if r.get("sql"):
            done.add(r["h"])
    except Exception:
        pass
P = {}
for l in open(os.path.join(RESULTS, "prompts_all2.jsonl"), encoding="utf-8"):
    r = json.loads(l); P.setdefault(r["h"], l if l.endswith("\n") else l + "\n")
seen = set()
with open(sys.argv[2], "w", encoding="utf-8") as out:
    for h in need:
        if h not in done and h not in seen and h in P:
            seen.add(h); out.write(P[h])
print(f"need {len(set(need))} done {len(set(need) & done)} written {len(seen)} no_prompt {len(set(need) - done - set(P))}")
