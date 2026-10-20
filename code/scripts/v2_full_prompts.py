"""Full-schema generation prompts in our format (same prompt_of as every other system) for a new generator.
-> results/full_genprompts.jsonl {h, prompt}, results/tasks_full.jsonl {sys: full, i, h, db, gold}"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2data import RES, load_examples
from v2_e2e import prompt_of

dev = load_examples("dev")
seen = set()
with open(os.path.join(RES, "full_genprompts.jsonl"), "w", encoding="utf-8") as f, \
        open(os.path.join(RES, "tasks_full.jsonl"), "w", encoding="utf-8") as ft:
    for i, e in enumerate(dev):
        p = prompt_of(e, list(e.schema.columns())); h = hashlib.md5(p.encode()).hexdigest()
        ft.write(json.dumps({"sys": "full", "i": i, "h": h, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
        if h not in seen:
            seen.add(h); f.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
print("full prompts", len(seen))
