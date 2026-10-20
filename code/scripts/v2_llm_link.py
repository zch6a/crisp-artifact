"""LLM schema-linking baseline (CHESS / RSL-SQL style column selection by an LLM reading the full schema).

  dump  : write linking prompts (one per dev question) -> results/llmlink_prompts.jsonl
  parse : read {h, text} outputs, build kept sets (raw and + FK closure), report SRR / Cbar / linking tokens,
          and dump SQL-generation prompts for those sets -> results/llmlink_genprompts.jsonl
"""
import argparse, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, load_examples
from v2_e2e import prompt_of
from jevschema.certify import covers
from jevschema.selection import closure, parse_closure
from jevschema.tokens import n_tokens, schema_prompt_text

TMPL = """You are an expert in SQLite schema linking. Given the database schema, the question and the evidence,
list EVERY column that a correct SQL query needs (SELECT, WHERE, JOIN keys, GROUP BY, ORDER BY).
Output one column per line in the form table.column and nothing else.

Database: {db}
Schema:
{schema}

Evidence: {ev}
Question: {q}
Columns:"""


def link_prompt(e):
    return TMPL.format(db=e.db_id, schema=schema_prompt_text(e.schema), ev=e.evidence or "", q=e.question)


def parse_cols(text, schema):
    valid = {(t.name.lower(), c.name.lower()): (t.name, c.name) for t in schema.tables for c in t.columns}
    out = []
    for line in text.splitlines():
        m = re.search(r"[`\"]?([\w ]+?)[`\"]?\s*\.\s*[`\"]?([^`\"\n]+?)[`\"]?\s*$", line.strip().strip("-* ,"))
        if m:
            k = (m.group(1).strip().lower(), m.group(2).strip().lower())
            if k in valid and valid[k] not in out:
                out.append(valid[k])
    return out


ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "parse"]); ap.add_argument("--outputs", default="")
a = ap.parse_args()
dev = load_examples("dev")
if a.mode == "dump":
    with open(os.path.join(RES, "llmlink_prompts.jsonl"), "w", encoding="utf-8") as f:
        for e in dev:
            p = link_prompt(e)
            f.write(json.dumps({"h": hashlib.md5(p.encode()).hexdigest(), "id": e.ex_id, "prompt": p}, ensure_ascii=False) + "\n")
    print("dumped", len(dev))
else:
    got = {json.loads(l)["h"]: json.loads(l)["text"] for l in open(a.outputs, encoding="utf-8")}
    f1 = parse_closure("f1")
    rows, sets = [], {"llm": [], "llm+fk": []}
    for e in dev:
        p = link_prompt(e); t = got.get(hashlib.md5(p.encode()).hexdigest(), "")
        cols = parse_cols(t, e.schema)
        fk = closure(e.schema, cols + [x for x in e.schema.columns() if x not in cols], max(1, len(cols)), f1) if cols else []
        sets["llm"].append(cols); sets["llm+fk"].append(fk)
        rows.append((covers(e.gold, set(cols)), covers(e.gold, set(fk)), len(cols), len(fk), n_tokens(p) + n_tokens(t)))
    R = np.asarray(rows, float)
    print(f"LLM linking (Qwen2.5-32B): SRR={R[:,0].mean():.4f} Cbar={R[:,2].mean():.1f} | +FK closure SRR={R[:,1].mean():.4f} "
          f"Cbar={R[:,3].mean():.1f} | linking LLM tokens/q={R[:,4].mean():.0f}")
    json.dump({"ids": [e.ex_id for e in dev], "sets": {k: [[list(x) for x in s] for s in v] for k, v in sets.items()}},
              open(os.path.join(RES, "llmlink_sets.json"), "w"))
    with open(os.path.join(RES, "llmlink_genprompts.jsonl"), "w", encoding="utf-8") as f:
        seen = set()
        for k, v in sets.items():
            for e, s in zip(dev, v):
                gp = prompt_of(e, s if s else None)
                h = hashlib.md5(gp.encode()).hexdigest()
                if h not in seen:
                    seen.add(h); f.write(json.dumps({"h": h, "prompt": gp}, ensure_ascii=False) + "\n")
    print("gen prompts dumped", len(seen))
