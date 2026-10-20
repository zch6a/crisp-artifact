"""Sanity-check LLM-linking parse quality and compare CRISP at matched Cbar."""
import json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_frontier import combine
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_llm_link.py"), encoding="utf-8").read()
_ns = {"__file__": __file__}
exec(_src.split("\nap = argparse")[0], _ns)
link_prompt, parse_cols = _ns["link_prompt"], _ns["parse_cols"]
dev = load_examples("dev")
got = {json.loads(l)["h"]: json.loads(l)["text"] for l in open(os.path.join(CACHE, "llmlink_out_qwen32b.jsonl"), encoding="utf-8")}
n_lines, n_parsed, empty = [], [], 0
for e in dev[:400]:
    t = got.get(hashlib.md5(link_prompt(e).encode()).hexdigest(), "")
    lines = [l for l in t.splitlines() if "." in l]
    cols = parse_cols(t, e.schema)
    n_lines.append(len(lines)); n_parsed.append(len(cols)); empty += (len(cols) == 0)
print(f"lines-with-dot/q={np.mean(n_lines):.2f} parsed/q={np.mean(n_parsed):.2f} empty={empty}/400")
for e in dev[:3]:
    print("---", e.question[:80]); print(got.get(hashlib.md5(link_prompt(e).encode()).hexdigest(), "")[:300])
bm, em = lexical("dev"); f0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
J = read_scores(os.path.join(CACHE, "jev_fkm_dev.jsonl"))
for cfg in ("f1", "f1k1"):
    v = View(dev, combine(dev, f0, [J[e.ex_id] for e in dev], 0.02), cfg)
    for k in (1, 2, 3, 4):
        print(f"CRISP Jev {cfg} k={k}: SRR={np.mean(v.r <= k):.4f} Cbar={np.mean([v.size(i, k) for i in range(len(dev))]):.1f}")
