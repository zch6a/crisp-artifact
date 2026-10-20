"""Build the extended schema-scaling workload (BIRD-Dev stride-3 subset, 512 questions; distractor tables from the
BIRD-train holdout schemas, jevschema.scale.Scaler, deterministic per (N, question)).
  python scale_build.py <scale_inputs.pkl> <out_dir> <tokenizer_dir> [sizes]
Writes <out_dir>/scale<N>.pkl (merged examples) and stats.json (columns, full-schema prompt tokens)."""
import json, os, pickle, sys
import numpy as np
from jevschema.scale import Scaler
from jevschema.tokens import build_gen_prompt, schema_prompt_text
from transformers import AutoTokenizer

inp, out, tokdir = sys.argv[1:4]
sizes = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,500,2000,5000,10000").split(",")]
d = pickle.load(open(inp, "rb")); dev = d["dev_sub"]
sc = Scaler.__new__(Scaler); sc.pool = d["pool"]
tok = AutoTokenizer.from_pretrained(tokdir)
os.makedirs(out, exist_ok=True); stats = {}
for n in sizes:
    exs = dev if n == 0 else [sc.merge(e, n) for e in dev]
    pickle.dump(exs, open(os.path.join(out, f"scale{n}.pkl"), "wb"))
    cols = [e.schema.n_columns() for e in exs]
    toks = [len(tok.encode(build_gen_prompt(e.db_id, schema_prompt_text(e.schema), e.question, e.evidence or ""))) for e in exs]
    stats[n] = {"questions": len(exs), "mean_cols": float(np.mean(cols)), "mean_full_prompt_tokens": float(np.mean(toks)),
                "max_full_prompt_tokens": int(max(toks)), "frac_full_over_32k": float(np.mean([t > 32768 for t in toks]))}
    print(n, stats[n], flush=True)
json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
