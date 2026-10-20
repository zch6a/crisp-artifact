"""Gate cascade: when Jev's best probability inside the structural gate is low (likely gate miss),
read the next tables (ranks 21-40 by BM25 + their 1-hop FK neighbours) in a second Jev request.
Writes merged scores to cache/jev_fk2_<split>.jsonl (all questions; routed ones extended)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_score_jev as SJ
from v2data import CACHE, load_examples, read_scores
from jevschema.gating import bm25_cols, ranked_tables
from jevschema.risk import VIEWS
from jevschema.summary import SummaryConfig, build_state

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="scale2000")
ap.add_argument("--th", type=float, default=0.7)
ap.add_argument("--extra", type=int, default=20)
a = ap.parse_args()
key = sys.stdin.readline().strip()
J = read_scores(os.path.join(CACHE, f"jev_fk_{a.split}.jsonl"))
exs = [e for e in load_examples(a.split) if e.ex_id in J]
out_p = os.path.join(CACHE, f"jev_fk2_{a.split}.jsonl")
done = {json.loads(l)["id"] for l in open(out_p, encoding="utf-8")} if os.path.exists(out_p) else set()
spent = json.load(open(SJ.USAGE))
n_route = 0
with open(out_p, "a", encoding="utf-8") as f:
    for e in exs:
        if e.ex_id in done:
            continue
        s = dict(J[e.ex_id]); extra_tok = 0
        if max(s.values()) < a.th:
            n_route += 1
            gated = {t for t, _ in s}
            rk = [t for t in ranked_tables(bm25_cols(e)) if t not in gated][: a.extra]
            adj = {}
            for fk in e.schema.fks:
                adj.setdefault(fk.src_table, set()).add(fk.dst_table); adj.setdefault(fk.dst_table, set()).add(fk.src_table)
            tabs = set(rk)
            for t in list(rk):
                tabs |= adj.get(t, set())
            tabs -= gated
            state, _ = build_state(e.question_block(), e.schema, SummaryConfig(), tables=sorted(tabs))
            cols = [(t, c) for (t, c) in e.schema.columns() if t in tabs]
            for i in range(0, len(cols), 128):
                part = cols[i:i + 128]
                qs = {f"q{j}": {"type": "noul", "instructions": VIEWS["ask"].format(t=t, c=c)} for j, (t, c) in enumerate(part)}
                resp = SJ.post(key, {"model": "jev-latest", "state": state, "questions": qs})
                for j, (t, c) in enumerate(part):
                    s[(t, c)] = float(resp["answers"][f"q{j}"]["noul"])
                extra_tok += int((resp.get("usage") or {}).get("input_tokens", 0))
            spent["in_tok"] += extra_tok; spent["usd"] = spent["in_tok"] * SJ.PRICE
            json.dump(spent, open(SJ.USAGE, "w"))
        f.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": v for (t, c), v in s.items()}, "extra_tok": extra_tok}) + "\n")
        f.flush()
print(f"routed {n_route}/{len(exs)}; total spent ${spent['usd']:.4f}")
