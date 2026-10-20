"""Tier-1 (System-One laya) column scoring for v2 -- resumable, bf16, memory-capped.

Scores every column of the BM25 top-T tables with the pointwise laya reader
(template "asks_for", i.e. risk.VIEWS["ask"]), exactly the protocol of laya_pareto.py,
but additionally logs the *physical* encoder cost per question:
    enc_tokens  = sum over scored columns of len(state+question sequence)   (what the GPU encodes)
    n_seq       = number of encoder sequences (= columns scored)
    wall_s      = wall-clock seconds on this GPU
The README's "state once + questions" accounting under-counts pointwise laya: every
column is a separate sequence that re-encodes the whole state.

Usage:  v2_score_laya.py --split dev|holdout [--n 0] [--top-tables 10]
Output: <CRISP_CACHE>/laya_t{T}_{split}.jsonl  (one line per question, append-only)
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

from jevschema.bird_data import load_bird_dev, load_bird_schemas
from jevschema.bird_train import load_bird_train
from jevschema.risk import VIEWS
from jevschema.summary import SummaryConfig, bm25_scores, build_state

from jevschema.paths import CACHE, MODELS, RESULTS


def bm25_of(ex):
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def pick_tables_cheap(bm, top_t):
    best = {}
    for (t, c), v in bm.items():
        if v > best.get(t, -1e9):
            best[t] = v
    return [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_t]]


def load_split(split, n):
    if split == "dev":
        ex = load_bird_dev(schemas=load_bird_schemas(), shuffle=False)
    else:
        allx, _ = load_bird_train(limit=None, log_every=0)
        sp = json.load(open(os.path.join(RESULTS, "bird_train_split.json")))
        keep = set(sp["holdout_ids"])
        ex = [e for e in allx if e.ex_id in keep]
    return ex[:n] if n > 0 else ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--top-tables", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--model-path", default=os.path.join(MODELS, "laya_ft_bird"))
    ap.add_argument("--mem-fraction", type=float, default=0.40)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    a = ap.parse_args()

    out_path = os.path.join(CACHE, f"laya_t{a.top_tables}_{a.split}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    ex = load_split(a.split, a.n)
    todo = [e for e in ex if e.ex_id not in done]
    print(f"split={a.split} n={len(ex)} done={len(done)} todo={len(todo)} -> {out_path}", flush=True)
    if not todo:
        return

    if torch.cuda.is_available() and a.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    from jevschema.backends.laya_backend import LayaBackend
    be = LayaBackend(model_path=a.model_path, max_len=a.max_len, batch_tokens=a.batch_tokens)
    tmpl = VIEWS["ask"]
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.amp == "bf16" else None
    t_all = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, e in enumerate(todo):
            t0 = time.time()
            bm = bm25_of(e)
            gate = set(pick_tables_cheap(bm, a.top_tables)) if a.top_tables > 0 else None
            state, tables = build_state(e.question_block(), e.schema, SummaryConfig())
            keep = gate if gate is not None else set(tables)
            cols = [(t, c) for (t, c) in e.schema.columns() if t in keep]
            stmts = [tmpl.format(t=t, c=c) for t, c in cols]
            from jevschema.backends.base import TypedQuestion
            qs = [TypedQuestion(s, ("no", "yes"), "noul") for s in stmts]
            enc_tok = int(sum(len(it["ids"]) for it in be._items(state, qs))) if qs else 0
            if qs:
                if amp_ctx is not None:
                    with amp_ctx():
                        ps = be.judge_batch(state, stmts)
                else:
                    ps = be.judge_batch(state, stmts)
            else:
                ps = []
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rec = {"id": e.ex_id, "s": {f"{t}|{c}": float(p) for (t, c), p in zip(cols, ps)},
                   "n_seq": len(qs), "enc_tokens": enc_tok, "wall_s": round(time.time() - t0, 3),
                   "gate": sorted(keep)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t_all
                print(f"  {i+1}/{len(todo)}  {el/60:.1f} min  {(i+1)/el*60:.1f} q/min  "
                      f"last enc_tok={enc_tok} n_seq={len(qs)}", flush=True)
    print(f"finished {len(todo)} in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
