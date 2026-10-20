"""Spider-Dev (992, empty gold removed): zero-shot transfer of the BIRD-trained block reader."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, full_tokens, fused_scores, lexical, load_examples, read_meta, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, evaluate_budgets, fixed_family

exs = load_examples("spider")
bm, em = lexical("spider")
B = read_scores(os.path.join(CACHE, "block_t10_spider.jsonl")); meta = read_meta(os.path.join(CACHE, "block_t10_spider.jsonl"))
Jv = read_scores(os.path.join(CACHE, "jev_t10_spider.jsonl"))
exs = [e for e in exs if e.ex_id in B and (not Jv or e.ex_id in Jv)]
s0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in exs]
s1 = combine(exs, s0, [B[e.ex_id] for e in exs], 0.02)
full = full_tokens(exs)
print(f"Spider-Dev n={len(exs)} cols={np.mean([len(e.schema.columns()) for e in exs]):.1f} full tok={full.mean():.0f} "
      f"block enc/q={np.mean([m['enc_tokens'] for m in meta.values()]):.0f} wall/q={np.mean([m['wall_s'] for m in meta.values()]):.3f}s")
out, rng = [], np.random.default_rng(0)
readers = [("fused", s0), ("block", s1)]
if Jv:
    readers.append(("jev", combine(exs, s0, [Jv[e.ex_id] for e in exs], 0.02)))
for name, s in readers:
    for cfg in ("f1", "f1k1", "f1p5s15k1"):
        v = View(exs, s, cfg)
        line = []
        for k in (3, 5, 8, 10, 15):
            srr = float(np.mean(v.r <= k)); cb = float(np.mean([v.size(i, k) for i in range(len(exs))]))
            line.append(f"k{k}:{srr:.4f}/{cb:.1f}")
            out.append({"scorer": name, "cfg": cfg, "k": k, "srr": srr, "cbar": cb})
        fam = fixed_family(v.r, v.n_cols)
        cert = []
        for eps in (0.01, 0.02, 0.05):
            cov, cbs, tks = [], [], []
            for rep in range(100):
                p = rng.permutation(len(exs)); cal, te = p[: len(exs) // 2], p[len(exs) // 2:]
                lam = calibrate(fam, cal, eps); ks, cv = evaluate_budgets(fam, lam, te, v.r)
                cov.append(cv.mean()); cbs.append(np.mean([v.size(i, k) for i, k in zip(te, ks)]))
                if rep < 5:
                    tks.append(np.mean([v.tokens(i, k) for i, k in zip(te, ks)]) / np.mean(full[te]))
            cert.append(f"eps{eps}:cov={np.mean(cov):.4f} Cbar={np.mean(cbs):.1f} tok={100*np.mean(tks):.0f}%")
            out.append({"scorer": name, "cfg": cfg, "eps": eps, "cov": float(np.mean(cov)), "cbar": float(np.mean(cbs)), "tok_frac": float(np.mean(tks))})
        print(f"{name:5s} {cfg:10s} SRR/Cbar {' '.join(line)}\n      certified {' | '.join(cert)}", flush=True)
json.dump(out, open(os.path.join(RES, "spider_eval.json"), "w"), indent=1)
