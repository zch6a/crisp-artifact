"""Schema-scaling results: SRR/Cbar/tokens at fixed budgets and certified (protocol A) per size."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, full_tokens, fused_scores, load_examples, read_meta, read_scores
from v2_frontier import combine
from jevschema.certify import calibrate, evaluate_budgets, fixed_family
from jevschema.scale import Scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="250,500,1000,2000")
    ap.add_argument("--eps", default="0.02,0.05")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--src", default="scale{n}_block")
    ap.add_argument("--out", default="scale_eval.json")
    ap.add_argument("--kind", default="scale", choices=["scale", "hard"])
    a = ap.parse_args()
    dev0 = load_examples("dev")
    sc = Scaler(load_examples("holdout"))
    rng = np.random.default_rng(0)
    out = []
    for n in [int(x) for x in a.sizes.split(",")]:
        bm = read_scores(os.path.join(CACHE, f"{a.kind}{n}_bm25.jsonl"))
        em = read_scores(os.path.join(CACHE, f"{a.kind}{n}_emb.jsonl"))
        bk = read_scores(os.path.join(CACHE, a.src.format(n=n) + ".jsonl"))
        meta = read_meta(os.path.join(CACHE, a.src.format(n=n) + ".jsonl"))
        ids = [e.ex_id for e in dev0 if e.ex_id in bk and e.ex_id in em]
        if len(ids) < 100 or len(bk) < 100:
            print(f"N={n}: only {len(ids)} scored, skip"); continue
        keep = set(ids)
        exs = ([sc.merge(e, n) for e in dev0 if e.ex_id in keep] if a.kind == "scale"
               else [e for e in load_examples(f"hard{n}") if e.ex_id in keep])
        s0 = [fused_scores(e, bm[e.ex_id], em[e.ex_id]) for e in exs]
        s1 = combine(exs, s0, [bk[e.ex_id] for e in exs], 0.02)
        full = full_tokens(exs)
        enc = np.mean([meta[e.ex_id]["enc_tokens"] or 0 for e in exs]); wall = np.mean([meta[e.ex_id]["wall_s"] or 0 for e in exs])
        ncol = np.mean([len(e.schema.columns()) for e in exs])
        print(f"\nN={n} n={len(exs)} cols={ncol:.0f} full-schema prompt tok={full.mean():.0f} block enc/q={enc:.0f} wall/q={wall:.2f}s", flush=True)
        for name, s, cfg in (("fused", s0, "f1p5s15k1"), ("block", s1, "f1p5s15k1")):
            v = View(exs, s, cfg)
            for k in (10, 15, 25, 35):
                srr = float(np.mean(v.r <= k)); cb = float(np.mean([v.size(i, k) for i in range(len(exs))]))
                tk = float(np.mean([v.tokens(i, k) for i in range(0, len(exs), 3)]))
                out.append({"N": n, "scorer": name, "k": k, "srr": srr, "cbar": cb, "tok": tk, "full_tok": float(full.mean())})
                print(f"  {name:6s} k={k:<3} SRR={srr:.4f} Cbar={cb:5.1f} tok={tk:5.0f} ({100*(1-tk/full.mean()):.1f}% saved)", flush=True)
            fam = fixed_family(v.r, v.n_cols)
            for eps in [float(x) for x in a.eps.split(",")]:
                cov, cbs, tks = [], [], []
                for rep in range(a.reps):
                    p = rng.permutation(len(exs)); cal, te = p[: len(exs) // 2], p[len(exs) // 2:]
                    lam = calibrate(fam, cal, eps)
                    ks, cv = evaluate_budgets(fam, lam, te, v.r)
                    cov.append(cv.mean()); cbs.append(np.mean([v.size(i, kk) for i, kk in zip(te, ks)]))
                    if rep < 5:
                        tks.append(np.mean([v.tokens(i, kk) for i, kk in zip(te[:200], ks[:200])]))
                row = {"N": n, "scorer": name, "eps": eps, "cov": float(np.mean(cov)), "cbar": float(np.mean(cbs)),
                       "tok": float(np.mean(tks)), "full_tok": float(full.mean()), "enc": float(enc) if name == "block" else 0.0}
                out.append(row)
                print(f"  {name:6s} certified eps={eps}: cov={row['cov']:.4f} Cbar={row['cbar']:.1f} tok={row['tok']:.0f} "
                      f"({100*(1-row['tok']/full.mean()):.1f}% saved)", flush=True)
    json.dump(out, open(os.path.join(RES, a.out), "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
