"""Fixed-budget frontiers on BIRD-Dev: SRR vs Cbar vs prompt tokens, per scorer x closure.

Scorers: fused (tier-0), laya10 (tier-1, top-10 table gate), laya10+beta*fused (beta chosen
on BIRD-train holdout by mean log oracle depth).  Also reports the points that matter for the
SOTA comparison: best SRR at Cbar <= 35.8 (AutoLink self-report) and SRR at AutoLink-like
precision.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, full_tokens, fused_scores, lexical, load_examples, read_scores
from jevschema.metrics import mean_ci


def mm(d, pairs):
    v = np.asarray([d.get(p, 0.0) for p in pairs], dtype=float)
    return (v - v.min()) / max(1e-9, v.max() - v.min()) if len(v) else v


def combine(exs, s0, s1, beta):
    out = []
    for e, a, b in zip(exs, s0, s1):
        pairs = e.schema.columns()
        n0 = mm(a, pairs)
        out.append({p: (b[p] if p in b else -1.0) + beta * x for p, x in zip(pairs, n0)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="f1,f1k1,f1p0s15k1,f1p3s15k1,f1p5s15k1,f1p8k1,f1p8s15k1")
    ap.add_argument("--ks", default="3,4,5,6,8,10,12,15,18,20,25,30,35,40,50")
    ap.add_argument("--betas", default="0,0.02,0.05,0.1,0.2,0.3,0.5,1.0")
    ap.add_argument("--tier1", default="laya_t10")
    a = ap.parse_args()
    t0 = time.time()
    dev, hold = load_examples("dev"), load_examples("holdout")
    bm, em = lexical("dev"); hbm, hem = lexical("holdout")
    f_d = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
    f_h = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
    ld = read_scores(os.path.join(CACHE, f"{a.tier1}_dev.jsonl")); lh = read_scores(os.path.join(CACHE, f"{a.tier1}_holdout.jsonl"))
    l_d = [ld.get(e.ex_id, {}) for e in dev]; l_h = [lh.get(e.ex_id, {}) for e in hold]
    betas = [float(x) for x in a.betas.split(",")]
    sel = {}
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)
    for cfg in ("f1p5s15k1", "f1p8k1"):
        best = None
        for b in betas:
            vh = View(hold, combine(hold, f_h, l_h, b), cfg)
            vd = View(dev, combine(dev, f_d, l_d, b), cfg)
            mh, md = float(np.mean(np.log(vh.r))), float(np.mean(np.log(vd.r)))
            print(f"  beta sweep {cfg} beta={b:<5} holdout mean_log_r={mh:.4f} | dev mean_log_r={md:.4f} "
                  f"srr@10={np.mean(vd.r <= 10):.4f} srr@15={np.mean(vd.r <= 15):.4f} srr@25={np.mean(vd.r <= 25):.4f}", flush=True)
            if best is None or mh < best[1]:
                best = (b, mh)
        sel[cfg] = best[0]
    beta = sel["f1p5s15k1"]
    print(f"beta* (holdout) = {sel}", flush=True)
    scorers = {"fused": f_d, "laya10": l_d, "laya10+fused": combine(dev, f_d, l_d, beta)}
    ks = [int(x) for x in a.ks.split(",")]
    full = full_tokens(dev)
    rows = []
    for name, sc in scorers.items():
        for cfg in a.cfgs.split(","):
            v = View(dev, sc, cfg)
            for k in ks:
                cov = (v.r <= k).astype(float)
                size = np.asarray([v.size(i, k) for i in range(len(dev))], float)
                tok = np.asarray([v.tokens(i, k) for i in range(len(dev))], float)
                lo, hi = mean_ci(list(cov))[1:3]
                prec = np.mean([len(dev[i].gold.flat_pairs()) / max(1, s) for i, s in enumerate(size)])
                rows.append({"scorer": name, "cfg": cfg, "k": k, "srr": float(cov.mean()), "ci": [lo, hi],
                             "cbar": float(size.mean()), "tok": float(tok.mean()), "prec": float(prec),
                             "tok_saving": float(1 - tok.sum() / full.sum())})
            print(f"  {name:13s} {cfg:10s} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nfull schema tokens/query = {full.mean():.1f}")
    for name in scorers:
        R = [r for r in rows if r["scorer"] == name]
        pts = sorted(R, key=lambda r: (r["cbar"], -r["srr"]))
        front, best = [], -1
        for r in pts:
            if r["srr"] > best + 1e-9:
                front.append(r); best = r["srr"]
        print(f"\n== {name} Cbar-frontier ==")
        for r in front:
            print(f"  {r['cfg']:10s} k={r['k']:<3} Cbar={r['cbar']:5.1f} tok={r['tok']:5.0f} prec={r['prec']:.3f} "
                  f"SRR={r['srr']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]")
        ok = [r for r in R if r["cbar"] <= 35.8]
        if ok:
            m = max(ok, key=lambda r: r["srr"])
            print(f"  -> best SRR at Cbar<=35.8: {m['srr']:.4f} ({m['cfg']} k={m['k']} Cbar={m['cbar']:.1f} tok={m['tok']:.0f})")
        for tgt in (0.974, 0.9889, 0.99):
            hit = [r for r in R if r["srr"] >= tgt]
            if hit:
                m = min(hit, key=lambda r: r["cbar"])
                print(f"  -> min Cbar for SRR>={tgt}: {m['cbar']:.1f} ({m['cfg']} k={m['k']} tok={m['tok']:.0f} prec={m['prec']:.3f})")
    json.dump({"beta": sel, "full_tokens": float(full.mean()), "rows": rows},
              open(os.path.join(RES, f"frontier_dev_{a.tier1}.json"), "w"), indent=1)
    print(f"saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
