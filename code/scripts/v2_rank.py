"""Tier-0 learned rankers on BIRD-train pool: log-loss LR vs bottleneck-aware (SRR-aligned).

Hyper-parameters are selected on BIRD-train holdout (mean log oracle depth);
BIRD-Dev is evaluated once.  Outputs results/rank_tier0.json and cache/rank_*_{dev,holdout}.jsonl.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples
from jevschema.bottleneck import Packed, apply_linear, col_features, implied_mask, train_linear


def rows_for(exs, bm, em, max_cols=300):
    rows, feats = [], []
    for e in exs:
        pairs, X = col_features(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {}))
        feats.append((pairs, X))
        if len(pairs) > max_cols:
            continue
        gold = e.gold.flat_pairs()
        pos = np.asarray([p in gold for p in pairs])
        if not pos.any():
            continue
        rows.append((X, pos, implied_mask(e, pairs)))
    return rows, feats


def scores_from(M, feats):
    return [{p: float(s) for p, s in zip(pairs, apply_linear(M, X))} for pairs, X in feats]


def depth_stats(exs, scores, cfg):
    v = View(exs, scores, cfg)
    out = {"mean_log_r": float(np.mean(np.log(v.r))), "median_r": float(np.median(v.r))}
    for k in (8, 12, 15, 20, 25, 35):
        out[f"srr@{k}"] = float(np.mean(v.r <= k))
    for k in (15, 35):
        out[f"cbar@{k}"] = float(np.mean([v.size(i, k) for i in range(len(exs))]))
    return out


def dump(path, exs, scores):
    with open(path, "w", encoding="utf-8") as f:
        for e, s in zip(exs, scores):
            f.write(json.dumps({"id": e.ex_id, "s": {f"{t}|{c}": v for (t, c), v in s.items()}}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="f1p5s15k1")
    ap.add_argument("--grid", default="0.05:0.05,0.1:0.05,0.2:0.1,0.1:0.02")
    a = ap.parse_args()
    t0 = time.time()
    pool, hold, dev = load_examples("pool"), load_examples("holdout"), load_examples("dev")
    tbm, tem = lexical("holdout")
    dbm, dem = lexical("dev")
    rows_p, _ = rows_for(pool, tbm, tem)
    _, feats_h = rows_for(hold, tbm, tem)
    _, feats_d = rows_for(dev, dbm, dem)
    P = Packed(rows_p)
    print(f"pool rows={len(rows_p)} padded={P.X.shape}  ({time.time()-t0:.0f}s)", flush=True)
    out = {"cfg": a.cfg, "models": {}, "holdout": {}, "dev": {}}
    base_h = [fused_scores(e, tbm.get(e.ex_id, {}), tem.get(e.ex_id, {})) for e in hold]
    base_d = [fused_scores(e, dbm.get(e.ex_id, {}), dem.get(e.ex_id, {})) for e in dev]
    out["holdout"]["fused"] = depth_stats(hold, base_h, a.cfg)
    out["dev"]["fused"] = depth_stats(dev, base_d, a.cfg)
    print("fused holdout", out["holdout"]["fused"], "\nfused dev    ", out["dev"]["fused"], flush=True)
    cands = [("logloss", {"loss": "logloss"})]
    for g in a.grid.split(","):
        T, tau = (float(x) for x in g.split(":"))
        cands.append((f"bneck_T{T}_tau{tau}", {"loss": "bottleneck", "T": T, "tau": tau}))
    best, best_v = None, 1e9
    for name, kw in cands:
        M = train_linear(P, **kw)
        sh = depth_stats(hold, scores_from(M, feats_h), a.cfg)
        out["holdout"][name] = sh
        out["models"][name] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in M.items()}
        print(f"{name:22s} holdout mean_log_r={sh['mean_log_r']:.4f} srr@15={sh['srr@15']:.4f} "
              f"srr@25={sh['srr@25']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if name != "logloss" and sh["mean_log_r"] < best_v:
            best, best_v = name, sh["mean_log_r"]
    for name in ("logloss", best):
        M = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out["models"][name].items()}
        sd = scores_from(M, feats_d)
        out["dev"][name] = depth_stats(dev, sd, a.cfg)
        print(f"{name:22s} dev {out['dev'][name]}", flush=True)
        tag = "logloss" if name == "logloss" else "bneck"
        dump(os.path.join(CACHE, f"rank_{tag}_dev.jsonl"), dev, sd)
        dump(os.path.join(CACHE, f"rank_{tag}_holdout.jsonl"), hold, scores_from(M, feats_h))
    out["best"] = best
    json.dump(out, open(os.path.join(RES, "rank_tier0.json"), "w"), indent=1)
    print(f"saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
