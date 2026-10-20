"""End-to-end EX on BIRD-Dev with certified schema sets (2-fold cross-fitted calibration).

For every system the kept set of a question in fold A uses lam_hat calibrated on fold B and
vice versa, so each set carries the split-conformal guarantee without using its own label.
Same prompt template as tokens.build_gen_prompt / e2e_bird.py, so old generations are reused
through the md5(prompt) cache.  Generator fixed (temperature 0).
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_meta, read_scores
from v2_cascade import tier1_scores, feats
from jevschema.certify import calibrate, fixed_family, kept_at
from jevschema.difficulty import Difficulty
from jevschema.tokens import build_gen_prompt, n_tokens, schema_prompt_text


SAMPLES = 0
EPS_LIST = (0.02, 0.05, 0.10, 0.20)


def prompt_of(e, kept):
    if kept is None:
        txt = schema_prompt_text(e.schema, with_samples=SAMPLES)
    else:
        txt = schema_prompt_text(e.schema, tables=sorted({t for t, _ in kept}), columns=kept, with_samples=SAMPLES)
    return build_gen_prompt(e.db_id, txt, e.question, e.evidence or "")


def crossfit(view, fam, eps, idx_all, fold):
    """kept sets for idx_all; lam for fold-0 items calibrated on fold-1 items and vice versa."""
    out = {}
    for f in (0, 1):
        cal = [i for i in idx_all if fold[i] != f]
        te = [i for i in idx_all if fold[i] == f]
        if not te:
            continue
        lam = calibrate(fam, cal, eps) if cal else float("inf")
        for i in te:
            out[i] = kept_at(view.exs[i].schema, view.orders[i], fam.budget(i, lam), view.cfg)
    return out


def build_systems(a, dev, hold):
    bm, em = lexical("dev"); hbm, hem = lexical("holdout")
    s0d = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
    s0h = [fused_scores(e, hbm.get(e.ex_id, {}), hem.get(e.ex_id, {})) for e in hold]
    ld = read_scores(os.path.join(CACHE, f"{a.tier1}_dev.jsonl"))
    s1d = tier1_scores(dev, s0d, [ld[e.ex_id] for e in dev], a.beta)
    v0 = View(dev, s0d, "f1p5s15k1"); v1 = View(dev, s1d, "f1k1")
    f0, f1 = fixed_family(v0.r, v0.n_cols), fixed_family(v1.r, v1.n_cols)
    fold = np.random.default_rng(0).permutation(len(dev)) % 2
    all_i = list(range(len(dev)))
    S = {"full": [None] * len(dev)}
    for eps in EPS_LIST:
        k0 = crossfit(v0, f0, eps, all_i, fold); S[f"t0_e{eps}"] = [k0[i] for i in all_i]
        k1 = crossfit(v1, f1, eps, all_i, fold); S[f"t1_e{eps}"] = [k1[i] for i in all_i]
    # cascade: difficulty-routed, threshold from holdout (target 40% escalation)
    v0h = View(hold, s0h, "f1p5s15k1")
    diff = Difficulty().fit(feats(v0h), np.log(v0h.r))
    tau = float(np.quantile(diff.predict(feats(v0h))[0], 1 - a.route_frac))
    R = diff.predict(feats(v0))[0] > tau
    for eps in (0.02,):
        kk = {}
        kk.update(crossfit(v0, f0, eps, [i for i in all_i if not R[i]], fold))
        kk.update(crossfit(v1, f1, eps, [i for i in all_i if R[i]], fold))
        S[f"cascade_e{eps}"] = [kk[i] for i in all_i]
    # oracle upper bound: gold columns + FK endpoints between gold tables
    orc = []
    for e in dev:
        g = set(e.gold.flat_pairs()); tabs = {t for t, _ in g}
        for fk in e.schema.fks:
            if fk.src_table in tabs and fk.dst_table in tabs:
                g.add((fk.src_table, fk.src_col)); g.add((fk.dst_table, fk.dst_col))
        orc.append(sorted(g))
    S["oracle"] = orc
    return S, float(R.mean())


def paired(a_flags, b_flags, n_boot=5000, seed=0):
    idx = [i for i, (x, y) in enumerate(zip(a_flags, b_flags)) if x is not None and y is not None]
    x = np.asarray([a_flags[i] for i in idx], float); y = np.asarray([b_flags[i] for i in idx], float)
    d = y - x
    rng = np.random.default_rng(seed)
    bs = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    p = float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    b01, b10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    return {"diff": float(d.mean()), "p_boot": min(1.0, p), "wins": b01, "losses": b10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--tier1", default="laya_t10")
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--route-frac", type=float, default=0.4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--systems", default="full,t1_e0.02,cascade_e0.02,t0_e0.02,t1_e0.05,t0_e0.05,oracle")
    a = ap.parse_args()
    from jevschema.generator import OllamaGenerator
    from jevschema.exec_eval import evaluate_execution
    t0 = time.time()
    global SAMPLES
    SAMPLES = a.samples
    dev, hold = load_examples("dev", samples=a.samples), load_examples("holdout")
    S, esc = build_systems(a, dev, hold)
    sel = list(range(0, len(dev), a.stride))
    print(f"systems built ({time.time()-t0:.0f}s) cascade escalation={esc:.3f} n={len(sel)}", flush=True)
    gen = OllamaGenerator(model=a.model, timeout_s=300)
    gpath = os.path.join(CACHE, f"gen_{a.model.replace(':', '_')}.jsonl")
    gc = {}
    if os.path.exists(gpath):
        for line in open(gpath, encoding="utf-8"):
            r = json.loads(line); gc[r["h"]] = r["sql"]
    if not a.cache_only and not gen.generate_sql("Answer with a single SQL: SELECT 1;").strip():
        raise SystemExit(f"generator smoke test failed: {gen.last_error}")
    out = {"model": a.model, "tier1": a.tier1, "n": len(sel), "escalation": esc, "systems": {}}
    flags_by, gold_cache = {}, {}
    for name in a.systems.split(","):
        prompts = [prompt_of(dev[i], S[name][i]) for i in sel]
        sqls, hits, t1 = [], 0, time.time()
        for j, p in enumerate(prompts):
            h = hashlib.md5(p.encode("utf-8")).hexdigest()
            if h in gc:
                hits += 1
            else:
                if a.cache_only:
                    raise SystemExit(f"cache miss for {name} (cache-only mode)")
                s = gen.generate_sql(p)
                if not s.strip():
                    raise SystemExit(f"generation failed (not cached): {gen.last_error}")
                gc[h] = s
                with open(gpath, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"h": h, "sql": s}, ensure_ascii=False) + "\n")
            sqls.append(gc[h])
            if (j + 1) % 200 == 0:
                print(f"    {name} {j+1}/{len(prompts)} cache_hits={hits} {(time.time()-t1)/60:.1f} min", flush=True)
        fl = []
        rep = evaluate_execution([(dev[i].db_id, dev[i].sql, s, name) for i, s in zip(sel, sqls)],
                                 gold_cache=gold_cache, flags=fl)
        flags_by[name] = fl
        out.setdefault("flags", {})[name] = fl
        kept = [S[name][i] for i in sel]
        srr = float(np.mean([1.0 if k is None or all(req & set(k) for req in dev[i].gold.requirements) else 0.0
                             for i, k in zip(sel, kept)]))
        cbar = float(np.mean([len(k) if k is not None else len(dev[i].schema.columns()) for i, k in zip(sel, kept)]))
        ptok = float(np.mean([n_tokens(p) for p in prompts]))
        out["systems"][name] = {"ex": rep.correct / max(1, rep.n - rep.gold_fail), "correct": rep.correct,
                                "n_eval": rep.n - rep.gold_fail, "pred_fail": rep.pred_fail, "srr": srr,
                                "cbar": cbar, "prompt_tok": ptok, "cache_hits": hits}
        v = out["systems"][name]
        print(f"  {name:14s} EX={v['ex']:.4f} ({v['correct']}/{v['n_eval']}) SRR={srr:.4f} Cbar={cbar:.1f} "
              f"prompt_tok={ptok:.0f} hits={hits} ({(time.time()-t1)/60:.1f} min)", flush=True)
        if "full" in flags_by and name != "full":
            v["vs_full"] = paired(flags_by["full"], fl)
            print(f"      vs full: {v['vs_full']}", flush=True)
        json.dump(out, open(os.path.join(RES, f"e2e_{a.model.replace(':', '_')}_{a.tier1}_s{a.stride}{'_samp' + str(a.samples) if a.samples else ''}.json"), "w"), indent=1)
    print(f"done ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
