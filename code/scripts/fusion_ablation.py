"""融合分 ⊕ 学习式列概率：能不能把 BIRD 的 headline SRR 顶上去？

动机：两个信号各有各的毛病。
  * `bm25 ⊕ emb` 融合分 —— **相关性排序**好，AUROC 0.8772，但它没有"这列到底要不要"
    的绝对概念，分数是相对量。
  * `train_lex_classifier` 的学习式概率 —— AUROC 0.9058（比融合高），
    而且是**校准过的**（ECE 0.0057），但单独当排序器会输给融合
    （SRR@C=35：0.8481 vs 0.8814）。

所以两者**互补**：融合定顺序、概率定绝对量。这里直接扫 α：
    score = α · fused_norm + (1 − α) · prob
看有没有一个 α 能在同一 token 上把 SRR 推高。

⚠️ 泄漏检查（做之前必须先确认）：分类器是在 `bird_train`（6048 例 / 989 库，
**与 BIRD-Dev 零库重叠**）上学的，dev 只做一次评估。所以 dev 上的数字是干净的。
如果哪天改成在 dev 上拟合权重，这张表就作废 —— 注释写在这里就是为了防止这种事。

产物：每个 (α, k) 的 (SRR, C̄, tokens)，以及"等 token 下最优 α"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import mean_ci, paired_bootstrap, strict_recall  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import make_stage_tokens  # noqa: E402

from budget_predictor import load_prob_model  # noqa: E402
from train_lex_classifier import features as lex_features  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


def _load(kind, examples, fn):
    p = os.path.join(CACHE, f"bird_{kind}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    if fn is None:
        return [store.get(e.ex_id, {}) for e in examples]
    todo = [e for e in examples if e.ex_id not in store]
    if todo:
        os.makedirs(CACHE, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for e in todo:
                sc = fn(e)
                store[e.ex_id] = sc
                f.write(json.dumps({"id": e.ex_id,
                                    "s": {"|".join(k): v for k, v in sc.items()}},
                                   ensure_ascii=False) + "\n")
    return [store[e.ex_id] for e in examples]


def bm25_of(ex):
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def _norm(d, how="max"):
    v = np.asarray(list(d.values()), dtype=float)
    if how == "max":
        hi = v.max()
        return {k: float(x) for k, x in
                zip(d.keys(), v / hi if hi > 0 else np.zeros_like(v))}
    lo, hi = v.min(), v.max()
    return {k: float(x) for k, x in zip(d.keys(), (v - lo) / max(1e-9, hi - lo))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--closure", default="f1p5s15k1,f1p8k1,f1p0s15k1,f1p3s15k1,f1p5k1",
                    help="逗号分隔多个闭包配置：headline 数字来自 f1p8k1，"
                         "只跑一个闭包比出来的最优 α 换配置可能就不成立")
    ap.add_argument("--ks", default="8,10,12,15,18,20,25,30,35,40,50,60")
    ap.add_argument("--alphas", default="0,0.2,0.35,0.5,0.65,0.8,1.0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    cfgs = [(g, parse_closure(g)) for g in a.closure.split(",")]

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    n0 = len(ex)
    ex = [e for e in ex if e.gold is not None and e.gold.flat_pairs()]
    if n0 != len(ex):
        print(f"  剔除空 gold {n0 - len(ex)} 条")

    pm = load_prob_model(os.path.join(RESULTS, "lex_classifier_bird.json"))
    if pm is None:
        print("⚠️ 没有学习式概率模型，退出")
        return
    print(f"BIRD-Dev n={len(ex)}  closures={[g for g, _ in cfgs]}  "
          f"alphas={alphas}", flush=True)

    bm = _load("bird_bm25", ex, bm25_of)
    eb = _load("bird_embedding", ex, None)
    have_emb = any(len(d) > 0 for d in eb)

    fused, prob = [], []
    for e, b, d in zip(ex, bm, eb):
        pairs = e.schema.columns()
        bn = _norm({p: float(b.get(p, 0.0)) for p in pairs}, "max")
        dn = (_norm({p: float(d.get(p, 0.0)) for p in pairs}, "minmax")
              if have_emb else {p: 0.0 for p in pairs})
        fused.append(np.asarray([0.5 * bn[p] + 0.5 * dn[p] for p in pairs]))
        _, X, _ = lex_features(e, None)
        w, bias, mu, sd = pm
        prob.append(1.0 / (1.0 + np.exp(-np.clip((X - mu) / sd @ w + bias, -30, 30))))

    full_tok = float(np.mean([make_stage_tokens(e.schema, e.question, e.evidence)
                              .total_tokens for e in ex]))
    print(f"  全 schema 直给 {full_tok:.0f} tokens", flush=True)

    all_rows = []
    for gtag, cfg in cfgs:
        print(f"\n-- 闭包 {gtag} --")
        print(f"{'α':>6}" + "".join(f"{'k='+str(k):>9}" for k in ks))
        for al in alphas:
            line = f"{al:>6.2f}"
            for k in ks:
                srrs, cb, tk = [], [], []
                for i, e in enumerate(ex):
                    pairs = e.schema.columns()
                    s = al * fused[i] + (1.0 - al) * prob[i]
                    order = [pairs[j] for j in np.argsort(-s, kind="stable")]
                    kept = closure(e.schema, order, k, cfg)
                    srrs.append(strict_recall(e.gold, kept))
                    cb.append(len(kept))
                    tk.append(make_stage_tokens(e.schema, e.question, e.evidence,
                                                kept_columns=kept).gen_prompt_tokens)
                v = float(np.mean(srrs))
                all_rows.append({"closure": gtag, "alpha": al, "k": k,
                                 "srr": round(v, 4),
                                 "cbar": round(float(np.mean(cb)), 2),
                                 "tokens": round(float(np.mean(tk)), 1),
                                 "ci": [round(x, 4) for x in mean_ci(srrs)[1:3]],
                                 "_srr": srrs})
                line += f"{v:>9.4f}"
            print(line, flush=True)

    # 等 token 下谁最好：把 (tokens, srr) 摊平后取帕累托前沿
    print(f"\n== 帕累托前沿（x=tokens 越小越好，y=SRR 越大越好）==")
    pts = sorted(all_rows, key=lambda r: (r["tokens"], -r["srr"]))
    front, best = [], -1.0
    for r in pts:
        if r["srr"] > best + 1e-9:
            front.append(r); best = r["srr"]
    print(f"{'α':>6}{'k':>4}{'tokens':>9}{'C̄':>7}{'SRR':>9}  95%CI")
    for r in front:
        print(f"{r['alpha']:>6.2f}{r['k']:>4}{r['tokens']:>9.0f}{r['cbar']:>7.1f}"
              f"{r['srr']:>9.4f}  [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]")

    # α=1（纯融合，现状基线）vs 前沿上每个点：同 k 比 + 显著性
    base = {(r["closure"], r["k"]): r
            for r in all_rows if abs(r["alpha"] - 1.0) < 1e-9}
    print(f"\n== 与纯融合（α=1.0）同闭包同 k 对比，配对 bootstrap ==")
    print(f"{'闭包':<11}{'α':>6}{'k':>4}{'fused SRR':>11}{'本 SRR':>9}{'Δ':>9}{'p':>9}")
    for r in all_rows:
        if abs(r["alpha"] - 1.0) < 1e-9:
            continue
        b = base.get((r["closure"], r["k"]))
        if b is None:
            continue
        pb = paired_bootstrap(b["_srr"], r["_srr"], n_boot=2000)
        flag = "  << 赢" if (pb["diff"] > 0 and pb["p_boot"] < 0.05) else ""
        if abs(pb["diff"]) >= 0.005:
            print(f"{r['closure']:<11}{r['alpha']:>6.2f}{r['k']:>4}"
                  f"{b['srr']:>11.4f}{r['srr']:>9.4f}"
                  f"{pb['diff']:>+9.4f}{pb['p_boot']:>9.4f}{flag}")

    # ------------------------------------------------------------ headline 对照
    print(f"\n== headline 口径：达到 SRR≥0.974（对齐 AutoLink）的最省配置 ==")
    for al in sorted({r["alpha"] for r in all_rows}):
        ok = [r for r in all_rows
              if abs(r["alpha"] - al) < 1e-9 and r["srr"] >= 0.974]
        if ok:
            r = min(ok, key=lambda z: z["tokens"])
            print(f"  α={al:<5.2f} {r['closure']:<10} k={r['k']:<3} "
                  f"SRR={r['srr']:.4f} tokens={r['tokens']:.0f} "
                  f"省 {100*(1 - r['tokens']/full_tok):.1f}%")
        else:
            m = max((r for r in all_rows if abs(r["alpha"] - al) < 1e-9),
                    key=lambda z: z["srr"])
            print(f"  α={al:<5.2f} 不可达（最高 {m['srr']:.4f} @ "
                  f"{m['closure']} k={m['k']}, {m['tokens']:.0f} tok）")

    path = a.out or os.path.join(RESULTS, "fusion_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(ex), "closure": cfg.tag(),
                   "full_schema_tokens": round(full_tok, 1),
                   "rows": [{kk: vv for kk, vv in r.items() if kk != "_srr"}
                            for r in all_rows],
                   "front": [{kk: vv for kk, vv in r.items() if kk != "_srr"}
                             for r in front]}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
