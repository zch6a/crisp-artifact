"""排序策略：把逻辑回归掰回 top-k，再与融合分做无参数名次融合。

问题：逻辑回归在 dev 上 AUROC 0.9058（高于融合的 0.8772），但 top-k 排序更差
（C=35 上 SRR 0.8481 vs 0.8814）。原因是它优化全局对数似然，负例占 90%，
分数被摊成一片"差不多都不需要"，顶部分辨率不够 —— AUROC 看全局，SRR 只看头部。

两条修法：
1. **正类加权**：把正例的损失权重放大 w 倍，逼模型把头部拉开。代价是概率不再校准，
   所以用训练集留出段做 Platt 重标定（只动 a、b 两个标量，不动排序）。
   w 用 **BIRD 训练集上的 SRR@C=35** 选，dev 不参与。
2. **名次平均（RRF 式，无参数）**：融合分和 LR 概率各给一个名次，取平均名次排序。
   没有权重要挑，所以不存在调参污染 —— 这是能安全报的组合方式。

对照基线：融合分单独（SRR@C=35 = 0.9074，权重也是在 dev 上扫出来的那个 0.5/0.5）。
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
from jevschema.bird_train import load_bird_train  # noqa: E402
from jevschema.metrics import auroc, ece, mean_ci, paired_bootstrap, strict_recall  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import make_stage_tokens  # noqa: E402

from train_lex_classifier import FEATS, features, fit  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


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


def _load(kind, examples, fn=None):
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
                s = fn(e)
                store[e.ex_id] = s
                f.write(json.dumps({"id": e.ex_id,
                                    "s": {"|".join(k): v for k, v in s.items()}},
                                   ensure_ascii=False) + "\n")
    return [store[e.ex_id] for e in examples]


def rank_of(score):
    """分数 -> 名次（0 最好）。名次平均要用这个。"""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    r = np.empty(len(score))
    r[order] = np.arange(len(score))
    return r


def platt(p, y, iters=200, lr=0.5):
    """Platt 重标定：在 logit 空间拟合 a·z + b，只改概率不改排序。"""
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    a, b = 1.0, 0.0
    n = len(y)
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(a * z + b)))
        ga = float(np.sum((q - y) * z)) / n
        gb = float(np.sum(q - y)) / n
        a -= lr * ga
        b -= lr * gb
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-dev", type=int, default=0)
    ap.add_argument("--budgets", default="15,20,25,30,35,40,50")
    ap.add_argument("--closure", default="f1p5")
    ap.add_argument("--pos-weights", default="1,3,6,12,25")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    cfg = parse_closure(a.closure)

    tr, _ = load_bird_train(limit=a.n_train)
    Xs, ys = [], []
    for e in tr:
        _, X, gold = features(e, None)
        Xs.append(X)
        ys.append(np.asarray([1.0 if p in gold else 0.0
                              for p in e.schema.columns()]))
    X = np.vstack(Xs); y = np.concatenate(ys)
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = (X - mu) / sd
    print(f"训练集 {X.shape[0]} 行，正例 {y.mean()*100:.2f}%")

    # ---------------------------------------------------------- 在训练集上选 pos-weight
    print("\n== 正类权重选择（判据：训练集上的 SRR@C=35，dev 不参与）==")
    print(f"{'w':<6}{'train SRR@35':>14}{'train AUROC':>13}{'C̄':>7}")
    best = None
    for w in [float(x) for x in a.pos_weights.split(",")]:
        # fit() 不支持样本权重，用"正例重复 w 倍"等价实现（全批量 Adam，效果一样）
        idx = np.concatenate([np.nonzero(y > 0.5)[0]] * int(round(w)) +
                             [np.nonzero(y <= 0.5)[0]])
        wt, bt, _ = fit(Xn[idx], y[idx], iters=2500, lr=0.05)
        p = 1 / (1 + np.exp(-np.clip(Xn @ wt + bt, -30, 30)))
        auc = auroc(p, y)
        srrs, cb = [], []
        for e in tr[:800]:
            pairs = list(e.schema.columns())
            _, Xe, _ = features(e, None)
            pe = 1 / (1 + np.exp(-np.clip((Xe - mu) / sd @ wt + bt, -30, 30)))
            order = [pairs[j] for j in np.argsort(-pe, kind="stable")]
            kept = closure(e.schema, order, 35, cfg)
            srrs.append(strict_recall(e.gold, kept)); cb.append(len(kept))
        v = float(np.mean(srrs))
        print(f"{w:<6}{v:>14.4f}{auc:>13.4f}{float(np.mean(cb)):>7.1f}")
        if best is None or v > best[0]:
            best = (v, w, wt, bt)
    v_best, w_best, w_vec, b_vec = best
    print(f"  -> 选 w={w_best}（训练集 SRR@35 = {v_best:.4f}）")

    # Platt 重标定（训练集上，不动排序）
    p_raw = 1 / (1 + np.exp(-np.clip(Xn @ w_vec + b_vec, -30, 30)))
    pa, pb = platt(p_raw, y)
    print(f"  Platt: a={pa:.4f} b={pb:.4f}")

    # ---------------------------------------------------------- dev 评估
    schemas = load_bird_schemas()
    dev = load_bird_dev(limit=a.n_dev if a.n_dev > 0 else None, schemas=schemas, shuffle=False)
    bm = _load("bm25", dev, None)
    eb = _load("embedding", dev, None)
    have_emb = any(len(d) > 0 for d in eb)
    print(f"\nBIRD-Dev n={len(dev)}  补全={a.closure}  稠密信号={'有' if have_emb else '无'}")

    rows = []
    for e, b, d in zip(dev, bm, eb):
        pairs, Xd, gold = features(e, None)
        plr = 1 / (1 + np.exp(-np.clip((Xd - mu) / sd @ w_vec + b_vec, -30, 30)))
        plr_cal = 1 / (1 + np.exp(-(pa * np.log(np.clip(plr, 1e-9, 1 - 1e-9) /
                                                (1 - np.clip(plr, 1e-9, 1 - 1e-9))) + pb)))
        bn = _norm({q: float(b.get(q, 0.0)) for q in pairs}, "max")
        dn = (_norm({q: float(d.get(q, 0.0)) for q in pairs}, "minmax")
              if have_emb else {q: 0.0 for q in pairs})
        fused = np.asarray([0.5 * bn[q] + 0.5 * dn[q] for q in pairs])
        rows.append({"e": e, "pairs": pairs, "fused": fused,
                     "lr": plr, "lr_cal": plr_cal,
                     "rankavg": -(rank_of(fused) + rank_of(plr)) / 2.0})

        # 关键分工：LR 最重的特征是 bm25_tbl（**表级**），所以它擅长选表；
        # 融合分在表内有区分度，擅长选列。按"表序用 LR、列序用融合"拼一个排序。
        for key, tbl_sc, col_sc in (("lrtbl_col", plr, fused),
                                    ("fusedtbl_col", fused, fused)):
            by_t = {}
            for i, (t, c) in enumerate(pairs):
                by_t.setdefault(t, []).append(i)
            tscore = {t: float(max(tbl_sc[i] for i in idxs)) for t, idxs in by_t.items()}
            torder = sorted(by_t, key=lambda t: -tscore[t])
            sub = np.zeros(len(pairs))
            r = 0.0
            for t in torder:
                idxs = sorted(by_t[t], key=lambda i: -col_sc[i])
                for i in idxs:
                    sub[i] = -r
                    r += 1.0
            rows[-1][key] = sub

    sigs = ["fused", "lr", "lr_cal", "rankavg", "lrtbl_col", "fusedtbl_col"]
    print(f"\n{'信号':<10}{'AUROC':>8}{'ECE(lr)':>9} " +
          " ".join(f"C{b}" for b in budgets))
    out = {"n": len(dev), "budgets": budgets, "closure": a.closure,
           "pos_weight": w_best, "platt": [round(pa, 4), round(pb, 4)], "sig": {}}
    per = {}
    for name in sigs:
        aucs = []
        per_b = {b: [] for b in budgets}
        kept_b = {b: [] for b in budgets}
        tok_b = {b: [] for b in budgets}
        pl, yl = [], []
        for r in rows:
            e, pairs = r["e"], r["pairs"]
            s = r[name]
            yv = np.asarray([1.0 if q in e.gold.flat_pairs() else 0.0 for q in pairs], int)
            aucs.append(auroc(s, yv))
            order = [pairs[j] for j in np.argsort(-s, kind="stable")]
            for b in budgets:
                kept = closure(e.schema, order, b, cfg)
                kept_b[b].append(len(kept))
                per_b[b].append(strict_recall(e.gold, kept))
                tok_b[b].append(make_stage_tokens(e.schema, e.question, e.evidence,
                                                  kept_columns=kept).gen_prompt_tokens)
            if name == "lr_cal":
                pl.append(r["lr_cal"]); yl.append(yv)
        per[name] = per_b
        rec = {"auroc": round(float(np.nanmean(aucs)), 4)}
        if pl:
            pp = np.concatenate(pl); yy = np.concatenate(yl)
            rec["ece"] = round(float(ece(pp, yy)[0]), 4)
        for b in budgets:
            rec[f"srr@{b}"] = round(float(np.mean(per_b[b])), 4)
            ci = mean_ci(per_b[b])
            rec[f"srr@{b}_ci"] = [round(ci[1], 4), round(ci[2], 4)]
            rec[f"cbar@{b}"] = round(float(np.mean(kept_b[b])), 2)
            rec[f"tok@{b}"] = round(float(np.mean(tok_b[b])), 1)
        out["sig"][name] = rec
        print(f"{name:<10}{rec['auroc']:>8.4f}{rec.get('ece', float('nan')):>9.4f}  " +
              " ".join(f"{rec[f'srr@{b}']:.3f}" for b in budgets))

    for b in budgets:
        print(f"\n== C={b} ==")
        for name in sigs:
            r = out["sig"][name]
            extra = ""
            if name != "fused":
                pb_ = paired_bootstrap(per["fused"][b], per[name][b])["p_boot"]
                extra = f"  Δvs fused={r[f'srr@{b}']-out['sig']['fused'][f'srr@{b}']:+.4f}" \
                        f"  p={pb_:.4f}"
            print(f"  {name:<10} SRR={r[f'srr@{b}']:.4f} "
                  f"CI=[{r[f'srr@{b}_ci'][0]:.4f},{r[f'srr@{b}_ci'][1]:.4f}] "
                  f"C̄={r[f'cbar@{b}']:.1f} tok={r[f'tok@{b}']:.0f}{extra}")

    path = a.out or os.path.join(RESULTS, "rank_policy_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feats": FEATS, "w": [float(x) for x in w_vec], "bias": float(b_vec),
                   "mu": [float(x) for x in mu], "sd": [float(x) for x in sd],
                   "pos_weight": w_best, "platt": [float(pa), float(pb)],
                   "res": out}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
