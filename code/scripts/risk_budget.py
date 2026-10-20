"""风险感知预算分配 v2：融合分排序 + 校准概率定预算。

v1 的教训（同一脚本的上一版）：让逻辑回归的概率**同时**负责排序和定预算是错的 ——
它在 dev 上 AUROC 更高（0.9058 vs 融合的 0.8772），但 top-k 排序反而更差
（C=35 上 SRR 0.8481 vs 0.8814）。AUROC 高不等于 top-k 好：LR 优化的是全局对数似然，
负例占 90%，它把分数摊得很平（"风险很分散"），顶部区分度被牺牲掉了。

所以 v2 把两件事拆开，各用各擅长的：
    排序   用 bm25+embedding 融合分（顶部区分度最好，AUROC 0.8772 / SRR@C=35 0.9074）
    预算   用训练集上拟合的**校准概率** p_i，对"没进 top-m 的那些列"算
              R(m) = 1 - ∏_{i ∉ top-m} (1 - p_i)
           取最小的 m 使 R(m) ≤ ε。ε 是可解释的目标失败率，逐题自适应。
    闭包   FK + 入选表内补齐垫底，兜住"别名断链"。

SRR 是逐题全有或全无，所以 R(m) 恰好就是"这题会挂"的概率 —— 这就是题目里
"risk-aware" 的字面意思，不是一个比喻。

另一个修正：v1 里预测风险比实测失败率高约 2 倍（ε=0.05 -> 预测 0.0415 / 实测 0.0200）。
原因是 R(m) 只统计了 top-m 之外的列，而闭包层随后又补回来一批。v2 直接对
**最终保留集合的补集**算 R，预测和实测才对得上。
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

from train_lex_classifier import features  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")
MODEL = os.path.join(RESULTS, "lex_classifier_bird.json")


def _load(kind, examples, fn):
    p = os.path.join(CACHE, f"bird_{kind}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    if fn is None:      # 只读模式：缓存缺的留空，绝不用别的信号顶替
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


def risk_of_complement(p_all, kept_idx):
    """1 - ∏_{i ∉ kept}(1 - p_i)。用 log 空间算，避免长尾下溢。"""
    mask = np.ones(len(p_all), dtype=bool)
    mask[kept_idx] = False
    q = np.clip(1.0 - p_all[mask], 1e-12, 1.0)
    return float(-np.expm1(np.sum(np.log(q))))


# ---------------------------------------------------------------- 风险校准
def pava(y, w=None):
    """保序回归（PAVA）：返回单调不降的拟合值。用于把"独立性假设下的风险"
    校正成"实测失败率"—— 不需要假设函数形式。"""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    vals, wts = [], []
    for i in range(len(y)):
        vals.append(y[i]); wts.append(w[i])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    out = []
    for v, wt in zip(vals, wts):
        out.extend([v] * int(round(wt)))
    return np.asarray(out[: len(y)])


class RiskCalibrator:
    """把 R（独立性假设）映射到实测失败率。在 BIRD 训练集上拟合，dev 上只应用。"""

    def __init__(self, knots_x, knots_y):
        self.x = np.asarray(knots_x)
        self.y = np.asarray(knots_y)

    def __call__(self, r):
        r = float(np.clip(r, self.x[0], self.x[-1]))
        return float(np.interp(r, self.x, self.y))

    def to_json(self):
        return {"x": [float(v) for v in self.x], "y": [float(v) for v in self.y]}

    @staticmethod
    def from_json(d):
        return RiskCalibrator(d["x"], d["y"])


def fit_risk_calibrator(prep_rows, cfg, m_grid):
    """在给定样本上扫不同的 m，收集 (R_before(m), 闭包后是否真漏了 gold 列)。

    R_before 故意用**闭包前**的 top-m 补集来算：它是 m 的单调函数，
    求解"取最小的 m 使风险 ≤ ε"时只要一次前缀积，不用对每个候选 m 重跑闭包。
    标签 y 则取**闭包后**的真实结果 —— 我们要校的正是"闭包还会救回一批"
    造成的系统性高估。
    """
    xs, ys = [], []
    for r in prep_rows:
        e, pairs, order = r["e"], r["pairs"], r["order"]
        gold = e.gold.flat_pairs()
        for m in m_grid:
            if m > len(pairs):
                continue
            kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
            kidx = np.asarray([j for j, q in enumerate(pairs) if q in set(kept)], dtype=int)
            R = risk_of_complement(r["p"], kidx)
            miss = 1.0 if any(g not in set(kept) for g in gold) else 0.0
            xs.append(R); ys.append(miss)
    xs = np.asarray(xs); ys = np.asarray(ys)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    fit = pava(ys)
    idx = np.unique(np.linspace(0, len(xs) - 1, min(200, len(xs))).astype(int))
    return RiskCalibrator(xs[idx], fit[idx]), (xs, ys, fit)


def _risk_before(p_sorted, m):
    """R(m) = 1 - ∏_{i >= m}(1 - p_i)，p 已按降序排好。"""
    q = np.clip(1.0 - p_sorted, 1e-12, 1.0)
    tail = np.concatenate([np.cumprod(q[::-1])[::-1], [1.0]])
    return float(1.0 - tail[min(m, len(p_sorted))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--eps", default="0.50,0.30,0.20,0.10,0.05,0.02,0.01")
    ap.add_argument("--closure", default="f1p5")
    ap.add_argument("--fixed-ks", default="10,15,20,25,30,35,40,50")
    ap.add_argument("--rank", default="fused", choices=["fused", "lr"])
    ap.add_argument("--max-m", type=int, default=120)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="")
    ap.add_argument("--calib-train", type=int, default=0,
                    help=">0 时在 BIRD 训练集上取 N 例拟合风险校正（保序回归），"
                         "把独立性假设下的 R 映射成实测失败率。dev 不参与拟合。")
    ap.add_argument("--calib-out", default="")
    a = ap.parse_args()
    eps_list = [float(x) for x in a.eps.split(",")]
    ks = [int(x) for x in a.fixed_ks.split(",")]
    cfg = parse_closure(a.closure)

    with open(a.model, "r", encoding="utf-8") as f:
        M = json.load(f)
    w = np.asarray(M["w"]); bias = float(M["bias"])
    mu = np.asarray(M["mu"]); sd = np.asarray(M["sd"])
    # 保存时的 round(...,6) 会把常量特征的 1e-8 抹成 0，这里重新夹一下
    sd = np.where(sd < 1e-6, 1.0, sd)
    print(f"打分器: {os.path.basename(a.model)}  "
          f"(训练 {M['rec']['n_train']} 例, dev AUROC {M['rec']['auroc']}, "
          f"ECE {M['rec']['ece']})  排序={a.rank}  补全={a.closure}")

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(ex)}")
    bm = _load("bm25", ex, bm25_of)
    emb_c = os.path.join(CACHE, "bird_embedding.jsonl")
    eb = _load("embedding", ex, None)   # 只读：缺缓存就退化成纯 bm25，不允许顶替
    have_emb = any(len(d) > 0 for d in eb)

    prep = []
    for e, b, d in zip(ex, bm, eb):
        pairs, X, gold = features(e, None)
        p = 1.0 / (1.0 + np.exp(-np.clip((X - mu) / sd @ w + bias, -30, 30)))
        bn = _norm({q: float(b.get(q, 0.0)) for q in pairs}, "max")
        dn = (_norm({q: float(d.get(q, 0.0)) for q in pairs}, "minmax")
              if have_emb else {q: 0.0 for q in pairs})
        s = np.asarray([0.5 * bn[q] + 0.5 * dn[q] for q in pairs]) if a.rank == "fused" else p
        order = np.argsort(-s, kind="stable")
        prep.append({"e": e, "pairs": pairs, "p": p, "order": order})

    def run(k_of):
        """k_of(prep_row) -> m"""
        srrs, cb, tk, ms, pr = [], [], [], [], []
        for r in prep:
            e, pairs, order = r["e"], r["pairs"], r["order"]
            m = int(k_of(r))
            m = max(1, min(m, min(a.max_m, len(pairs))))
            ms.append(m)
            kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
            kidx = [j for j, q in enumerate(pairs) if q in set(kept)]
            pr.append(risk_of_complement(r["p"], np.asarray(kidx, dtype=int)))
            srrs.append(strict_recall(e.gold, kept))
            cb.append(len(kept))
            tk.append(make_stage_tokens(e.schema, e.question, e.evidence,
                                        kept_columns=kept).gen_prompt_tokens)
        return {"srr": float(np.mean(srrs)), "cbar": float(np.mean(cb)),
                "tok": float(np.mean(tk)), "m_med": float(np.median(ms)),
                "m_p90": float(np.percentile(ms, 90)),
                "pred_risk": float(np.mean(pr)), "_srr": srrs}

    # ---------------------------------------------------------- 风险校准（训练集上拟合）
    calib = None
    if a.calib_train > 0:
        from jevschema.bird_train import load_bird_train
        tr, _ = load_bird_train(limit=a.calib_train)
        trp = []
        for e in tr:
            b = bm25_of(e)
            pairs, X, _gold = features(e, None)
            p = 1.0 / (1.0 + np.exp(-np.clip((X - mu) / sd @ w + bias, -30, 30)))
            bn = _norm({q: float(b.get(q, 0.0)) for q in pairs}, "max")
            s = np.asarray([bn[q] for q in pairs])   # 训练集上不加载稠密编码器，退化成 bm25 排序
            trp.append({"e": e, "pairs": pairs, "p": p,
                        "order": np.argsort(-s, kind="stable")})
        m_grid = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60]
        calib, diag = fit_risk_calibrator(trp, cfg, m_grid)
        xs, ys, fit = diag
        print(f"\n风险校准（BIRD 训练集 {len(tr)} 例，保序回归，dev 不参与）:")
        print(f"  采样点 {len(xs)} 个（R 从 {xs[0]:.4f} 到 {xs[-1]:.4f}）")
        for lo, hi in [(0.0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 1.01)]:
            sel = (xs >= lo) & (xs < hi)
            if sel.sum():
                print(f"  R∈[{lo:.2f},{hi:.2f})  n={int(sel.sum()):<6} "
                      f"预测均值={xs[sel].mean():.4f}  实测失败率={ys[sel].mean():.4f}  "
                      f"校正后={fit[sel].mean():.4f}")
        if a.calib_out:
            with open(a.calib_out, "w", encoding="utf-8") as f:
                json.dump(calib.to_json(), f, indent=2)
            print(f"  校正表写入: {a.calib_out}")

    print(f"\n== 固定 k（对照）==\n{'k':<6}{'SRR':>8}{'C̄':>8}{'tokens':>9}{'失败率':>9}")
    fixed = {k: run(lambda r, k=k: k) for k in ks}
    for k, d in fixed.items():
        ci = mean_ci(d["_srr"])
        print(f"{k:<6}{d['srr']:>8.4f}{d['cbar']:>8.1f}{d['tok']:>9.0f}"
              f"{1-d['srr']:>9.4f}   CI=[{ci[1]:.4f},{ci[2]:.4f}]")

    print(f"\n== 风险自适应（逐题解 R(m) ≤ ε）==\n"
          f"{'ε':<7}{'SRR':>8}{'C̄':>8}{'tokens':>9}{'预测风险':>10}{'实测失败':>10}"
          f"{'m中位':>7}{'m p90':>7}")

    def k_of_eps(eps):
        """二分找最小的 m，使**闭包后**补集的（校准）风险 ≤ ε。

        为什么必须看闭包后：闭包会额外救回一批列（实测能砍掉约 6 倍风险），
        只看 top-m 补集会把预算放大到几乎全留。闭包后的 R 随 m 单调不增，可以二分。
        """
        def f(r):
            e, pairs, order, p = r["e"], r["pairs"], r["order"], r["p"]
            lo, hi = 1, len(pairs)
            while lo < hi:
                mid = (lo + hi) // 2
                kept = closure(e.schema, [pairs[j] for j in order], mid, cfg)
                kidx = np.asarray([j for j, q in enumerate(pairs) if q in set(kept)],
                                  dtype=int)
                R = risk_of_complement(p, kidx)
                val = calib(R) if calib is not None else R
                if val <= eps:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        return f

    out = {"n": len(ex), "closure": a.closure, "rank": a.rank,
           "fixed": {str(k): {kk: vv for kk, vv in d.items() if kk != "_srr"} for k, d in fixed.items()},
           "risk": {}}
    for eps in eps_list:
        d = run(k_of_eps(eps))
        ci = mean_ci(d["_srr"])
        rec = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
               for kk, vv in d.items() if kk != "_srr"}
        rec["eps"] = eps
        rec["ci"] = [round(ci[1], 4), round(ci[2], 4)]
        out["risk"][str(eps)] = rec
        print(f"{eps:<7}{d['srr']:>8.4f}{d['cbar']:>8.1f}{d['tok']:>9.0f}"
              f"{d['pred_risk']:>10.4f}{1-d['srr']:>10.4f}{d['m_med']:>7.0f}"
              f"{d['m_p90']:>7.0f}   CI=[{ci[1]:.4f},{ci[2]:.4f}]")

    print("\n== 等 C̄ 对照（自适应 vs 最接近的固定 k）==")
    for eps in eps_list:
        r = out["risk"][str(eps)]
        kbest = min(fixed, key=lambda k: abs(fixed[k]["cbar"] - r["cbar"]))
        d = fixed[kbest]
        pb = paired_bootstrap(d["_srr"], [0]) if False else None
        print(f"  ε={eps:<6} C̄={r['cbar']:5.1f} SRR={r['srr']:.4f}   "
              f"vs k={kbest:<3} C̄={d['cbar']:5.1f} SRR={d['srr']:.4f}   "
              f"Δ={r['srr']-d['srr']:+.4f}")

    print("\n== 风险估计对账（闭包后补集的预测风险 vs 实测失败率）==")
    for eps in eps_list:
        r = out["risk"][str(eps)]
        print(f"  ε={eps:<6} 预测={r['pred_risk']:.4f}  实测={1-r['srr']:.4f}  "
              f"偏差={1-r['srr']-r['pred_risk']:+.4f}")

    path = a.out or os.path.join(RESULTS, "risk_budget_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
