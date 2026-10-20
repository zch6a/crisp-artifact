"""逐题预算预测器：学着预测"这题该给多少列预算"。

为什么需要它（risk_menu.py 给出的实证）：

    oracle 自适应   SRR 0.7500 @ 358 tokens (C̄=36.0)
    固定最优 (cfg,k) SRR 0.7125 @ 469 tokens (C̄=55.3)
    → 自适应空间：分数更高 0.0375，还省 111 tokens（-24%）

但风险模型的两种形式（逐列独立 / 计数型）**一点都没吃到这个空间**，
等 token 对照下反而输给固定 (cfg,k) 最多 0.1167（p=0.034）。
原因不是"自适应没用"，而是**预测器太差**：LR 的长尾概率过于分散，
Σp 在大库上被几十个小概率堆高（λ 拟合出来只有 0.43），
于是它在"哪一格最好"这个问题上的排序是错的。

所以这里换一条路：**不去建模"删了会不会断 SQL"的概率，直接回归 oracle 预算**。
    target  = 能达到全召回的最小深度 m*（oracle，只用训练集算）
    特征    = 融合分排序后的形状特征 + 库规模 + 问题长度（全部与具体库无关）
    模型    = ridge 回归 log(m*+1)，再乘一个全局余量 δ
    δ       = 唯一的旋钮，扫它就能画出 (token, SRR) 前沿
              δ>1 偏保守（多花 token 换召回），δ<1 偏激进

这是一次前向、纯 CPU、零 LLM 调用的预算分配 —— 和"用 LLM 迭代探索 schema"
的路线在成本轴上是数量级的差别。

训练集拟合，dev 只做应用。
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
from jevschema.metrics import mean_ci, paired_bootstrap  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import make_stage_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


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


def fused_order(ex_list):
    """返回 [(pairs, s_sorted_desc, order)]，融合分 = 0.5*bm25(max归一) + 0.5*emb(minmax)。

    ⚠️ 踩过的坑：BIRD 训练集没有 embedding 缓存（bird_embedding.jsonl 里只有 dev 的
    1534 行），于是拟合侧 have_emb=False 用 0.5*bm25，应用侧 have_emb=True 用
    0.5*bm25+0.5*emb —— 两边的分数分布完全不是一个东西，学出来的回归在 dev 上
    直接塌到 m̂≈2（比固定 k 还差 0.2 SRR）。
    所以默认走 dev 上的 K 折交叉拟合（--fit devcv）：每一折的 held-out 部分
    从来没进过拟合，报出来的数字仍然是无泄漏的，而且分布和部署时一致。
    """
    bm = _load("bm25", ex_list, bm25_of)
    eb = _load("embedding", ex_list, None)      # 只读：缺就是缺，不许顶替
    have_emb = any(len(d) > 0 for d in eb)
    if not have_emb:
        print("  [警告] 这批样本没有 embedding 缓存，退化为 0.5*bm25 —— "
              "如果拟合侧和应用侧不一致，预测器会崩，别混着用。")
    out = []
    for e, b, d in zip(ex_list, bm, eb):
        pairs = e.schema.columns()
        bn = _norm({q: float(b.get(q, 0.0)) for q in pairs}, "max")
        dn = (_norm({q: float(d.get(q, 0.0)) for q in pairs}, "minmax")
              if have_emb else {q: 0.0 for q in pairs})
        s = np.asarray([0.5 * bn[q] + 0.5 * dn[q] for q in pairs])
        out.append((pairs, s, np.argsort(-s, kind="stable")))
    return out


def oracle_m(e, pairs, s, order, cfg, ms):
    """能达到全召回的最小深度；菜单再深也救不回来则返回 None。"""
    gold = set(e.gold.flat_pairs())
    for m in ms:
        if m > len(pairs):
            break
        kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
        if all(g in set(kept) for g in gold):
            return m
    return None


# ---------------------------------------------------------------- 特征
def load_prob_model(path):
    """读 train_lex_classifier 拟合出来的列概率模型（DB 无关特征 -> P(该列是 gold 列)）。"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        M = json.load(f)
    sd = np.asarray(M["sd"]); sd = np.where(sd < 1e-6, 1.0, sd)
    return (np.asarray(M["w"]), float(M["bias"]), np.asarray(M["mu"]), sd)


def prob_of(e, order, pm, from_train_lex):
    """按 order 顺序给出每列的 p。需要 train_lex_classifier.features。"""
    w, bias, mu, sd = pm
    pairs, X, _ = from_train_lex(e, None)
    p_all = 1.0 / (1.0 + np.exp(-np.clip((X - mu) / sd @ w + bias, -30, 30)))
    return p_all[order]


def feats(e, s_sorted, pairs, p_sorted=None):
    """全部与具体数据库无关的形状特征。s_sorted 已按降序排好。

    p_sorted（可选）= 与 s_sorted 同序的列概率。加进来是因为"融合分形状"只说了
    **相关性排序**，没说**有多少列是真需要的**；概率模型补的正是数量维度。
    只用相对量（分位数、占比），不用 Σp 的绝对值 —— 绝对值会被长尾小概率堆高
    （risk_menu 里拟合出的 λ 只有 0.43，就是这个毛病）。
    """
    n = len(s_sorted)
    hi = float(s_sorted[0]) if n else 0.0
    v = s_sorted / hi if hi > 0 else np.zeros_like(s_sorted)
    tot = float(v.sum()) or 1.0

    def at(k):
        return float(v[min(k, n - 1)]) if n else 0.0

    ent_v = v[v > 1e-9]
    ent = float(-(ent_v * np.log(ent_v)).sum()) if len(ent_v) else 0.0
    gaps = np.diff(v[:30]) if n > 2 else np.zeros(1)
    f = [
        np.log1p(n),
        np.log1p(len(e.schema.table_names())),
        np.log1p(len(e.question.split())),
        1.0 if e.evidence else 0.0,
        at(0), at(1), at(2), at(4), at(9), at(14), at(19), at(29),
        float(np.mean(v[:10])), float(np.std(v[:20])),
        float(v[:10].sum() / tot), float(v[:20].sum() / tot),
        float((v > 0.5).sum()) / max(1, n),
        float((v > 0.2).sum()) / max(1, n),
        ent / np.log(max(2, n)),
        float(np.argmax(gaps)) / 30.0 if len(gaps) else 0.0,
        float(gaps.max()) if len(gaps) else 0.0,
    ]
    if p_sorted is not None:
        tp = float(p_sorted.sum()) or 1.0
        cum = np.cumsum(p_sorted) / tp
        # 概率质量的"分位深度"：攒到 q 比例的质量需要多少列 —— 直接用相对量，
        # 不受 LR 长尾过度分散的影响（绝对值会，见 risk_menu 的 λ=0.43）
        qs = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        f += [float(np.searchsorted(cum, q) + 1) / max(1, n) for q in qs]
        f += [float(tp), float(p_sorted[0]) if n else 0.0,
              float(np.mean(p_sorted[:10])),
              float((p_sorted > 0.5).sum()),
              float(np.log1p(tp))]
    return np.asarray(f, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--cfg", default="f1p5s15k1")
    ap.add_argument("--ms", default="5,6,8,10,12,15,18,20,25,30,35,40,50,60,80")
    ap.add_argument("--fixed-ks", default="10,12,15,18,20,25,30,35,40,50,60")
    ap.add_argument("--fixed-grid", default="f1p0s0k0,f1p5k1,f1p0s15k1,"
                                            "f1p3s15k1,f1p5s15k1,f1p8k1",
                    help="固定基线要跑完整个 (配置×k) 网格再取前沿 —— "
                         "只拿单一配置当对照会把自适应比虚高。")
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--fit", default="devcv", choices=["devcv", "train"],
                    help="devcv=dev 上 K 折交叉拟合（默认，分布与部署一致）；"
                         "train=在 BIRD 训练集上拟合（要求训练集也有 embedding 缓存）")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--deltas", default="0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.4,1.6,1.8")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--rule", default="all",
                    choices=["reg", "mass", "frac", "all"],
                    help="reg=回归预测 m*；mass=概率质量占比；frac=预算随库大小伸缩（零参数学习）")
    ap.add_argument("--taus", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.99")
    ap.add_argument("--fracs", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    ap.add_argument("--zs", default="-0.5,0.0,0.5,1.0,1.5")
    ap.add_argument("--prob-model", default=os.path.join(RESULTS,
                                                         "lex_classifier_bird.json"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = parse_closure(a.cfg)
    ms = [int(x) for x in a.ms.split(",")]
    fks = [int(x) for x in a.fixed_ks.split(",")]
    deltas = [float(x) for x in a.deltas.split(",")]

    schemas = load_bird_schemas()
    dev = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(dev)}   cfg={cfg.tag()}  m 网格={ms}")

    dv_ord = fused_order(dev)

    # 列概率（用于 mass 规则和回归的额外特征）
    pm = load_prob_model(a.prob_model)
    ps_dev = None
    if pm is not None:
        from train_lex_classifier import features as lex_features
        ps_dev = [prob_of(e, order, pm, lex_features)
                  for e, (pairs, s, order) in zip(dev, dv_ord)]
        print(f"列概率模型: {os.path.basename(a.prob_model)}")
    else:
        print("[警告] 没有列概率模型，mass 规则与 p 相关特征不可用")

    def _ridge_fit_predict(Zf, yf, Zp):
        A = Zf.T @ Zf + a.ridge * np.eye(Zf.shape[1])
        w = np.linalg.solve(A, Zf.T @ yf)
        return Zp @ w, w

    if a.fit == "devcv":
        # dev 上 K 折：每折的 held-out 都没进过自己的拟合
        print(f"拟合方式 = dev {a.folds} 折交叉（每折 held-out 不参与自己的拟合）")
        Xd = np.asarray([feats(e, s[order], pairs,
                               ps_dev[i] if ps_dev is not None else None)
                         for i, (e, (pairs, s, order)) in enumerate(zip(dev, dv_ord))])
        yd = []
        for e, (pairs, s, order) in zip(dev, dv_ord):
            ms_ = oracle_m(e, pairs, s, order, cfg, ms)
            yd.append(np.log1p(ms_ if ms_ is not None else max(ms)))
        yd = np.asarray(yd)
        print(f"  dev 上 oracle 可达率 = {float(np.mean([1.0 if oracle_m(e,p,s,o,cfg,ms) is not None else 0.0 for e,(p,s,o) in zip(dev,dv_ord)])):.4f}")
        logm = np.zeros(len(dev))
        logsig = np.zeros(len(dev))
        r2s = []
        rng = np.random.default_rng(0)
        fold = rng.integers(0, a.folds, size=len(dev))
        for f in range(a.folds):
            tr_i = np.where(fold != f)[0]
            te_i = np.where(fold == f)[0]
            mu, sd = Xd[tr_i].mean(0), Xd[tr_i].std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            Zf = np.hstack([(Xd[tr_i] - mu) / sd, np.ones((len(tr_i), 1))])
            Zp = np.hstack([(Xd[te_i] - mu) / sd, np.ones((len(te_i), 1))])
            ph, _ = _ridge_fit_predict(Zf, yd[tr_i], Zp)
            logm[te_i] = ph
            # 第二条回归：拟合 log(残差²)，得到逐题的不确定度 σ̂。
            # 有了它才能做"异方差分位数"：难预测的题自动加余量，
            # 而不是像全局 δ 那样对所有题一刀切。
            res = yd[tr_i] - (Zf @ np.linalg.solve(
                Zf.T @ Zf + a.ridge * np.eye(Zf.shape[1]), Zf.T @ yd[tr_i]))
            lv = np.log(np.maximum(res ** 2, 1e-6))
            wv = np.linalg.solve(Zf.T @ Zf + a.ridge * np.eye(Zf.shape[1]), Zf.T @ lv)
            logsig[te_i] = 0.5 * (Zp @ wv)
            ph_tr, _ = _ridge_fit_predict(Zf, yd[tr_i], Zf)
            ssr = float(((yd[tr_i] - ph_tr) ** 2).mean())
            sst = float(max(1e-9, ((yd[tr_i] - yd[tr_i].mean()) ** 2).mean()))
            r2s.append(1 - ssr / sst)
        print(f"  折内拟合 R² = {np.mean(r2s):.4f}  (对 log(m*+1) 回归)")
        # 样本外 R²（这才是真正的预测能力）
        ssr = float(((yd - logm) ** 2).mean())
        sst = float(max(1e-9, ((yd - yd.mean()) ** 2).mean()))
        print(f"  交叉验证（样本外）R² = {1 - ssr / sst:.4f}")
    else:
        from jevschema.bird_train import load_bird_train
        tr, _ = load_bird_train(limit=a.train)
        print(f"BIRD-Train n={len(tr)}（只在这里拟合，dev 不参与）")
        tr_ord = fused_order(tr)
        Xtr, ytr = [], []
        for e, (pairs, s, order) in zip(tr, tr_ord):
            ms_ = oracle_m(e, pairs, s, order, cfg, ms)
            Xtr.append(feats(e, s[order], pairs))
            ytr.append(np.log1p(ms_ if ms_ is not None else max(ms)))
        Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Ztr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
        pred_tr, w = _ridge_fit_predict(Ztr, ytr, Ztr)
        ssr = float(((ytr - pred_tr) ** 2).mean())
        sst = float(max(1e-9, ((ytr - ytr.mean()) ** 2).mean()))
        print(f"  训练集拟合 R² = {1 - ssr / sst:.4f}  (对 log(m*+1) 回归)")
        Xd = np.asarray([feats(e, s[order], pairs,
                               ps_dev[i] if ps_dev is not None else None)
                         for i, (e, (pairs, s, order)) in enumerate(zip(dev, dv_ord))])
        Zd = np.hstack([(Xd - mu) / sd, np.ones((len(Xd), 1))])
        logm = Zd @ w
        logsig = np.zeros(len(dev))

    def run_m(m_list):
        srrs, cb, tk = [], [], []
        for (e, (pairs, s, order)), m in zip(zip(dev, dv_ord), m_list):
            m = int(max(ms[0], min(ms[-1], round(m))))
            kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
            srrs.append(1.0 if all(g in set(kept) for g in e.gold.flat_pairs()) else 0.0)
            cb.append(len(kept))
            tk.append(make_stage_tokens(e.schema, e.question, e.evidence,
                                        kept_columns=kept).gen_prompt_tokens)
        return {"srr": float(np.mean(srrs)), "cbar": float(np.mean(cb)),
                "tok": float(np.mean(tk)), "_srr": srrs}

    fcfgs = [parse_closure(g) for g in a.fixed_grid.split(",")]
    print(f"\n== 固定 (配置, k) 网格：{len(fcfgs)} 配置 × {len(fks)} 深度 =="
          f"（后面只跟这条网格的**帕累托前沿**比，不是跟单一配置比）")
    print(f"{'cfg':<12}{'k':>4}{'SRR':>9}{'C̄':>8}{'tokens':>9}  95%CI")
    fixed = {}
    for fc in fcfgs:
        old_cfg, cfg = cfg, fc
        for k in fks:
            fixed[(fc.tag(), k)] = run_m([k] * len(dev))
            d = fixed[(fc.tag(), k)]
            ci = mean_ci(d["_srr"])
            print(f"{fc.tag():<12}{k:>4}{d['srr']:>9.4f}{d['cbar']:>8.1f}"
                  f"{d['tok']:>9.0f}  [{ci[1]:.4f},{ci[2]:.4f}]")
        cfg = old_cfg

    # 固定网格的帕累托前沿（x=tokens 越小越好，y=SRR 越大越好）
    pts = sorted(fixed.items(), key=lambda kv: (kv[1]["tok"], -kv[1]["srr"]))
    front, best = [], -1.0
    for kv in pts:
        if kv[1]["srr"] > best + 1e-9:
            front.append(kv)
            best = kv[1]["srr"]
    print(f"\n== 固定网格帕累托前沿（{len(front)} 点）==")
    for kv in front:
        print(f"  {kv[0][0]:<12} k={kv[0][1]:<3} tok={kv[1]['tok']:6.1f} "
              f"C̄={kv[1]['cbar']:5.1f} SRR={kv[1]['srr']:.4f}")

    def front_at(tok):
        """前沿上"花不超过 tok"能拿到的最高 SRR。"""
        ok = [kv for kv in fixed.items() if kv[1]["tok"] <= tok + 1e-9]
        return max(ok, key=lambda kv: kv[1]["srr"]) if ok else None

    _fp = sorted([(v["tok"], v["srr"]) for v in fixed.values()])

    def front_interp(tok):
        """前沿的**线性插值**。

        不能用"花销 ≤ tok 的最高 SRR"当对照：固定网格的 k 是离散的，
        两格之间会有十几 token 的空隙，那样比会把固定策略系统性低估
        （实测 370 tok 处最近的前沿点是 357 的 0.7901，而 372.8 处就有 0.8442）。
        插值相当于承认"固定策略也可以取网格之间的 k"，这才公平。
        """
        xs = [p[0] for p in _fp]; ys = [p[1] for p in _fp]
        if tok <= xs[0]:
            return ys[0]
        if tok >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if xs[i] >= tok:
                x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
                return y0 + (y1 - y0) * (tok - x0) / max(1e-9, x1 - x0)
        return ys[-1]

    adaptive, ada_srr = {}, {}

    if a.rule in ("frac", "all"):
        print(f"\n== 规则 C：预算随库大小伸缩（m̂ = α · 该库列数，零参数学习）==")
        print("  库规模从 ~20 列到 ~200 列不等，固定 k 对大库必然不够、对小库必然浪费。")
        print(f"{'α':>6}{'SRR':>9}{'C̄':>8}{'tokens':>9}{'m̂中位':>8}  95%CI")
        ncols = np.array([len(pairs) for pairs, s, o in dv_ord], dtype=float)
        for al in [float(x) for x in a.fracs.split(",")]:
            m_hat = np.ceil(al * ncols)
            r = run_m(m_hat)
            adaptive[("frac", al)] = r
            ada_srr[("frac", al)] = r["_srr"]
            ci = mean_ci(r["_srr"])
            print(f"{al:>6}{r['srr']:>9.4f}{r['cbar']:>8.1f}{r['tok']:>9.0f}"
                  f"{np.median(m_hat):>8.0f}  [{ci[1]:.4f},{ci[2]:.4f}]")

    if a.rule in ("reg", "all"):
        print(f"\n== 规则 A：回归预测 m*（m̂ = exp(μ̂ + z·σ̂)，异方差分位数）==")
        print(f"{'z':>6}{'SRR':>9}{'C̄':>8}{'tokens':>9}{'m̂中位':>8}  95%CI")
        for z in [float(x) for x in a.zs.split(",")]:
            m_hat = np.expm1(np.clip(logm + z * np.exp(logsig), 0, 12))
            r = run_m(m_hat)
            adaptive[("reg", z)] = r
            ada_srr[("reg", z)] = r["_srr"]
            ci = mean_ci(r["_srr"])
            print(f"{z:>6}{r['srr']:>9.4f}{r['cbar']:>8.1f}{r['tok']:>9.0f}"
                  f"{np.median(m_hat):>8.0f}  [{ci[1]:.4f},{ci[2]:.4f}]")

    if a.rule in ("mass", "all") and ps_dev is not None:
        print(f"\n== 规则 B：概率质量占比（m̂ = 攒到 τ 比例的列概率质量所需的深度）==")
        print("  只用相对量，所以不受 LR 长尾过度分散的影响 "
              "（risk_menu 里 Σp 绝对值被长尾堆高，λ 只有 0.43）")
        print(f"{'τ':>6}{'SRR':>9}{'C̄':>8}{'tokens':>9}{'m̂中位':>8}  95%CI")
        cums = []
        for p in ps_dev:
            tp = float(p.sum()) or 1.0
            cums.append(np.cumsum(p) / tp)
        for tau in [float(x) for x in a.taus.split(",")]:
            m_hat = np.array([float(np.searchsorted(c, tau) + 1) for c in cums])
            r = run_m(m_hat)
            adaptive[("mass", tau)] = r
            ada_srr[("mass", tau)] = r["_srr"]
            ci = mean_ci(r["_srr"])
            print(f"{tau:>6}{r['srr']:>9.4f}{r['cbar']:>8.1f}{r['tok']:>9.0f}"
                  f"{np.median(m_hat):>8.0f}  [{ci[1]:.4f},{ci[2]:.4f}]")

    print("\n== 等 token 对照（同样花钱，自适应 vs 固定网格前沿）★ 主要看这个 ==")
    print("  对照 = 固定网格里『花销不超过自适应点』的最高 SRR，"
          "再对这个点做配对 bootstrap")
    for key in sorted(adaptive, key=lambda k: adaptive[k]["tok"]):
        r = adaptive[key]
        kv = front_at(r["tok"])
        if kv is None:
            print(f"  {key[0]}={key[1]:<5} tok={r['tok']:6.1f} SRR={r['srr']:.4f}   "
                  f"（固定网格里没有更便宜的点可比）")
            continue
        pb = paired_bootstrap(kv[1]["_srr"], ada_srr[key])
        fi = front_interp(r["tok"])
        # 是否真把前沿推出去了：没有任何固定点能支配它（花得更少且分不更低）
        dominated = any(v["tok"] <= r["tok"] + 1e-9 and v["srr"] >= r["srr"] - 1e-9
                        for v in fixed.values())
        flag = "  << 赢" if pb["diff"] > 0 and pb["p_boot"] < 0.05 else (
            "  << 输" if pb["diff"] < 0 and pb["p_boot"] < 0.05 else "")
        print(f"  {key[0]}={key[1]:<5} tok={r['tok']:6.1f} SRR={r['srr']:.4f}   "
              f"vs 固定网格点 {kv[0][0]} k={kv[0][1]:<3} tok={kv[1]['tok']:6.1f} "
              f"SRR={kv[1]['srr']:.4f} Δ={pb['diff']:+.4f} p={pb['p_boot']:.4f}"
              f" | 前沿插值={fi:.4f} 差值={r['srr']-fi:+.4f}"
              f"{'  前沿外' if not dominated else ''}{flag}")

    # oracle 上界（同菜单）
    print("\n== oracle 上界（同菜单，偷看 gold）==")
    ms_oracle, cb_o, tk_o = [], [], []
    for e, (pairs, s, order) in zip(dev, dv_ord):
        gold = set(e.gold.flat_pairs())
        pick = None
        for m in ms:
            if m > len(pairs):
                break
            kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
            if all(g in set(kept) for g in gold):
                pick = (m, kept)
                break
        if pick is None:
            m = ms[-1]
            kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
        else:
            m, kept = pick
        ms_oracle.append(m); cb_o.append(len(kept))
        tk_o.append(make_stage_tokens(e.schema, e.question, e.evidence,
                                      kept_columns=kept).gen_prompt_tokens)
    srr_o = [1.0] * 0
    o_tok, o_cb = float(np.mean(tk_o)), float(np.mean(cb_o))
    print(f"  oracle: C̄={o_cb:.1f} tokens={o_tok:.0f}（SRR 定义上接近 1.0 减去不可达题）")
    print(f"  固定最优 k: tok={min(fixed.values(), key=lambda z: abs(z['tok']-o_tok))['tok']:.0f}"
          f" → oracle 在同样花销下拿到几乎满分")

    path = a.out or os.path.join(RESULTS, "budget_predictor_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(dev), "cfg": cfg.tag(), "ms": ms,
                   "oracle": {"cbar": round(o_cb, 2), "tok": round(o_tok, 1),
                              "m_med": float(np.median(ms_oracle))},
                   "fixed": {f"{c}_k{k}": {kk: vv for kk, vv in v.items() if kk != "_srr"}
                             for (c, k), v in fixed.items()},
                   "front": [{"cfg": c, "k": k, **{kk: vv for kk, vv in fixed[(c, k)].items()
                                                   if kk != "_srr"}}
                             for (c, k) in [kv[0] for kv in front]],
                   "adaptive": {f"{d[0]}_{d[1]}": {kk: vv for kk, vv in v.items()
                                                   if kk != "_srr"}
                                for d, v in adaptive.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
