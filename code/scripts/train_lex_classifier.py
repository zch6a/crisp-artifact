"""学习式列打分器：在 BIRD 训练集上拟合，在 BIRD-Dev 上评估。

为什么必须做这一步（方法论问题，不是刷分问题）：

前面所有数字里的**权重都是人挑的** —— bm25/embedding 融合 0.5/0.5、补齐 pad=5/8/12，
全都是在 BIRD-Dev 上扫出来的。Dev 是我们要报数的测试集，在上面挑超参属于测试集调参，
审稿人一眼就会挑出来。正确的做法是：**在 BIRD 训练集（arcwise/bird_sql_annotated，
6048 例 / 989 个库，与 dev 零库重叠）上把权重学出来，dev 只做一次评估。**

做法：每列抽 15 个与具体库无关的特征（词法分、稠密相似度、表级先验、结构标志、
名字形态），在训练集上拟合一个带 L2 的逻辑回归，输出"这一列被 gold SQL 用到"的
**校准概率**。它同时解决三件事：
    1. 融合权重变成学出来的，不再是手挑的；
    2. 概率是真概率 → 可以做阈值路由、可以算 ECE/Brier（Pilot 门限要的）；
    3. 成本仍然是 $0（CPU 打分，不调 LLM）。

对照：手挑的 bm25+emb 0.5/0.5（BIRD-Dev AUROC 0.8772 / SRR@C=35 0.9074）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import jevschema.paths as _P  # noqa: E402  (env CRISP_DATA / CRISP_MODELS)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.bird_train import load_bird_train  # noqa: E402
from jevschema.metrics import auroc, brier, ece, mean_ci, paired_bootstrap, strict_recall  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")

FEATS = [
    "bm25_col", "bm25_rel", "bm25_hit", "bm25_rank",
    "emb_col", "emb_rel", "emb_rank",
    "bm25_tbl", "tbl_rank",
    "is_fk", "is_pk", "is_num", "has_digit",
    "n_tok", "tbl_size", "pos_in_tbl",
]


# ---------------------------------------------------------------- 特征
def _tok(s):
    out = []
    for raw in re.findall(r"[a-z0-9_]+", s.lower()):
        out.extend(p for p in raw.split("_") if p)
    return out


def _emb_sims(ex, model):
    pairs = ex.schema.columns()
    docs = [f"{t.replace('_',' ')} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    e = model.encode([q] + docs, normalize_embeddings=True, show_progress_bar=False,
                     batch_size=256)
    sims = e[1:] @ e[0]
    return dict(zip(pairs, (float(x) for x in sims)))


def features(ex, emb=None):
    """返回 (pairs, X, gold_set)。X 的每一行是一列的特征向量。"""
    pairs = list(ex.schema.columns())
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    qt = set(_tok(q))
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    b = np.asarray(bm25_scores(q, docs), dtype=float)
    bmax = b.max() if b.max() > 0 else 1.0
    bn = b / bmax

    tnames = [t.name for t in ex.schema.tables]
    tdocs = [f"{t.name} " + " ".join(c.name for c in t.columns) for t in ex.schema.tables]
    tb = np.asarray(bm25_scores(q, tdocs), dtype=float)
    tbmax = tb.max() if tb.max() > 0 else 1.0
    tbn = {t: v / tbmax for t, v in zip(tnames, tb)}
    torder = {t: i for i, t in enumerate(np.argsort(-tb))}

    if emb is None:
        e = np.zeros(len(pairs))
    else:
        es = _emb_sims(ex, emb)
        e = np.asarray([float(es.get(p, 0.0)) for p in pairs])
    elo, ehi = e.min(), e.max()
    en = (e - elo) / max(1e-9, ehi - elo)

    fk = set()
    for f in ex.schema.fks:
        fk.add((f.src_table, f.src_col))
        fk.add((f.dst_table, f.dst_col))
    pk = {(t.name, c.name) for t in ex.schema.tables for c in t.columns if c.is_pk}
    tmap = {t.name: {c.name: c for c in t.columns} for t in ex.schema.tables}
    is_num = np.array([
        1.0 if re.search(r"int|real|doub|floa|dec|num",
                         tmap.get(t, {}).get(c).dtype if tmap.get(t, {}).get(c) else "text",
                         re.I) else 0.0
        for t, c in pairs])

    # 每列在所属表内的位次 / 表内相对分
    by_t = {}
    for i, (t, c) in enumerate(pairs):
        by_t.setdefault(t, []).append(i)
    tbl_size = np.zeros(len(pairs))
    pos_in = np.zeros(len(pairs))
    brel = np.zeros(len(pairs))
    erel = np.zeros(len(pairs))
    for t, idxs in by_t.items():
        sub_b = b[idxs]
        sub_e = e[idxs]
        mb = sub_b.max() if sub_b.max() > 0 else 1.0
        me = sub_e.max() if sub_e.max() > 1e-9 else 1.0
        for r, i in enumerate(idxs):
            tbl_size[i] = len(idxs)
            pos_in[i] = r / max(1, len(idxs) - 1)
            brel[i] = sub_b[r] / mb
            erel[i] = sub_e[r] / me

    bm_rank = np.empty(len(pairs))
    bm_rank[np.argsort(-b, kind="stable")] = np.arange(len(pairs)) / max(1, len(pairs) - 1)
    em_rank = np.empty(len(pairs))
    em_rank[np.argsort(-e, kind="stable")] = np.arange(len(pairs)) / max(1, len(pairs) - 1)

    X = np.column_stack([
        bn, brel, (b > 0).astype(float), bm_rank,
        en, erel, em_rank,
        np.array([tbn.get(t, 0.0) for t, _ in pairs]),
        np.array([torder.get(t, 0) / max(1, len(tnames) - 1) for t, _ in pairs]),
        np.array([1.0 if p in fk else 0.0 for p in pairs]),
        np.array([1.0 if p in pk else 0.0 for p in pairs]),
        is_num,
        np.array([1.0 if re.search(r"\d", c) else 0.0 for _, c in pairs]),
        np.array([min(len(_tok(c)), 12) / 12.0 for _, c in pairs]),
        np.minimum(tbl_size, 200.0) / 200.0,
        pos_in,
    ])
    return pairs, X, ex.gold.flat_pairs()


def fit(X, y, l2=1e-4, iters=3000, lr=0.05, seed=0):
    """全批量 Adam 逻辑回归（125k×15 的规模，全批量几秒就收敛）。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    m_w = np.zeros(d)
    v_w = np.zeros(d)
    m_b = v_b = 0.0
    pos = float(y.mean())
    for it in range(1, iters + 1):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (p - y) / n
        g[-1] += 0.0
        gb = float((p - y).mean())
        gw = g + l2 * w
        m_w = 0.9 * m_w + 0.1 * gw
        v_w = 0.999 * v_w + 0.001 * gw * gw
        m_b = 0.9 * m_b + 0.1 * gb
        v_b = 0.999 * v_b + 0.001 * gb * gb
        w -= lr * (m_w / (1 - 0.9 ** it)) / (np.sqrt(v_w / (1 - 0.999 ** it)) + 1e-8)
        b -= lr * (m_b / (1 - 0.9 ** it)) / (np.sqrt(v_b / (1 - 0.999 ** it)) + 1e-8)
    return w, b, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-dev", type=int, default=0)
    ap.add_argument("--budgets", default="10,15,20,25,30,35,40,50")
    ap.add_argument("--closure", default="f1")
    ap.add_argument("--use-emb", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    cfg = parse_closure(a.closure)

    # ---------------------------------------------------------- 训练
    t0 = time.time()
    tr, st = load_bird_train(limit=a.n_train)
    print(f"BIRD 训练集: {len(tr)} 例 / {st['n_dbs']} 库 / 平均 {st['avg_columns']:.1f} 列 "
          f"(取 {time.time()-t0:.1f}s)")
    emb = None
    if a.use_emb:
        try:
            from sentence_transformers import SentenceTransformer
            emb = SentenceTransformer(os.environ.get(
                "JEVSCHEMA_EMB_MODEL", os.path.join(_P.MODELS, "miniLM")))
            print("稠密编码器已加载")
        except Exception as exc:  # noqa: BLE001
            print(f"稠密编码器加载失败（退化成纯词法特征）: {type(exc).__name__}: {exc}")

    Xs, ys = [], []
    for i, ex in enumerate(tr):
        _, X, gold = features(ex, emb)
        y = np.asarray([1.0 if p in gold else 0.0 for p in ex.schema.columns()])
        Xs.append(X)
        ys.append(y)
        if (i + 1) % 500 == 0:
            print(f"  特征 {i+1}/{len(tr)}")
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)      # 常量特征（如未启用的 emb_*）不能除 0
    Xs_n = (X - mu) / sd
    print(f"样本 {X.shape[0]} 行（正例 {int(y.sum())}，占比 {y.mean()*100:.2f}%）")

    # 留 20% 做训练内验证，避免又在 dev 上看曲线
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    ntr = int(0.8 * len(y))
    w, b, _ = fit(Xs_n[idx[:ntr]], y[idx[:ntr]])
    from jevschema.metrics import auroc as _auc
    p_va = 1 / (1 + np.exp(-np.clip(Xs_n[idx[ntr:]] @ w + b, -30, 30)))
    print(f"训练内留出 AUROC = {_auc(p_va, y[idx[ntr:]]):.4f}")
    # 用全量重fit（留出集只是看曲线）
    w, b, _ = fit(Xs_n, y)
    print("逻辑回归权重（标准化后）:")
    for f, wi in sorted(zip(FEATS, w), key=lambda kv: -abs(kv[1])):
        print(f"  {f:<12} {wi:+.4f}")

    # ---------------------------------------------------------- Dev 评估
    schemas = load_bird_schemas()
    dev = load_bird_dev(limit=a.n_dev if a.n_dev > 0 else None, schemas=schemas, shuffle=False)
    print(f"\nBIRD-Dev n={len(dev)}  补全={a.closure}")

    aucs, per_b, kept_b, probs_all, ys_all = [], {b_: [] for b_ in budgets}, \
        {b_: [] for b_ in budgets}, [], []
    # 手挑基线的分数（bm25+emb 0.5/0.5）用于对照
    base_per_b = {b_: [] for b_ in budgets}
    for ex in dev:
        pairs, Xd, gold = features(ex, emb)
        Xn = (Xd - mu) / sd
        p = 1 / (1 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        yv = np.asarray([1.0 if q in gold else 0.0 for q in pairs], dtype=int)
        aucs.append(auroc(p, yv))
        probs_all.append(p)
        ys_all.append(yv)
        order = [pairs[j] for j in np.argsort(-p, kind="stable")]
        # 手挑基线
        bn = Xd[:, 0]
        en = Xd[:, 4]
        s0 = 0.5 * bn + 0.5 * en
        order0 = [pairs[j] for j in np.argsort(-s0, kind="stable")]
        for bb in budgets:
            kept = closure(ex.schema, order, bb, cfg)
            kept_b[bb].append(len(kept))
            per_b[bb].append(strict_recall(ex.gold, kept))
            base_per_b[bb].append(strict_recall(
                ex.gold, closure(ex.schema, order0, bb, cfg)))

    pp = np.concatenate(probs_all)
    yy = np.concatenate(ys_all)
    rec = {
        "n_train": len(tr), "n_dev": len(dev), "closure": a.closure,
        "auroc": round(float(np.nanmean(aucs)), 4),
        "ece": round(float(ece(pp, yy)[0]), 4),
        "brier": round(float(brier(pp, yy)), 4),
    }
    print(f"\n学习式打分器  AUROC={rec['auroc']:.4f}  ECE={rec['ece']:.4f}  "
          f"Brier={rec['brier']:.4f}")
    print(f"\n{'预算':<6}{'SRR(学习)':>10}{'SRR(手挑)':>11}{'Δ':>9}{'C̄':>8}{'p':>9}")
    for bb in budgets:
        v = float(np.mean(per_b[bb]))
        v0 = float(np.mean(base_per_b[bb]))
        pb = paired_bootstrap(base_per_b[bb], per_b[bb])
        rec[f"srr@{bb}"] = round(v, 4)
        rec[f"baseline_srr@{bb}"] = round(v0, 4)
        rec[f"delta@{bb}"] = round(v - v0, 4)
        rec[f"p@{bb}"] = round(float(pb["p_boot"]), 4)
        rec[f"cbar@{bb}"] = round(float(np.mean(kept_b[bb])), 2)
        ci = mean_ci(per_b[bb])
        print(f"C={bb:<4}{v:>10.4f}{v0:>11.4f}{v-v0:>+9.4f}"
              f"{float(np.mean(kept_b[bb])):>8.1f}{pb['p_boot']:>9.4f}"
              f"   CI=[{ci[1]:.4f},{ci[2]:.4f}]")

    path = a.out or os.path.join(RESULTS, "lex_classifier_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feats": FEATS, "w": [float(x) for x in w],
                   "bias": float(b),
                   "mu": [float(x) for x in mu],
                   "sd": [float(x) for x in sd],
                   "rec": rec}, f, ensure_ascii=False, indent=2)
    print(f"\n模型与结果写入: {path}")


if __name__ == "__main__":
    main()
