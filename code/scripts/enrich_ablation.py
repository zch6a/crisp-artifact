"""列文档富化消融：样例值 / 类型 / 数据库内容匹配 / RRF 融合。

上一轮（scorer_fusion.py）已经证明 bm25+embedding 融合把 AUROC 0.8465 → 0.8772、
SRR@C=35 0.8814 → 0.9074。但两条信号用的都是**同一份列文档**（"表名 列名"），
它们互补的只是"稀疏 vs 稠密"，文档本身的信息量没有变。

这一轮改的是**文档本身**：
    +type      把列类型写进文档（数值列才能聚合，匹配到 "average/COUNT" 时有用）
    +samp      把 sqlite 里真实样例值写进文档（BIRD 列名大量是缩略/歧义的，
               样例值是 evidence 之外最强的消歧信号）
    +content   数据库内容匹配：问题里的词直接命中某列的**值**（不只是名字），
               这是 CHESS / RSL-SQL 那一类方法的核心信号，纯 CPU 就能做

融合方式也从加权和扩到 **RRF（倒数排序融合）** —— 加权和要两路分数同量纲，
RRF 只吃名次，对 BM25 那 80% 的并列 0 分天然免疫。

预算仍用 C̄（top-k 列 + fk 闭包），与 AutoLink 的 35.8 同口径。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import (  # noqa: E402
    BIRD_SQLITE_DIR,
    enrich_sample_values,
    load_bird_dev,
    load_bird_schemas,
)
from jevschema.fusion import FusionConfig, lexical_scores  # noqa: E402
from jevschema.metrics import auroc, mean_ci, strict_recall  # noqa: E402
from jevschema.selection import expand_kept  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


# ---------------------------------------------------------------- 文档构造
def col_doc(t, c, kind):
    base = f"{t} {t.replace('_',' ')} {c.name} {c.name.replace('_',' ')}"
    if kind == "base":
        return base
    if kind == "type":
        return f"{base} {c.dtype}"
    if kind == "samp":
        return f"{base} " + " ".join(c.sample_values or [])
    if kind == "samp_type":
        return f"{base} {c.dtype} " + " ".join(c.sample_values or [])
    raise ValueError(kind)


def docs_for(ex, kind):
    return [col_doc(t.name, c, kind)
            for t in ex.schema.tables for c in t.columns]


# ---------------------------------------------------------------- 数据库内容匹配
_WORD = re.compile(r"[a-z0-9']+")


def content_hits(ex, min_len=3):
    """问题里的词直接命中某列的样例值 → 该列 +1。

    这是"值检索"（value retrieval）的轻量版：不建索引，直接拿 enrich 出来的
    样例值做大小写无关的子串匹配。样例值越多召回越强，所以 k 是一个可调旋钮。
    """
    q = (ex.question + " " + (ex.evidence or "")).lower()
    toks = {w for w in _WORD.findall(q) if len(w) >= min_len}
    out = {}
    for t in ex.schema.tables:
        for c in t.columns:
            vals = " ".join(c.sample_values or []).lower()
            if not vals or not toks:
                out[(t.name, c.name)] = 0.0
                continue
            hit = 0
            for w in toks:
                # 词边界匹配，避免 "id" 命中 "acid"
                if re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", vals):
                    hit += 1
            out[(t.name, c.name)] = float(hit)
    return out


# ---------------------------------------------------------------- 分数缓存
def _load(kind, ex, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"bird_{kind}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    todo = [e for e in ex if e.ex_id not in store]
    if todo:
        with open(p, "a", encoding="utf-8") as f:
            for e in todo:
                sc = fn(e)
                store[e.ex_id] = sc
                f.write(json.dumps({"id": e.ex_id,
                                    "s": {"|".join(k): v for k, v in sc.items()}},
                                   ensure_ascii=False) + "\n")
        print(f"  [{kind}] 新算 {len(todo)} 例")
    return [store[e.ex_id] for e in ex]


def _bm25_view(kind):
    def f(ex):
        return dict(zip(ex.schema.columns(), bm25_scores(
            ex.question + (" " + ex.evidence if ex.evidence else ""),
            docs_for(ex, kind))))
    return f


def _emb_view(kind):
    def f(ex):
        cfg = FusionConfig(lexical="embedding")
        # embedding 的文档构造复用 fusion._embed 的文本；这里为省时间只测一种富化
        from jevschema import fusion
        docs = docs_for(ex, kind)
        if kind == "base":
            return fusion._embed(ex, cfg)
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg.embedding_model)
        q = ex.question + (" " + ex.evidence if ex.evidence else "")
        emb = m.encode([q] + docs, normalize_embeddings=True, show_progress_bar=False)
        sims = emb[1:] @ emb[0]
        return {p: float(s) for p, s in zip(ex.schema.columns(), sims)}
    return f


# ---------------------------------------------------------------- 融合
def _norm(d, how="max"):
    if not d:
        return {}
    v = np.asarray(list(d.values()), dtype=float)
    if how == "max":
        hi = v.max()
        out = v / hi if hi > 0 else np.zeros_like(v)
    elif how == "minmax":
        lo, hi = v.min(), v.max()
        out = (v - lo) / max(1e-9, hi - lo)
    else:
        out = v
    return {k: float(x) for k, x in zip(d.keys(), out)}


def rrf(lists, k=60):
    """倒数排序融合：只吃名次，对并列 0 分免疫。"""
    n = len(lists[0])
    out = np.zeros(n)
    for lst in lists:
        v = np.asarray(lst, dtype=float)
        order = np.argsort(-v, kind="stable")
        r = np.empty(n)
        r[order] = np.arange(1, n + 1)
        out += 1.0 / (k + r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--samples", type=int, default=8, help="每列样例值个数 k")
    ap.add_argument("--budgets", default="15,20,25,30,35,40,50")
    ap.add_argument("--out", default="")
    ap.add_argument("--with-emb-doc", action="store_true")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]

    schemas = load_bird_schemas()
    nok = enrich_sample_values(schemas, BIRD_SQLITE_DIR, k=a.samples)
    print(f"样例值回填：{nok}/{len(schemas)} 个库 (k={a.samples})")
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(ex)}")

    sig = {}
    for kind in ("base", "type", "samp", "samp_type"):
        sig[f"bm25_{kind}"] = _load(f"bm25doc_{kind}_{a.samples}", ex, _bm25_view(kind))
    # 稠密信号用已有的 base 缓存（1534 例齐全）。富化版要先 import
    # sentence_transformers -> transformers -> sklearn -> scipy.linalg._flapack，
    # 在 16GB 内存 / 小页面文件上会偶发 "页面文件太小" 的 DLL 加载失败，
    # 所以默认关掉，需要时用 --with-emb-doc 单独跑。
    sig["emb_base"] = _load("embedding", ex, _emb_view("base"))
    have_emb_st = False
    if a.with_emb_doc:
        try:
            sig["emb_samp_type"] = _load(f"embdoc_samp_type_{a.samples}", ex,
                                         _emb_view("samp_type"))
            have_emb_st = True
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过 emb_samp_type] {type(exc).__name__}: {exc}")
    sig["content"] = _load(f"content_{a.samples}", ex, content_hits)
    print("信号就绪:", ", ".join(sig))

    meta = []
    for e in ex:
        pairs = list(e.schema.columns())
        gold = e.gold.flat_pairs()
        y = np.asarray([1 if p in gold else 0 for p in pairs], dtype=int)
        meta.append((e, pairs, gold, y))

    variants = {
        "bm25_base":         [("bm25_base", 1.0, "max")],
        "bm25_type":         [("bm25_type", 1.0, "max")],
        "bm25_samp":         [("bm25_samp", 1.0, "max")],
        "bm25_samp_type":    [("bm25_samp_type", 1.0, "max")],
        "emb_base":          [("emb_base", 1.0, "minmax")],
        "content":           [("content", 1.0, "max")],
        "bm25st+emb":        [("bm25_samp_type", 0.5, "max"), ("emb_base", 0.5, "minmax")],
        "bm25st+emb+ct":     [("bm25_samp_type", 0.4, "max"), ("emb_base", 0.35, "minmax"),
                              ("content", 0.25, "max")],
        "RRF(bm25st,emb,ct)": None,   # 特殊处理（只吃名次）
        "RRF(bm25,emb)": None,
        "RRF(bm25st,emb)": None,
    }
    rrf_sets = {
        "RRF(bm25st,emb,ct)": ["bm25_samp_type", "emb_base", "content"],
        "RRF(bm25,emb)": ["bm25_base", "emb_base"],
        "RRF(bm25st,emb)": ["bm25_samp_type", "emb_base"],
    }
    if have_emb_st:
        variants["bm25st+embst"] = [("bm25_samp_type", 0.5, "max"),
                                    ("emb_samp_type", 0.5, "minmax")]
        variants["bm25st+embst+ct"] = [("bm25_samp_type", 0.4, "max"),
                                       ("emb_samp_type", 0.35, "minmax"),
                                       ("content", 0.25, "max")]

    print(f"\n{'variant':<20}{'AUROC':>8} " + " ".join(f"C{b}" for b in budgets))
    out = {"n": len(ex), "budgets": budgets, "samples_k": a.samples, "variants": {}}
    for name, parts in variants.items():
        if parts is None:
            keys = rrf_sets[name]
            cols = [[sig[k][i] for k in keys] for i in range(len(ex))]
        else:
            cols = [[sig[k][i] for k, _, _ in parts] for i in range(len(ex))]
        aucs, per_b, kept_b = [], {b: [] for b in budgets}, {b: [] for b in budgets}
        for i, (e, pairs, gold, y) in enumerate(meta):
            if parts is None:
                ls = [[float(d.get(p, 0.0)) for p in pairs] for d in cols[i]]
                s = rrf(ls)
            else:
                s = np.zeros(len(pairs))
                for (k, w, how), d in zip(parts, cols[i]):
                    dn = _norm({p: float(d.get(p, 0.0)) for p in pairs}, how)
                    s += w * np.asarray([dn[p] for p in pairs])
            aucs.append(auroc(s, y))
            order = np.argsort(-s, kind="stable")
            for b in budgets:
                kept = expand_kept(e.schema, [pairs[j] for j in order[:b]], "fk")
                kept_b[b].append(len(kept))
                per_b[b].append(strict_recall(e.gold, kept))
        rec = {"auroc": round(float(np.nanmean(aucs)), 4)}
        for b in budgets:
            rec[f"srr@{b}"] = round(float(np.mean(per_b[b])), 4)
            ci = mean_ci(per_b[b])
            rec[f"srr@{b}_ci"] = [round(ci[1], 4), round(ci[2], 4)]
            rec[f"cbar@{b}"] = round(float(np.mean(kept_b[b])), 2)
        out["variants"][name] = rec
        print(f"{name:<20}{rec['auroc']:>8.4f}  " +
              " ".join(f"{rec[f'srr@{b}']:.3f}" for b in budgets))

    for b in (25, 35, 50):
        print(f"\n== C={b} 排序 ==")
        for name, rec in sorted(out["variants"].items(), key=lambda kv: -kv[1][f"srr@{b}"]):
            ci = rec[f"srr@{b}_ci"]
            print(f"  {name:<20} SRR={rec[f'srr@{b}']:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}] "
                  f"AUROC={rec['auroc']:.4f} C̄={rec[f'cbar@{b}']:.1f}")

    path = a.out or os.path.join(RESULTS, "enrich_ablation_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
