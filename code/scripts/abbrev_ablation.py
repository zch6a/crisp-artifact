"""缩写断链专项：辅音骨架匹配 + 缩写词典。

失败诊断说得很清楚 —— BIRD-Dev 上 SRR 的缺口 100% 来自 A 类：
gold 列名与 question+evidence **零词汇重叠**，embedding 也救不回来。看具体样例：

    "number of test takers"     -> satscores.NumTstTakr
    "average score in Writing"  -> satscores.AvgScrWrite
    "eligible free rate for K-12" -> frpm.`Free Meal Count (K-12)`

BIRD 的列名是**缩写体**：元音被大量省略。英文里"去元音"正是标准的
name-matching 手段（consonant skeleton / Soundex 那一家），因为
    test taker -> tst tkr -> tsttkr
    NumTstTakr -> num tst takr -> ...tst...takr...     骨架上高度重合
    average score -> vrg scr ；AvgScrWrite -> vg scr wrt  "scr" 直接命中

本脚本测三档信号，都是纯 CPU、零模型：
    cons      辅音骨架：doc 与 query 的 token 都额外吐一份去元音形式
    abbr      缩写词典：avg/num/cnt/scr/amt/qty... 常见缩写展开成全词
    cons+abbr 两者叠加
再与 bm25 / embedding 融合，看能不能把 A 类那 6.45 个点拿回来。

判定口径与 scorer_fusion 完全一致（AUROC + SRR@C，fk 闭包 f1）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import auroc, mean_ci, paired_bootstrap, strict_recall  # noqa: E402
from jevschema.selection import parse_closure, closure  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")

VOWELS = "aeiou"


def skeleton(w: str) -> str:
    """去元音骨架。全元音词（如 "of"/"in"）保留原样，否则会退化成空串噪声。"""
    s = re.sub(r"[aeiouy]", "", w)
    return s if len(s) >= 2 else w


def split_tokens(s: str):
    out = []
    for raw in re.findall(r"[a-z0-9_]+", s.lower()):
        out.extend(p for p in raw.split("_") if p)
    return out


ABBR = {
    "avg": "average", "avrg": "average", "amt": "amount", "qty": "quantity",
    "cnt": "count", "num": "number", "nbr": "number", "no": "number",
    "scr": "score", "tst": "test", "tkr": "taker", "takrs": "takers",
    "std": "student", "stdnt": "student", "stu": "student",
    "yr": "year", "dt": "date", "mon": "month", "pct": "percent",
    "desc": "description", "nm": "name", "addr": "address",
    "tel": "telephone", "phn": "phone", "id": "identifier",
    "cat": "category", "catg": "category", "typ": "type",
    "elig": "eligible", "frpm": "free reduced price meal",
    "enrl": "enrollment", "schl": "school", "sch": "school",
    "dist": "district", "cnty": "county", "cd": "code", "cds": "codes",
    "tot": "total", "sum": "sum", "max": "maximum", "min": "minimum",
    "val": "value", "vals": "values", "txt": "text", "flg": "flag",
    "ind": "indicator", "pct": "percent", "prc": "price", "cst": "cost",
    "wgt": "weight", "hgt": "height", "len": "length", "amt": "amount",
}


def expand(w: str, use_abbr: bool, use_cons: bool):
    """一个 token 展开成若干匹配形式。"""
    out = [w]
    if use_abbr:
        if w in ABBR:
            out.extend(ABBR[w].split())
        # 反向：全词 -> 常见缩写（让 "average" 能命中 "avg"）
        for k, v in ABBR.items():
            if v == w:
                out.append(k)
    if use_cons:
        out.append(skeleton(w))
    return out


def make_scorer(use_abbr: bool, use_cons: bool, doc_base: bool = True):
    """返回一个 (query_tokens, doc_tokens) 版本的 BM25。

    复用 summary.bm25_scores 的 IDF/TF 公式，但把分词替换成"展开版"：
    两边都吐全词 + 骨架 + 缩写展开，缩写体就能被普通词命中。
    """
    import math

    def tok(s: str):
        out = []
        for w in split_tokens(s):
            out.extend(expand(w, use_abbr, use_cons))
        return out

    def sc(ex):
        pairs = ex.schema.columns()
        q = tok(ex.question + " " + (ex.evidence or ""))
        docs = []
        for t, c in pairs:
            base = f"{t} {c}" if doc_base else c
            docs.append(tok(t.replace("_", " ") + " " + c.replace("_", " "))
                        if not doc_base else
                        tok(t.replace("_", " ") + " " + c.replace("_", " ")))
        dl = [len(d) for d in docs]
        avgdl = (sum(dl) / len(dl)) if dl else 1.0
        df, tfs = {}, []
        for d in docs:
            tf = {}
            for x in d:
                tf[x] = tf.get(x, 0) + 1
            tfs.append(tf)
            for x in tf:
                df[x] = df.get(x, 0) + 1
        n = len(docs)
        k1, b = 1.5, 0.75
        out = []
        for tf, L in zip(tfs, dl):
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
                denom = tf[w] + k1 * (1 - b + b * L / max(avgdl, 1e-9))
                s += idf * tf[w] * (k1 + 1) / max(denom, 1e-9)
            out.append(s)
        return dict(zip(pairs, out))

    return sc


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
        print(f"  [{kind}] 新算 {len(todo)} 例")
    return [store[e.ex_id] for e in examples]


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
    ap.add_argument("--budgets", default="10,15,20,25,30,35,40")
    ap.add_argument("--closure", default="f1")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    cfg = parse_closure(a.closure)

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(ex)}  补全={a.closure}")

    base = _load("bm25", ex, make_scorer(False, False))
    cons = _load("bm25cons", ex, make_scorer(False, True))
    abbr = _load("bm25abbr", ex, make_scorer(True, False))
    both = _load("bm25consabbr", ex, make_scorer(True, True))
    emb_cache = os.path.join(CACHE, "bird_embedding.jsonl")
    emb = _load("embedding", ex, None)   # 只读：缺缓存就退化成纯 bm25，不允许顶替
    have_emb = any(len(d) > 0 for d in emb)
    print(f"embedding 信号: {'有' if have_emb else '无'}")

    meta = [(e, list(e.schema.columns()), e.gold.flat_pairs()) for e in ex]
    y_all = []
    for e, pairs, gold in meta:
        y_all.append(np.asarray([1 if p in gold else 0 for p in pairs], dtype=int))

    variants = {
        "bm25":            [("base", 1.0, "max")],
        "bm25+cons":       [("cons", 1.0, "max")],
        "bm25+abbr":       [("abbr", 1.0, "max")],
        "bm25+cons+abbr":  [("both", 1.0, "max")],
        "bm25+emb":        [("base", 0.5, "max"), ("emb", 0.5, "minmax")],
        "cons+emb":        [("cons", 0.5, "max"), ("emb", 0.5, "minmax")],
        "consabbr+emb":    [("both", 0.5, "max"), ("emb", 0.5, "minmax")],
        "cons+abbr+emb3":  [("base", 0.3, "max"), ("both", 0.35, "max"),
                            ("emb", 0.35, "minmax")],
    }
    sig = {"base": base, "cons": cons, "abbr": abbr, "both": both, "emb": emb}

    print(f"\n{'variant':<18}{'AUROC':>8} " + " ".join(f"C{b}" for b in budgets))
    out = {"n": len(ex), "budgets": budgets, "closure": a.closure, "variants": {}}
    per_lists = {}
    for name, parts in variants.items():
        aucs = []
        per_b = {b: [] for b in budgets}
        kept_b = {b: [] for b in budgets}
        for i, (e, pairs, gold) in enumerate(meta):
            cols = [sig[k][i] for k, _, _ in parts]
            s = np.zeros(len(pairs))
            for (k, w, how), d in zip(parts, cols):
                dn = _norm({p: float(d.get(p, 0.0)) for p in pairs}, how)
                s += w * np.asarray([dn[p] for p in pairs])
            aucs.append(auroc(s, y_all[i]))
            order = [pairs[j] for j in np.argsort(-s, kind="stable")]
            for b in budgets:
                kept = closure(e.schema, order, b, cfg)
                kept_b[b].append(len(kept))
                per_b[b].append(strict_recall(e.gold, kept))
        per_lists[name] = per_b
        rec = {"auroc": round(float(np.nanmean(aucs)), 4)}
        for b in budgets:
            rec[f"srr@{b}"] = round(float(np.mean(per_b[b])), 4)
            ci = mean_ci(per_b[b])
            rec[f"srr@{b}_ci"] = [round(ci[1], 4), round(ci[2], 4)]
            rec[f"cbar@{b}"] = round(float(np.mean(kept_b[b])), 2)
        out["variants"][name] = rec
        print(f"{name:<18}{rec['auroc']:>8.4f}  " +
              " ".join(f"{rec[f'srr@{b}']:.3f}" for b in budgets))

    b35 = 35 if 35 in budgets else budgets[-1]
    print(f"\n== C={b35} 排序 ==")
    ref = "bm25+emb"
    for name, rec in sorted(out["variants"].items(), key=lambda kv: -kv[1][f"srr@{b35}"]):
        ci = rec[f"srr@{b35}_ci"]
        extra = ""
        if name != ref and ref in per_lists:
            pb = paired_bootstrap(per_lists[ref][b35], per_lists[name][b35])
            extra = (f"  Δvs bm25+emb={rec[f'srr@{b35}']-out['variants'][ref][f'srr@{b35}']:+.4f}"
                     f" p={pb['p_boot']:.4f}")
        print(f"  {name:<18} SRR={rec[f'srr@{b35}']:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}] "
              f"AUROC={rec['auroc']:.4f} C̄={rec[f'cbar@{b35}']:.1f}{extra}")

    path = a.out or os.path.join(RESULTS, "abbrev_ablation_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
