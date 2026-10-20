"""结构补全消融：FK 闭包 → +入选表内补齐 → +小表整表保留。

失败诊断（failure_diag.py）给出的结论非常干净：

    A_词汇断链   99 题（占失败 69.7%）—— gold 列名与 question+evidence 零词汇重叠，
                 且 embedding 也排不进前列
    D_表级漏召    0 题 —— gold 列所在的表**从来没有被整体漏掉**

也就是说：这 99 题里缺失的列，几乎都躺在"已经被选中至少一列"的表里
（样例里同表已保留 1/2/4/5/9/15/22 列）。真正断的不是"表"，是"表内某一列的别名"
（satscores.NumTstTakr ← "number of test takers"）。

这直接推出一个新机制 —— **入选表内补齐（table padding）**：
    一张表只要有任意一列进了 top-k，就把这张表里分数次高的若干列也带上。
代价是几列 token，收益是把"别名断链"这一整类失败按表兜住。

它和 FK 闭包是互补的：
    FK 闭包     补的是"跨表连接键"（横向，跨表）
    表内补齐     补的是"表内同义别名"（纵向，表内）
两者都是**结构性规则、零模型调用、零额外 LLM 开销**，
正好落在论文要讲的"把 schema linking 从昂贵的迭代探索变成一次性批量判断"上。

消融的三个旋钮：
    --fk        FK 闭包（已验证 +15.3 SRR）
    --pad p     入选表内再补 p 列（按融合分取该表内次高的 p 列）
    --small M   入选表里总列数 ≤ M 的，整表保留
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
from jevschema.selection import expand_kept  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


def _load(kind, examples, fn):
    """fn=None 表示只读：缓存里没有就留空，**绝不用别的信号顶替**。

    曾经在这里踩过坑：`_load("embedding", ex, bm25_of)` 会把 bm25 分数写进
    embedding 缓存，2000 行假 embedding 混进 bird_embedding.jsonl，
    后面所有"融合"实验静默变成"bm25 的两种归一化相加"。
    """
    p = os.path.join(CACHE, f"bird_{kind}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    if fn is None:
        miss = sum(1 for e in examples if e.ex_id not in store)
        if miss:
            print(f"  [{kind}] 缓存缺 {miss} 例，不给算（避免信号顶替），按 0 处理")
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


def closure(schema, order_pairs, k, fk=True, pad=0, small=0):
    """top-k → 表内补齐 → 小表整表 → FK 闭包。返回保留列（去重、顺序稳定）。"""
    kept = list(order_pairs[:k])
    idx = {p: i for i, p in enumerate(order_pairs)}
    by_table = {}
    for p in order_pairs:
        by_table.setdefault(p[0], []).append(p)
    have = {p[0] for p in kept}
    add = []
    for t in have:
        cols = by_table[t]
        sel = sum(1 for p in kept if p[0] == t)
        if small and len(cols) <= small:
            add.extend(cols)
            continue
        if pad:
            add.extend(cols[: min(len(cols), sel + pad)])
    kept = list(dict.fromkeys(list(kept) + list(add)))
    if fk:
        kept = expand_kept(schema, kept, "fk")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--budgets", default="15,20,25,30,35,40,50")
    ap.add_argument("--grid", default="f1p0s0,f1p2s0,f1p3s0,f1p5s0,f0p3s0,"
                                      "f1p0s6,f1p0s10,f1p2s10,f1p3s10,f0p3s10")
    ap.add_argument("--out", default="")
    ap.add_argument("--source", default="bird_dev", choices=["bird_dev", "bird_train"],
                    help="bird_train = 在 BIRD 训练集（与 dev 零库重叠）上选超参。"
                         "dev 上挑的补齐参数属于测试集调参，必须用训练集重选一遍才有说服力。")
    ap.add_argument("--n-train", type=int, default=1500)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]

    if a.source == "bird_train":
        from jevschema.bird_train import load_bird_train
        ex, _st = load_bird_train(limit=a.n_train)
        print(f"超参选择集 = BIRD train，{len(ex)} 例")
    else:
        schemas = load_bird_schemas()
        ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    bm = _load("bm25", ex, bm25_of)
    # 只读：训练集没有 embedding 缓存，就退化成纯 bm25（并明确打印出来），
    # 不允许拿 bm25 顶替 embedding 写进缓存。
    eb = _load("embedding", ex, None)
    have_emb = any(len(d) > 0 for d in eb)
    print(f"BIRD-Dev n={len(ex)}  融合={'bm25+emb' if have_emb else 'bm25 only'}")

    meta = []
    for e in ex:
        pairs = list(e.schema.columns())
        gold = e.gold.flat_pairs()
        y = np.asarray([1 if p in gold else 0 for p in pairs], dtype=int)
        meta.append((e, pairs, gold, y))

    # 融合分（与 scorer_fusion 的 bm25+emb 同口径）
    fused = []
    for (e, pairs, gold, y), b, d in zip(meta, bm, eb):
        bn = _norm({p: float(b.get(p, 0.0)) for p in pairs}, "max")
        if have_emb:
            dn = _norm({p: float(d.get(p, 0.0)) for p in pairs}, "minmax")
        else:
            dn = {p: 0.0 for p in pairs}
        s = np.asarray([0.5 * bn[p] + 0.5 * dn[p] for p in pairs])
        fused.append(s)

    orders = []
    for s, (e, pairs, gold, y) in zip(fused, meta):
        o = np.argsort(-s, kind="stable")
        orders.append([pairs[j] for j in o])

    # 固定下标解析（g[1]/g[3]/g[5:]）在 pad 是两位数时会错（f1p12s0 -> g[5:]='s0'），用正则。
    _G = re.compile(r"^f(\d+)p(\d+)s(\d+)$")

    def parse(g):
        m = _G.match(g.strip())
        if not m:
            raise ValueError(f"无法解析补全配置: {g!r}（应为 f{{0,1}}p<数>s<数>）")
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    grids = [parse(g) for g in a.grid.split(",")]
    print(f"\n{'fk/pad/small':<14}{'C̄@35':>7} " + " ".join(f"C{b}" for b in budgets))
    out = {"n": len(ex), "budgets": budgets, "grid": {}}
    base_srr = {}
    for (fk, pad, small) in grids:
        name = f"f{fk}p{pad}s{small}"
        per_b, kept_b = {b: [] for b in budgets}, {b: [] for b in budgets}
        for (e, pairs, gold, y), o in zip(meta, orders):
            for b in budgets:
                kept = closure(e.schema, o, b, fk=bool(fk), pad=pad, small=small)
                kept_b[b].append(len(kept))
                per_b[b].append(strict_recall(e.gold, kept))
        rec = {}
        for b in budgets:
            rec[f"srr@{b}"] = round(float(np.mean(per_b[b])), 4)
            ci = mean_ci(per_b[b])
            rec[f"srr@{b}_ci"] = [round(ci[1], 4), round(ci[2], 4)]
            rec[f"cbar@{b}"] = round(float(np.mean(kept_b[b])), 2)
        base_srr[name] = per_b
        out["grid"][name] = rec
        print(f"{name:<14}{rec['cbar@35']:>7.1f}  " +
              " ".join(f"{rec[f'srr@{b}']:.3f}" for b in budgets))

    print("\n== C=35 排序（基线 f1p0s0 = 现有 FK 闭包）==")
    ref = "f1p0s0"
    for name, rec in sorted(out["grid"].items(), key=lambda kv: -kv[1]["srr@35"]):
        ci = rec["srr@35_ci"]
        extra = ""
        if ref in base_srr and name != ref:
            p = paired_bootstrap(base_srr[ref][35], base_srr[name][35])["p_boot"]
            extra = f"   Δvs基线={rec['srr@35']-out['grid'][ref]['srr@35']:+.4f} p={p:.4f}"
        print(f"  {name:<12} SRR={rec['srr@35']:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}] "
              f"C̄={rec['cbar@35']:.1f}{extra}")

    print("\n== 效率：每多留 1 列换到多少 SRR（C=35，相对 f0p0s0 无补全）==")
    if "f0p0s0" in out["grid"]:
        z = out["grid"]["f0p0s0"]
        for name, rec in sorted(out["grid"].items(), key=lambda kv: -kv[1]["srr@35"]):
            ds = rec["srr@35"] - z["srr@35"]
            dc = rec["cbar@35"] - z["cbar@35"]
            if abs(dc) < 1e-6:
                continue
            print(f"  {name:<12} ΔSRR={ds:+.4f}  ΔC̄={dc:+5.1f}  每列={ds/dc*100:+.3f} SRR点/列")

    path = a.out or os.path.join(RESULTS, "closure_ablation_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
