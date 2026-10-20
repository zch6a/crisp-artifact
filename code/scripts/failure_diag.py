"""失败诊断：SRR 掉的那 9% 到底错在哪。

bm25+emb 融合在 C=35 上 SRR=0.9074（C̄=37.6），AutoLink 是 0.974。
要补这 6.7 个点，必须先知道失败的题是**哪一类**失败：

    A 词汇断链   gold 列的名字在 question+evidence 里一个词都不出现，
                 且 embedding 相似度也排不进前列 → 只能靠"推理型"信号
                 （外键/连接键/共现统计/决策模型）救
    B 同形干扰   有另一个同名的列排在它前面（BIRD 里 `id`/`name` 满地都是）
    C 排序挤压   gold 列分数不低，但被同表其它高分列挤到 35 名开外
    D 表级漏召   gold 列所在的表整体就没进前列
    E 隐式需求   gold 需求里含有 question 完全没提到的列（聚合/排序/group by 的中介列）

分类是互斥按优先级判定的（D > A > B > C），因为要的是"下一步该补什么"，
不是精确归因。E 单独统计（可与其他类并存），因为它是 SRR 口径本身的性质。

输出还带一个**可救上限**：把词汇断链的那部分用 oracle 补上，SRR 能到多少。
这个数字决定了"System-One 判断模型"这条路的收益天花板 —— 如果上限只有 +2，
那就不该把论文押在打分器上，而应该押在预算分配/成本上。
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
from jevschema.metrics import strict_recall  # noqa: E402
from jevschema.selection import expand_kept  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402

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
        out = v / hi if hi > 0 else np.zeros_like(v)
    else:
        lo, hi = v.min(), v.max()
        out = (v - lo) / max(1e-9, hi - lo)
    return {k: float(x) for k, x in zip(d.keys(), out)}


_WORD = re.compile(r"[a-z0-9_]+")


def q_tokens(ex):
    s = (ex.question + " " + (ex.evidence or "")).lower()
    out = set()
    for w in _WORD.findall(s):
        out.update(p for p in w.split("_") if p)
    return out


def col_tokens(col):
    out = set()
    for w in _WORD.findall(col.lower()):
        out.update(p for p in w.split("_") if p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=35)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    bm = _load("bm25", ex, bm25_of)
    # embedding 需要加载 sentence_transformers（16GB 内存上偶发 DLL 失败），
    # 所以只在缓存已存在时读；没有缓存就退化成纯 bm25，并在表头注明。
    emb_cache = os.path.join(CACHE, "bird_embedding.jsonl")
    if os.path.exists(emb_cache):
        eb = _load("embedding", ex, None)   # 只读：缺缓存就退化成纯 bm25，不允许顶替
        have_emb = any(len(d) > 0 for d in eb)
    else:
        eb = [{} for _ in ex]
        have_emb = False
    print(f"embedding 信号: {'有' if have_emb else '无（退化成纯 bm25）'}")

    fk_cols = {}
    for e in ex:
        s = set()
        for f in e.schema.fks:
            s.add((f.src_table, f.src_col))
            s.add((f.dst_table, f.dst_col))
        fk_cols[e.ex_id] = s

    cats = {"A_词汇断链": 0, "B_同形干扰": 0, "C_排序挤压": 0, "D_表级漏召": 0}
    multi = 0
    implicit = 0          # E：gold 需求里 question/evidence 完全没提到的列
    fixable = 0           # 若把 A 类用 oracle 补上能救回多少题
    fails = []
    n_fail = 0
    for e, b, d in zip(ex, bm, eb):
        pairs = list(e.schema.columns())
        gold = e.gold.flat_pairs()
        bn = _norm({p: float(b.get(p, 0.0)) for p in pairs}, "max")
        if have_emb:
            dn = _norm({p: float(d.get(p, 0.0)) for p in pairs}, "minmax")
        else:
            dn = {p: 0.0 for p in pairs}
        s = np.asarray([0.5 * bn[p] + 0.5 * dn[p] for p in pairs])
        order = np.argsort(-s, kind="stable")
        kept = expand_kept(e.schema, [pairs[i] for i in order[:a.budget]], "fk")
        if strict_recall(e.gold, kept) >= 1.0:
            continue
        n_fail += 1
        miss = [p for p in gold if p not in set(kept)]
        qt = q_tokens(e)
        # order 里放的是下标，不是 pair —— 第一版写成 {p: i for i, p in enumerate(order)}
        # 导致每个 gold 列的 rank 都是 -1，A/B/C 三类判定全部塌到 C 类。
        ranks = {pairs[j]: int(i) for i, j in enumerate(order)}
        detail = []
        kinds = set()
        for p in miss:
            tb, cl = p
            ct = col_tokens(cl) | col_tokens(tb)
            lex = bool(ct & qt)
            rank = ranks.get(p, -1)
            # 该表进了多少列（表级漏召判定）
            same_table_kept = sum(1 for q in kept if q[0] == tb)
            # 同名干扰：有另一个同名列被保留
            homonym = any(q[1].lower() == cl.lower() and q != p for q in kept)
            if same_table_kept == 0:
                kinds.add("D_表级漏召")
            elif not lex and rank >= a.budget:
                kinds.add("A_词汇断链")
            elif homonym:
                kinds.add("B_同形干扰")
            else:
                kinds.add("C_排序挤压")
            if not lex:
                pass
            detail.append({"col": f"{tb}.{cl}", "rank": rank,
                           "bm25": round(float(b.get(p, 0.0)), 3),
                           "lex_overlap": lex, "homonym": homonym,
                           "table_kept_cols": same_table_kept,
                           "is_fk": p in fk_cols[e.ex_id]})
        # E：gold 里是否有完全无词汇重叠的需求（与上面分类并存）
        if any(not (col_tokens(c) | col_tokens(t)) & qt for t, c in gold):
            implicit += 1
        for k in kinds:
            cats[k] += 1
        if len(kinds) > 1:
            multi += 1
        if "A_词汇断链" in kinds or "D_表级漏召" in kinds:
            fixable += 1
        fails.append({"db": e.schema.db_id, "q": e.question[:90],
                      "kinds": sorted(kinds), "miss": detail})

    n = len(ex)
    srr = 1 - n_fail / n
    print(f"BIRD-Dev n={n}  C={a.budget}  SRR={srr:.4f}  失败 {n_fail} 题")
    print("\n失败原因（一题可命中多类，按 D>A>B>C 优先归类）")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v:>4} 题  ({v/max(1,n_fail)*100:.1f}% of 失败, "
              f"{v/n*100:.2f}% of 全集)")
    print(f"  多类并存        {multi} 题")
    print(f"  E_隐式需求      {implicit} 题（gold 里有 question 完全没提到的列）")
    print(f"\n天花板上限：把 A+D 类全部 oracle 补上 → SRR ≈ "
          f"{(n - n_fail + fixable)/n:.4f}（+{fixable/n*100:.2f} 点）")
    print(f"纯词汇融合已到 {srr:.4f}，AutoLink 0.974 → 缺口 {(0.974-srr)*100:.2f} 点；"
          f"A+D 全救回也只能补 {fixable/n*100:.2f} 点，"
          f"剩下 {(0.974-srr)*100 - fixable/n*100:.2f} 点必须靠别的机制。")

    print(f"\n== 失败样例（前 {a.show}）==")
    for f in fails[:a.show]:
        print(f"\n[{f['db']}] {f['q']}")
        print(f"  类型: {','.join(f['kinds'])}")
        for d in f["miss"][:4]:
            print(f"   - {d['col']:<40} rank={d['rank']:<4} bm25={d['bm25']:<7} "
                  f"词汇重叠={d['lex_overlap']} 同名={d['homonym']} "
                  f"同表保留={d['table_kept_cols']} FK={d['is_fk']}")

    path = a.out or os.path.join(RESULTS, "failure_diag_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": n, "budget": a.budget, "srr": round(srr, 4), "n_fail": n_fail,
                   "cats": cats, "multi": multi, "implicit": implicit,
                   "fixable_by_AD": fixable, "fails": fails[:200]}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
