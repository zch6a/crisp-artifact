"""按库规模拆解：schema linking 的 token 收益到底是谁的函数？

合成实验（scaling_lex.py）已经给出方向，但合成干扰库在论文里总会被质疑。
BIRD-Dev 自己就覆盖了 **11 ~ 199 列、231 ~ 1755 tokens** 的真实跨度（7.6 倍），
所以完全可以在**真实数据**上画这条曲线，不需要合成。

要回答的问题只有一个，但它决定整篇论文的立论方式：
    压缩省下的 token 比例，是"方法有多好"的函数，还是"库有多大"的函数？

如果主要是库规模的函数，那么：
  * 在 Spider（平均 26 列、262 tokens）上比 SRR 几乎没有成本意义 —— 省 7%；
  * 在 BIRD（平均 100 列、651 tokens）上省 21%~33%；
  * 在真实企业库（数百~数千列）上省 75%+；
  * 于是"该不该上 System-One"的判据就是**库规模**，而不是任何精度指标。
这正是路由规则的实证基础。

口径与 pareto_closure 完全一致（同一套融合分、同一套闭包配置串），
这样三张表（pareto_closure / scaling_lex / 本脚本）的数字能对得上。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import mean_ci, strict_recall  # noqa: E402
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
    ap.add_argument("--target", type=float, default=0.95,
                    help="SRR 门槛：达到它才有资格谈'省了多少'")
    ap.add_argument("--grid", default="f1p5s15k1,f1p0s15k1,f1p8k1,f1p5k1,f1p3s15k1")
    ap.add_argument("--ks", default="8,10,12,15,18,20,25,30,35,40,50,60")
    ap.add_argument("--s1-cost", type=float, default=1788.0,
                    help="System-One(laya) 实测 linking 成本 tokens/查询")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    cfgs = [parse_closure(g) for g in a.grid.split(",")]

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    # 空 gold 会让 strict_recall 平凡返回 1.0（BIRD 是 0 条，但保底过滤一次）
    n0 = len(ex)
    ex = [e for e in ex if e.gold is not None and e.gold.flat_pairs()]
    if n0 != len(ex):
        print(f"  剔除空 gold {n0 - len(ex)} 条")

    bm = _load("bm25", ex, bm25_of)
    eb = _load("embedding", ex, None)
    have_emb = any(len(d) > 0 for d in eb)
    print(f"BIRD-Dev n={len(ex)}  融合={'bm25+emb' if have_emb else 'bm25 only'}"
          f"  target SRR≥{a.target}  grid={a.grid}", flush=True)

    orders = []
    for e, b, d in zip(ex, bm, eb):
        pairs = e.schema.columns()
        bn = _norm({p: float(b.get(p, 0.0)) for p in pairs}, "max")
        dn = (_norm({p: float(d.get(p, 0.0)) for p in pairs}, "minmax")
              if have_emb else {p: 0.0 for p in pairs})
        s = np.asarray([0.5 * bn[p] + 0.5 * dn[p] for p in pairs])
        orders.append([pairs[j] for j in np.argsort(-s, kind="stable")])

    full_tok = np.asarray([
        make_stage_tokens(e.schema, e.question, e.evidence).total_tokens
        for e in ex])

    # 先算出每个 (example, cfg, k) 的 SRR 和 token，再按库聚合
    # ——不能先按库聚合再挑 k：每个库的最优 k 不同，这才是要看的东西。
    per = {(c.tag(), k): {"srr": np.zeros(len(ex)), "tok": np.zeros(len(ex))}
           for c in cfgs for k in ks}
    for cfg in cfgs:
        for k in ks:
            for i, e in enumerate(ex):
                kept = closure(e.schema, orders[i], k, cfg)
                per[(cfg.tag(), k)]["srr"][i] = strict_recall(e.gold, kept)
                per[(cfg.tag(), k)]["tok"][i] = make_stage_tokens(
                    e.schema, e.question, e.evidence,
                    kept_columns=kept).gen_prompt_tokens

    groups = defaultdict(list)
    for i, e in enumerate(ex):
        groups[e.db_id].append(i)

    print(f"\n== 每个库：达到 SRR≥{a.target:.2f} 的最省配置，以及省了多少 ==")
    print(f"{'db_id':<28}{'列数':>5}{'表':>4}{'n':>5}{'full':>7}"
          f"{'最省 tok':>9}{'SRR':>8}{'省下':>7}{'省%':>7}{'付得起S1':>10}")
    rows = []
    for db, idx in sorted(groups.items(),
                          key=lambda kv: -ex[kv[1][0]].schema.n_columns()):
        ii = np.asarray(idx)
        ft = float(full_tok[ii].mean())
        ncols = ex[idx[0]].schema.n_columns()
        ntabs = len(set(t for t, _ in ex[idx[0]].schema.columns()))
        best = None
        for (tag, k), d in per.items():
            s = float(d["srr"][ii].mean())
            t = float(d["tok"][ii].mean())
            if s < a.target:
                continue
            if best is None or t < best[2]:
                best = (tag, k, t, s)
        if best is None:
            # 达不到门槛：报全网格最高 SRR，明确标"不可达"，不拿低分充数
            hs = max(per.items(), key=lambda kv: float(kv[1]["srr"][ii].mean()))
            s = float(hs[1]["srr"][ii].mean()); t = float(hs[1]["tok"][ii].mean())
            r = {"db": db, "cols": ncols, "tables": ntabs, "n": len(ii),
                 "full_tokens": round(ft, 1), "reachable": False,
                 "best_cfg": hs[0][0], "best_k": hs[0][1],
                 "tokens": round(t, 1), "srr": round(s, 4),
                 "saving": round(1 - t / ft, 4)}
            print(f"{db:<28}{ncols:>5}{ntabs:>4}{len(ii):>5}{ft:>7.0f}"
                  f"{t:>9.0f}{s:>8.4f}{'—':>7}{'—':>7}{'不可达':>10}")
        else:
            tag, k, t, s = best
            saved = ft - t
            r = {"db": db, "cols": ncols, "tables": ntabs, "n": len(ii),
                 "full_tokens": round(ft, 1), "reachable": True,
                 "best_cfg": tag, "best_k": k, "tokens": round(t, 1),
                 "srr": round(s, 4), "saved_tokens": round(saved, 1),
                 "saving": round(1 - t / ft, 4),
                 "affords_s1": bool(saved >= a.s1_cost),
                 "ci": [round(x, 4) for x in
                        mean_ci(list(per[(tag, k)]["srr"][ii]))[1:3]]}
            print(f"{db:<28}{ncols:>5}{ntabs:>4}{len(ii):>5}{ft:>7.0f}"
                  f"{t:>9.0f}{s:>8.4f}{saved:>7.0f}{100*r['saving']:>6.1f}%"
                  f"{('是' if r['affords_s1'] else '否'):>10}")
        rows.append(r)

    # 相关性：省下的比例 vs 库规模（这是"收益是谁的函数"的直接证据）
    ok = [r for r in rows if r["reachable"]]
    if len(ok) >= 3:
        x = np.asarray([r["cols"] for r in ok], dtype=float)
        y = np.asarray([r["saving"] for r in ok], dtype=float)
        sx = np.asarray([r["full_tokens"] for r in ok], dtype=float)
        print(f"\n== 收益 vs 规模（Pearson r）==")
        print(f"  省下比例 vs 列数      r={np.corrcoef(x, y)[0,1]:+.4f}")
        print(f"  省下比例 vs full token r={np.corrcoef(sx, y)[0,1]:+.4f}")
        sv = np.asarray([r["saved_tokens"] for r in ok], dtype=float)
        print(f"  省下绝对值 vs 列数    r={np.corrcoef(x, sv)[0,1]:+.4f}")
        # 线性拟合：每多 100 列，省下比例涨多少
        A = np.vstack([x, np.ones_like(x)]).T
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        print(f"  拟合: saving ≈ {w[0]*100:+.2f}% /100列  + {w[1]:+.3f}")

        # 拐点：省下的 token 什么时候够付 System-One 的固定成本。
        # 用**绝对值**而不是比例去拟合——付账用的是绝对值。
        A2 = np.vstack([x, np.ones_like(x)]).T
        w2 = np.linalg.lstsq(A2, sv, rcond=None)[0]
        r2 = np.corrcoef(x, sv)[0, 1]
        print(f"\n== 路由拐点（省下的绝对值 ≥ System-One 的 {a.s1_cost:.0f} tok）==")
        print(f"  saved ≈ {w2[0]:+.2f} tok/列 {w2[1]:+.1f}   (r={r2:+.4f})")
        if w2[0] > 1e-6:
            c = (a.s1_cost - w2[1]) / w2[0]
            print(f"  → 交叉点 ≈ {c:.0f} 列")
            print(f"  ⚠️ BIRD-Dev 最大库只有 {int(x.max())} 列，这是**外推**，不是内插值。")
            print(f"     合成实验（scaling_lex --easy）给的交叉点是 ~230 列，"
                  f"两者差在干扰库的性质：")
            print(f"     合成拼的是**无关**库（小 k 就能守住 SRR），真实大库是**同一个**库")
            print(f"     （库越大必须留越多列）。所以真实世界的拐点应取 [230, {c:.0f}] 这个区间。")
            print(f"  → 把 System-One 的成本降到 L 时，拐点 ≈ {(a.s1_cost*0.5 - w2[1])/w2[0]:.0f} 列（L 减半）")

    path = a.out or os.path.join(RESULTS, "per_db_payoff.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"target": a.target, "grid": a.grid, "s1_cost": a.s1_cost,
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
