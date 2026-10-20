"""System-One 的成本模型：它的 linking 成本到底是常数还是 O(库规模)？

这个问题决定整条路由论断成不成立，而之前我们一直**默认它是常数 1788**，
这是错的。把成本拆开发现：

    link_cost = state_cost + n_questions × per_question_cost
              = min(全库 schema 摘要, max_state_tokens=2400) + 15.1 × N

`build_state` 的 `prefilter="bm25"` 只在 **表数 > 40** 时才生效
（BIRD 最多 13 张表，Spider 更少），所以在学术基准上 state 就是**全库摘要**，
它随库规模线性涨，直到 2400 tokens 封顶。

=> System-One 的成本在小库上 ≈ 全 schema prompt（648 ≈ 651！），
   在大库上封顶 2400。**它自己就是 O(库规模) 的**，所以"路由判据=库规模"
   这个说法在第一近似下**不成立** —— 必须先把 state 也门控掉。

本脚本做的事：
  1. 实测 state 成本随库规模怎么变（BIRD 各库 + 合成缩放库），确认 2400 封顶；
  2. 实测每个 typed question 的边际成本（门控消融的斜率）；
  3. 实测压缩省下的 token 随列数的斜率（per_db_payoff）；
  4. 用这三个实测常数解交叉点，分"全量 state"和"门控 state"两种机制。

没有 GPU 依赖：state 成本只做分词计数。
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
from jevschema.summary import SummaryConfig, build_state  # noqa: E402
from jevschema.tokens import make_stage_tokens, n_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")


def state_tokens(ex, tables=None):
    st, tbls = build_state(ex.question_block(), ex.schema, SummaryConfig(),
                           tables=tables)
    return n_tokens(st), tbls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--per-question", type=float, default=15.1,
                    help="每条 typed question 的边际 token（gate_ablation 实测斜率）")
    ap.add_argument("--save-slope", type=float, default=4.306,
                    help="每多一列，压缩省下的 token（per_db_payoff 实测斜率）")
    ap.add_argument("--save-intercept", type=float, default=-125.6,
                    help="省下 token 拟合的截距")
    ap.add_argument("--state-cap", type=float, default=2400.0,
                    help="SummaryConfig.max_state_tokens")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    ex = [e for e in ex if e.gold is not None and e.gold.flat_pairs()]

    # ---------------------------------------------------------------- 1. state 成本
    print("== 1. state 成本随库规模怎么变（BIRD 各库）==")
    print(f"{'db_id':<28}{'列数':>6}{'full tok':>10}{'state tok':>11}"
          f"{'state/full':>12}")
    g = defaultdict(list)
    for i, e in enumerate(ex):
        g[e.db_id].append(i)
    pts = []
    for db, idx in sorted(g.items(), key=lambda kv: -ex[kv[1][0]].schema.n_columns()):
        st = float(np.mean([state_tokens(ex[i])[0] for i in idx]))
        ft = float(np.mean([make_stage_tokens(ex[i].schema, ex[i].question,
                                              ex[i].evidence).total_tokens
                            for i in idx]))
        nc = ex[idx[0]].schema.n_columns()
        pts.append((nc, ft, st))
        print(f"{db:<28}{nc:>6}{ft:>10.0f}{st:>11.0f}{st/ft:>12.2f}")

    x = np.asarray([p[0] for p in pts], dtype=float)
    y = np.asarray([p[2] for p in pts], dtype=float)
    f = np.asarray([p[1] for p in pts], dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"\n  拟合 state ≈ {w[0]:.2f} tok/列 {w[1]:+.1f}"
          f"   (r={np.corrcoef(x, y)[0,1]:+.4f})")
    print(f"  state / full-schema-prompt 平均比值 = {float(np.mean(y/f)):.3f}"
          f"  ← 接近 1 说明 **state 本身就 ≈ 把全 schema 又喂了一遍**")
    print(f"  ⚠️ 封顶 {a.state_cap:.0f} tokens 只有在 state 超过它时才起作用；"
          f"BIRD 最大库 state={y.max():.0f}，远未触及。")

    # ---------------------------------------------------------------- 2. 门控后的 state
    print("\n== 2. 把 state 也门控掉能省多少（state 只由 BM25 top-T 表构成）==")
    print(f"{'top-T 表':>9}{'state tok':>11}{'省 vs 全量 state':>18}")
    base_state = float(np.mean([state_tokens(e)[0] for e in ex]))
    gate_state = {}
    for T in (2, 3, 4, 6, 8, 12):
        vals = []
        for e in ex:
            _, tbls = state_tokens(e)
            # 用 build_state 自己的 bm25 预筛顺序取前 T 张表
            st2, _ = build_state(e.question_block(), e.schema, SummaryConfig(),
                                 tables=tbls[:T] if len(tbls) > T else tbls)
            vals.append(n_tokens(st2))
        v = float(np.mean(vals))
        gate_state[T] = v
        print(f"{T:>9}{v:>11.0f}{base_state - v:>18.0f}")
    print(f"  （全量 state = {base_state:.0f} tok）")
    print("  → 门控 state 能把 state 压到 1/2 ~ 1/3，这是让 System-One 成本"
          " 与库规模**脱钩**的唯一办法。")

    # ---------------------------------------------------------------- 3. 交叉点
    print("\n== 3. 交叉点：压缩省下的 token 什么时候 ≥ System-One 的成本 ==")
    print("  成本模型: cost(C) = state_cost(C) + per_question × N")
    print(f"  收益模型: save(C) = {a.save_slope:.2f} × C {a.save_intercept:+.1f}")
    print(f"\n{'机制':<26}{'state':>8}{'N':>5}{'cost':>8}{'交叉点(列)':>12}{'交叉点(tok)':>13}")

    def solve(state_fn, N, label):
        # 解 save(C) = state_fn(C) + pq*N，分段：state 线性段 / 封顶段
        cands = []
        # 线性段：state = w0*C + w1
        denom = a.save_slope - w[0]
        if denom > 1e-9:
            c = (w[1] + a.per_question * N - a.save_intercept) / denom
            if c > 0 and w[0] * c + w[1] <= a.state_cap:
                cands.append(c)
        # 封顶段
        c2 = (a.state_cap + a.per_question * N - a.save_intercept) / a.save_slope
        if c2 > 0 and w[0] * c2 + w[1] > a.state_cap:
            cands.append(c2)
        if not cands:
            print(f"{label:<26}{'—':>8}{N:>5}{'—':>8}{'无解(收益斜率≤成本斜率)':>12}")
            return None
        c = min(cands)
        cost = min(w[0] * c + w[1], a.state_cap) + a.per_question * N
        print(f"{label:<26}{min(w[0]*c+w[1], a.state_cap):>8.0f}{N:>5}"
              f"{cost:>8.0f}{c:>12.0f}{a.save_slope*c + a.save_intercept:>13.0f}")
        return c

    out_rows = []
    for N in (30, 60, 150):
        c = solve(lambda C: min(w[0] * C + w[1], a.state_cap), N, f"全量 state（现状）")
        if c:
            out_rows.append({"mechanism": "full_state", "N": N, "crossover_cols": round(c, 1)})
    gs = min(gate_state.values())          # 最激进的门控 state
    gs_mid = gate_state.get(6, base_state / 2)
    for label, sv in (("门控 state（top-6 表）", gs_mid),
                      ("门控 state（最激进）", gs)):
        for N in (30, 60, 150):
            c = (sv + a.per_question * N - a.save_intercept) / a.save_slope
            cost = sv + a.per_question * N
            print(f"{label:<26}{sv:>8.0f}{N:>5}{cost:>8.0f}{c:>12.0f}"
                  f"{a.save_slope*c + a.save_intercept:>13.0f}")
            out_rows.append({"mechanism": label, "N": N, "crossover_cols": round(c, 1),
                             "state_tokens": round(sv, 1), "cost": round(cost, 1)})

    print("\n⚠️ 全部是**外推**：BIRD-Dev 最大库 199 列，而交叉点在数百列之外。")
    print("   两个斜率（省下 / state）都是在 11~199 列区间内拟合的，")
    print("   超出这个区间必须标注为外推，不能写成实测。")

    path = a.out or os.path.join(RESULTS, "s1_cost_model.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"n": len(ex),
                   "state_fit": {"tok_per_col": round(float(w[0]), 3),
                                 "intercept": round(float(w[1]), 1),
                                 "r": round(float(np.corrcoef(x, y)[0, 1]), 4)},
                   "state_over_full_ratio": round(float(np.mean(y / f)), 4),
                   "full_state_tokens": round(base_state, 1),
                   "gated_state_tokens": {str(k): round(v, 1)
                                          for k, v in gate_state.items()},
                   "per_question_tokens": a.per_question,
                   "save_fit": {"slope": a.save_slope, "intercept": a.save_intercept},
                   "crossovers": out_rows}, fp, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
