"""BIRD 上的阈值路由诊断：验证"绝对概率阈值"到底还能不能用。

背景（必须记录，否则会重复踩坑）：
    Spider 上拟合的 τ=0.1/0.9 三路分流，搬到 BIRD-Dev 后**完全变形** ——
    模型在跨域数据上整体不自信，p50 只有 0.0065，p<0.1 的列占 87.6%，
    于是"删掉低风险列"变成了"删掉 85% 的列"，SRR 掉到 0.5。
    同一个打分器改成按名次取 top-k 列，SRR 立刻回到 0.881。

所以这个脚本回答三个问题：
    1. 概率分布长什么样（是不是被压向 0）？
    2. 标定指标（ECE / Brier）和"阈值路由后的 SRR"是否一致？
       —— 实测不一致：ECE 可以被"绝大多数列是负例"这一片稀释掉。
    3. 域内微调 + 重拟合 temperature 之后，绝对阈值能恢复到什么程度？

用法：先跑过 eval_bird.py（分数会按 tag 缓存到 results/bird_cache/），再跑本脚本。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import auroc, brier, ece, strict_recall  # noqa: E402
from jevschema.selection import expand_kept  # noqa: E402

CACHE = os.path.join(_ROOT, "results", "bird_cache")
TAUS = [(0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.05, 0.50),
        (0.02, 0.30), (0.01, 0.20), (0.05, 0.95)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="eval_bird.py 用的分数缓存 tag（不含 .jsonl）")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--expand", default="fk", choices=["none", "fk", "table", "fk+table"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    path = os.path.join(CACHE, a.tag + ".jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"找不到分数缓存: {path}\n（先跑 eval_bird.py，分数会按 tag 缓存）")
    store = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}

    schemas = load_bird_schemas()
    examples = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    if a.stride > 1:
        examples = examples[::a.stride]
    examples = [e for e in examples if e.ex_id in store]
    print(f"tag={a.tag}  n={len(examples)}  expand={a.expand}")

    # ---------------------------------------------------------------- 分布 + 标定
    P, Y = [], []
    for ex in examples:
        sc = store[ex.ex_id]
        g = ex.gold.flat_pairs()
        for k, v in sc.items():
            P.append(v)
            Y.append(1 if k in g else 0)
    P, Y = np.asarray(P), np.asarray(Y)
    print(f"\n== 概率分布（n_pairs={len(P)}）==")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  分位 " + "  ".join(f"p{q}={np.percentile(P, q):.4f}" for q in qs))
    print(f"  mean={P.mean():.4f} std={P.std():.4f}  正例率={Y.mean():.4f}  "
          f"gold列平均预测={P[Y == 1].mean():.4f}")
    print(f"  p<0.1 占比={(P < 0.1).mean():.4f}   p>0.9 占比={(P > 0.9).mean():.4f}   "
          f"0.1<=p<=0.9 占比={((P >= 0.1) & (P <= 0.9)).mean():.4f}")
    print("\n== 标定（注意：ECE 会被「绝大多数是负例」稀释，别单独拿它当门限）==")
    print(f"  ECE(uniform,10) = {ece(P, Y, 10, 'uniform')[0]:.4f}")
    print(f"  ECE(quantile,10)= {ece(P, Y, 10, 'quantile')[0]:.4f}")
    print(f"  Brier           = {brier(P, Y):.4f}")
    print(f"  AUROC(pooled)   = {auroc(P, Y):.4f}")

    # ---------------------------------------------------------------- 阈值路由
    print(f"\n== 绝对阈值三路分流（escalate=keep，即中间带保守保留）expand={a.expand} ==")
    print(f"{'τ_lo':>6}{'τ_hi':>6}{'SRR':>9}{'recall':>9}{'过滤率':>9}{'不确定带':>9}{'C̄':>8}")
    rows = []
    for lo, hi in TAUS:
        srrs, recs, filts, bands, kepts = [], [], [], [], []
        for ex in examples:
            sc = store[ex.ex_id]
            pairs = list(sc.keys())
            p = np.asarray([sc[k] for k in pairs], dtype=float)
            kept = [pairs[i] for i in range(len(pairs)) if p[i] >= lo]
            bands.append(float(((p >= lo) & (p <= hi)).mean()))
            kept = expand_kept(ex.schema, kept, a.expand)
            srrs.append(strict_recall(ex.gold, kept))
            recs.append(ex.gold.recall({tuple(x) for x in kept}))
            kepts.append(len(kept))
            filts.append(1 - len(kept) / ex.schema.n_columns())
        r = {"tau_lo": lo, "tau_hi": hi, "srr": float(np.mean(srrs)),
             "recall": float(np.mean(recs)), "filter_rate": float(np.mean(filts)),
             "uncertain_band": float(np.mean(bands)), "avg_kept": float(np.mean(kepts))}
        rows.append(r)
        print(f"{lo:>6.2f}{hi:>6.2f}{r['srr']:>9.4f}{r['recall']:>9.4f}"
              f"{r['filter_rate']:>9.4f}{r['uncertain_band']:>9.4f}{r['avg_kept']:>8.1f}")

    out = {"tag": a.tag, "n": len(examples), "expand": a.expand,
           "dist": {"p50": float(np.percentile(P, 50)), "p90": float(np.percentile(P, 90)),
                    "mean": float(P.mean()), "pos_rate": float(Y.mean()),
                    "frac_p_lt_0.1": float((P < 0.1).mean()),
                    "frac_p_gt_0.9": float((P > 0.9).mean())},
           "calib": {"ece_uniform": ece(P, Y, 10, "uniform")[0],
                     "ece_quantile": ece(P, Y, 10, "quantile")[0],
                     "brier": brier(P, Y), "auroc": float(auroc(P, Y))},
           "thresholds": rows}
    path = a.out or os.path.join(_ROOT, "results", f"bird_thresholds_{a.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
