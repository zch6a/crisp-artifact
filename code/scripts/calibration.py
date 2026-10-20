"""校准分析（开题报告 实验 1 / §5.7）。

输入 run_pilot.py 产出的 risk jsonl（只打分那一步的产物），输出：
  * reliability diagram 数据 + ECE（uniform / quantile 两种分箱）
  * Brier、AUROC
  * 按"允许丢多少 gold"反解的 tau_lo / tau_hi
  * isotonic / platt 重标定前后的 ECE 对比（在 held-out 上评，不在 fit 集上评）

为什么 fit / eval 必须分开：在同一批数据上 fit isotonic 再算 ECE 会系统性低估，
报出来的 ECE 是自欺。
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

from jevschema.data_loader import RAW_DIR, load_spider_split, load_spider_schemas  # noqa: E402
from jevschema.metrics import auroc, brier, ece  # noqa: E402
from jevschema.pipeline import load_risk  # noqa: E402
from jevschema.router import ProbabilityCalibrator, pick_tau_hi, pick_tau_lo  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")


def collect(examples, risks):
    p, y, ex_id = [], [], []
    for ex in examples:
        if ex.ex_id not in risks:
            continue
        cols, _t, _ = risks[ex.ex_id]
        gold = ex.gold.flat_pairs()
        for pair, r in cols.items():
            p.append(r)
            y.append(1 if pair in gold else 0)
            ex_id.append(ex.ex_id)
    return np.asarray(p), np.asarray(y), np.asarray(ex_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", required=True, help="run_pilot.py 产出的 risk jsonl")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-fraction", type=float, default=0.3)
    ap.add_argument("--target-gold-loss", type=float, default=0.01)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    risks = load_risk(a.risk)
    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    parquet = os.path.join(RAW_DIR, "spider_val.parquet" if a.split == "dev" else "spider_train.parquet")
    examples = load_spider_split(parquet, schemas, limit=(a.n or None), seed=a.seed,
                                 shuffle=False)
    p, y, ex_id = collect(examples, risks)
    print(f"pairs: {len(p)}, positive rate: {y.mean():.4f}")

    # 按 example 划分 fit / eval，避免同一 example 的列被切开（同例内分数相关）
    rng = np.random.default_rng(0)
    ids = np.unique(ex_id)
    rng.shuffle(ids)
    n_fit = int(len(ids) * a.fit_fraction)
    fit_ids = set(ids[:n_fit])
    m_fit = np.array([e in fit_ids for e in ex_id])
    m_eval = ~m_fit

    res = {
        "n_pairs": int(len(p)), "pos_rate": round(float(y.mean()), 4),
        "n_fit_pairs": int(m_fit.sum()), "n_eval_pairs": int(m_eval.sum()),
    }
    for tag, m in (("all", np.ones_like(m_fit, dtype=bool)), ("eval", m_eval)):
        eu, rel = ece(p[m], y[m], 10, "uniform")
        eq, _ = ece(p[m], y[m], 10, "quantile")
        res[f"ece_uniform_{tag}"] = round(float(eu), 4)
        res[f"ece_quantile_{tag}"] = round(float(eq), 4)
        res[f"brier_{tag}"] = round(brier(p[m], y[m]), 4)
        res[f"auroc_{tag}"] = round(float(auroc(p[m], y[m])), 4)
        if tag == "eval":
            res["reliability"] = rel

    # 阈值：在 fit 上定，在 eval 上验
    tau_lo, loss = pick_tau_lo(p[m_fit], y[m_fit], max_gold_loss=a.target_gold_loss)
    tau_hi, prec = pick_tau_hi(p[m_fit], y[m_fit])
    eval_loss = float(y[m_eval][p[m_eval] < tau_lo].sum()) / max(1, int(y[m_eval].sum()))
    m_hi = p[m_eval] > tau_hi
    eval_prec = float(y[m_eval][m_hi].mean()) if m_hi.sum() else float("nan")
    res["tau_lo"] = tau_lo
    res["tau_lo_fit_gold_loss"] = round(loss, 4)
    res["tau_lo_eval_gold_loss"] = round(eval_loss, 4)
    res["tau_hi"] = tau_hi
    res["tau_hi_fit_precision"] = round(prec, 4)
    res["tau_hi_eval_precision"] = round(eval_prec, 4)
    res["eval_escalation_rate"] = round(float(((p[m_eval] >= tau_lo) & (p[m_eval] <= tau_hi)).mean()), 4)

    # 重标定：fit 上训练，eval 上报 ECE
    for method in ("isotonic", "platt"):
        cal = ProbabilityCalibrator(method).fit(p[m_fit], y[m_fit])
        q = cal.apply(p[m_eval])
        eu, _ = ece(q, y[m_eval], 10, "uniform")
        eq, _ = ece(q, y[m_eval], 10, "quantile")
        res[f"ece_after_{method}"] = round(float(eu), 4)
        res[f"ece_after_{method}_quantile"] = round(float(eq), 4)
        res[f"brier_after_{method}"] = round(brier(q, y[m_eval]), 4)

    path = a.out or os.path.join(RESULTS, f"calibration_{os.path.basename(a.risk).replace('.jsonl','')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in res.items() if k != "reliability"}, indent=2))
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
