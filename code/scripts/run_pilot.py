"""Pilot 实验（开题报告 §6.6）：Spider N 题，System-One 风险估计 + 路由。

用法（$0 成本，本地 this-that）：
    python scripts/run_pilot.py --backend thisthat --n 100 --views risk --sweep

分两阶段：先打分存盘（GPU 时间的大头），再在同一批分数上扫阈值。
重跑时加 --reuse-risk 直接跳过打分阶段。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.backends import build_backend, close_backend  # noqa: E402
from jevschema.data_loader import load_spider_split, load_spider_schemas, RAW_DIR  # noqa: E402
from jevschema.fusion import FusionConfig  # noqa: E402
from jevschema.metrics import auroc, brier, ece, mean_ci  # noqa: E402
from jevschema.refine import build_refiner  # noqa: E402
from jevschema.pipeline import (  # noqa: E402
    PipelineConfig, RiskConfig, RouteConfig, SummaryConfig,
    aggregate, calibration_pairs, evaluate_route, load_risk, save_risk, score_examples,
    sweep_thresholds,
)
from jevschema.cost import backend_cost, estimate_jev_cost  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")


def tag_of(a: argparse.Namespace) -> str:
    views = "-".join(a.views.split(","))
    mp = "ft" if a.model_path else "base"
    return (f"{a.backend}{'-' + mp}_{'dev' if a.split == 'dev' else 'train'}{a.n}_v{views}"
            f"_tl{int(a.table_level)}_f{a.fusion_lexical}{a.fusion_alpha}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="thisthat")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", default="risk", help="逗号分隔: risk,need,fk")
    ap.add_argument("--table-level", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-state-tokens", type=int, default=2400)
    ap.add_argument("--tau-lo", type=float, default=0.10)
    ap.add_argument("--tau-hi", type=float, default=0.90)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fk", type=int, default=1)
    ap.add_argument("--escalate", default="keep", choices=["keep", "drop", "llm"],
                    help="不确定带的处置：keep=直接保留（默认，最保守）/ drop=直接删 / "
                         "llm=交给 System-Two 精筛（配合 --refiner）")
    ap.add_argument("--reuse-risk", default="")
    ap.add_argument("--fusion-lexical", default="none", choices=["none", "bm25", "embedding", "both"])
    ap.add_argument("--fusion-alpha", type=float, default=0.3)
    ap.add_argument("--fusion-normalize", default="max", choices=["max", "rank", "minmax", "none"])
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--quant", default="")
    ap.add_argument("--model-path", default="", help="覆盖默认权重路径（微调模型用）")
    ap.add_argument("--keep-whole-tables", type=int, default=0)
    ap.add_argument("--table-select", default="none", choices=["none", "topk", "frac"])
    ap.add_argument("--top-k-tables", type=int, default=2)
    ap.add_argument("--table-frac", type=float, default=0.3)
    ap.add_argument("--refiner", default="keep-all",
                    choices=["keep-all", "drop-all", "embedding", "llm", "ollama"])
    ap.add_argument("--refiner-model", default="qwen2.5-coder:7b")
    ap.add_argument("--refiner-endpoint", default="http://localhost:11434")
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tag = tag_of(a)
    risk_path = a.reuse_risk or os.path.join(RESULTS, f"risk_{tag}.jsonl")

    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    parquet = os.path.join(RAW_DIR, "spider_val.parquet" if a.split == "dev" else "spider_train.parquet")
    # 取样口径与 compare_scorers.py / run_baselines.py 统一（shuffle=False，取前 N 条）
    examples = load_spider_split(parquet, schemas, limit=(a.n or None), seed=a.seed,
                                 shuffle=False)
    print(f"examples: {len(examples)} (split={a.split}, n={a.n}, seed={a.seed})")

    cfg = PipelineConfig(
        summary=SummaryConfig(max_state_tokens=a.max_state_tokens),
        risk=RiskConfig(views=tuple(a.views.split(",")), table_level=bool(a.table_level),
                        batch_size=a.batch_size),
        route=RouteConfig(tau_lo=a.tau_lo, tau_hi=a.tau_hi, use_fk_closure=bool(a.fk),
                          escalate=a.escalate,
                          keep_whole_tables=bool(a.keep_whole_tables),
                          table_select=a.table_select, top_k_tables=a.top_k_tables,
                          table_frac=a.table_frac),
        fusion=FusionConfig(lexical=a.fusion_lexical, alpha=a.fusion_alpha,
                            normalize=a.fusion_normalize),
    )

    # ---------------------------------------------------------------- 阶段 1：打分
    if a.reuse_risk and os.path.exists(a.reuse_risk):
        risks = load_risk(a.reuse_risk)
        print(f"复用已打分结果: {a.reuse_risk} ({len(risks)} 条)")
        backend = None
    else:
        kw = {}
        if a.device:
            kw["device"] = a.device
        if a.quant:
            kw["quant"] = a.quant
        if a.model_path:
            kw["model_path"] = a.model_path
        backend = build_backend(a.backend, **kw)
        print(f"backend: {getattr(backend, 'name', '?')}")
        t0 = time.time()
        scored = score_examples(backend, examples, cfg, every=max(1, len(examples) // 10))
        print(f"打分耗时 {time.time()-t0:.1f}s")
        save_risk(scored, risk_path)
        print(f"风险分数已保存: {risk_path}")
        risks = load_risk(risk_path)
        close_backend(backend)

    # ---------------------------------------------------------------- 阶段 2：路由/扫阈值
    rows = []
    if a.sweep:
        grid_lo = [round(x, 2) for x in np.arange(0.05, 0.85, 0.05)]
        grid_hi = [round(x, 2) for x in np.arange(0.15, 1.00, 0.05)]
        rows = sweep_thresholds(examples, risks, cfg.route, grid_lo, grid_hi)
        rows.sort(key=lambda r: (-r["recall"], -r["filter_rate"]))
        print("\n== 阈值扫描（按 recall 降序，前 15）==")
        print(f"{'tau_lo':>7} {'tau_hi':>7} {'recall':>7} {'tbl_rec':>8} {'filter':>7} "
              f"{'prec':>6} {'escal':>6} {'full':>6}")
        for r in rows[:15]:
            print(f"{r['tau_lo']:7.2f} {r['tau_hi']:7.2f} {r['recall']:7.3f} "
                  f"{r['table_recall']:8.3f} {r['filter_rate']:7.3f} {r['precision']:6.3f} "
                  f"{r['escalation_rate']:6.3f} {r['full_recall_rate']:6.3f}")
        # 过滤率 >= 0.7 里 recall 最高的一组，作为"达标点"
        ok = [r for r in rows if r["filter_rate"] >= 0.70]
        print("\n过滤率 >= 70% 中 recall 最高：",
              max(ok, key=lambda r: r["recall"]) if ok else "无（未达标）")

    # ---------------------------------------------------------------- 单点评估
    refiner = build_refiner(a.refiner, model=a.refiner_model,
                            endpoint=a.refiner_endpoint) if a.refiner != "keep-all" else None
    per_example = []
    for ex in examples:
        if ex.ex_id not in risks:
            continue
        cols, tbls, _ = risks[ex.ex_id]
        per_example.append(evaluate_route(ex, cols, tbls, cfg.route, refiner))
    agg = {
        k: round(float(np.mean([m[k] for m in per_example])), 4)
        for k in ("recall", "table_recall", "filter_rate", "precision",
                  "escalation_rate", "compression_per_gold")
    }
    recall_m, lo, hi = mean_ci([m["recall"] for m in per_example])
    filt_m, flo, fhi = mean_ci([m["filter_rate"] for m in per_example])
    agg["recall_ci"] = [round(lo, 4), round(hi, 4)]
    agg["filter_rate_ci"] = [round(flo, 4), round(fhi, 4)]
    agg["full_recall_rate"] = round(float(np.mean([m["recall"] >= 0.999 for m in per_example])), 4)
    agg["n_examples"] = len(per_example)

    # ---------------------------------------------------------------- 校准
    p, y = [], []
    for ex in examples:
        if ex.ex_id not in risks:
            continue
        cols, _t, _ = risks[ex.ex_id]
        gold_flat = ex.gold.flat_pairs()
        for pair, r in cols.items():
            p.append(r)
            y.append(1 if pair in gold_flat else 0)
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    ece_u, rel_u = ece(p, y, 10, "uniform")
    ece_q, _rel_q = ece(p, y, 10, "quantile")
    cal = {"n_pairs": int(len(p)), "pos_rate": round(float(y.mean()), 4),
           "ece_uniform": round(ece_u, 4), "ece_quantile": round(ece_q, 4),
           "brier": round(brier(p, y), 4), "auroc": round(float(auroc(p, y)), 4),
           "reliability": rel_u}

    # ---------------------------------------------------------------- 成本
    cost = {}
    if backend is not None:
        cost = backend_cost(backend, len(examples)).as_dict()
    # System-Two 的开销必须单独记：add_refiner_cost 只认 refiner.cost，
    # 而 OllamaRefiner 暴露的是 calls / latency_s，之前一直被记成 0。
    if refiner is not None:
        cost["llm_refine_calls"] = int(getattr(refiner, "calls", 0) or
                                       getattr(getattr(refiner, "cost", None), "calls", 0))
        cost["llm_refine_usd"] = round(float(getattr(refiner, "usd", 0.0) or
                                             getattr(getattr(refiner, "cost", None), "usd", 0.0)), 6)
        cost["llm_refine_latency_s"] = round(
            float(getattr(refiner, "latency_s", 0.0) or
                  getattr(getattr(refiner, "cost", None), "latency_s", 0.0)), 3)
    # 同一批判断如果改走 Jev API 要多少钱（口径：state 只付一次，questions 累加）
    state_chars = []
    with open(risk_path, "r", encoding="utf-8") as f:
        for line in f:
            state_chars.append(json.loads(line).get("state_chars", 6000))
    avg_state_tokens = int(np.mean(state_chars) / 4) if state_chars else 0
    avg_questions = int(np.mean([len(v[0]) for v in risks.values()])) if risks else 0
    cost["jev_equiv_usd"] = round(estimate_jev_cost(
        len(examples), avg_state_tokens, avg_questions, avg_question_tokens=40), 6)
    cost["avg_state_tokens"] = avg_state_tokens
    cost["avg_questions_per_example"] = avg_questions

    out = {"tag": tag, "config": {
        "backend": a.backend, "n": a.n, "split": a.split, "seed": a.seed,
        "views": a.views.split(","), "table_level": bool(a.table_level),
        "tau_lo": a.tau_lo, "tau_hi": a.tau_hi, "fk_closure": bool(a.fk),
        "escalate": a.escalate, "refiner": a.refiner,
        "refiner_model": (a.refiner_model if a.refiner != "keep-all" else ""),
    }, "metrics": agg, "calibration": cal, "cost": cost,
        "sweep": rows[:200], "risk_path": risk_path}
    path = a.out or os.path.join(RESULTS, f"pilot_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n== 单点结果 (tau_lo=%.2f, tau_hi=%.2f) ==" % (a.tau_lo, a.tau_hi))
    for k, v in agg.items():
        print(f"  {k:22s} {v}")
    print("\n== 校准 ==")
    for k in ("n_pairs", "pos_rate", "ece_uniform", "ece_quantile", "brier", "auroc"):
        print(f"  {k:22s} {cal[k]}")
    print("\n== 成本 ==")
    for k, v in cost.items():
        print(f"  {k:22s} {v}")
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
