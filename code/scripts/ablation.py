"""消融实验（开题报告 实验 8）。

在同一批样本、同一组阈值下逐项关掉组件，用 paired bootstrap 看差异是否可信：
  * full            完整方法
  * no_fk           去掉外键闭包
  * no_table_gate   去掉表级门控（列级全打分）
  * no_route        去掉路由（全部候选保留，等价于不做过滤）
  * single_view     只用 risk 一个视角
  * multi_view      risk + need 两个视角
  * backend_swap    换开源后端（this-that / laya）

关掉一个组件掉了多少分，比"我们的方法比基线高多少"更有信息量：
前者说明这个组件到底有没有在干活。
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

from jevschema.backends import build_backend, close_backend  # noqa: E402
from jevschema.data_loader import RAW_DIR, load_spider_split, load_spider_schemas  # noqa: E402
from jevschema.fusion import FusionConfig  # noqa: E402
from jevschema.metrics import paired_bootstrap, sign_flip_test  # noqa: E402
from jevschema.pipeline import (  # noqa: E402
    PipelineConfig, RiskConfig, RouteConfig, SummaryConfig,
    evaluate_route, load_risk, save_risk, score_examples,
)

RESULTS = os.path.join(_ROOT, "results")


def variant_cfg(name: str, base: PipelineConfig) -> PipelineConfig:
    import copy

    c = copy.deepcopy(base)
    if name == "full":
        pass
    elif name == "no_fk":
        c.route.use_fk_closure = False
    elif name == "no_table_gate":
        c.risk.table_level = False
    elif name == "no_route":
        c.route.tau_lo, c.route.tau_hi = 0.0, 1.01     # 什么都不删
    elif name == "single_view":
        c.risk.views = (base.risk.views[0],)
    elif name == "multi_view":
        # 第二个视角必须也是微调见过的问法，否则等于给模型喂分布外输入
        c.risk.views = tuple(dict.fromkeys(base.risk.views + ("value",)))[:2]
    else:
        raise ValueError(name)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="laya")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-lo", type=float, default=0.10)
    ap.add_argument("--tau-hi", type=float, default=0.90)
    ap.add_argument("--variants", default="full,no_fk,no_table_gate,no_route,single_view,multi_view")
    ap.add_argument("--reuse-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--views", default="ask", help="列级问法（必须与微调时的问法一致）")
    ap.add_argument("--fusion-lexical", default="none", choices=["none", "bm25", "embedding", "both"])
    ap.add_argument("--fusion-alpha", type=float, default=0.3)
    a = ap.parse_args()

    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    # 取样口径与 compare_scorers / run_pilot 统一（shuffle=False，取前 N 条）
    examples = load_spider_split(os.path.join(RAW_DIR, "spider_val.parquet"), schemas,
                                 limit=(a.n or None), seed=a.seed, shuffle=False)
    base = PipelineConfig(
        summary=SummaryConfig(),
        risk=RiskConfig(views=tuple(a.views.split(",")), table_level=True),
        route=RouteConfig(tau_lo=a.tau_lo, tau_hi=a.tau_hi),
        fusion=FusionConfig(lexical=a.fusion_lexical, alpha=a.fusion_alpha),
    )
    backend_kw = {"model_path": a.model_path} if a.model_path else {}

    per_example: dict[str, list] = {}
    summary = {}
    for name in a.variants.split(","):
        cfg = variant_cfg(name, base)
        tag = f"{a.backend}_ab_{name}_n{a.n}"
        risk_path = os.path.join(RESULTS, f"risk_{tag}.jsonl")
        reuse = os.path.join(a.reuse_dir, f"risk_{tag}.jsonl") if a.reuse_dir else ""
        rp = reuse if (reuse and os.path.exists(reuse)) else risk_path
        if os.path.exists(rp):
            risks = load_risk(rp)
        else:
            be = build_backend(a.backend, **backend_kw)
            scored = score_examples(be, examples, cfg, every=50)
            save_risk(scored, rp)
            close_backend(be)
            risks = load_risk(rp)
        mets = []
        for ex in examples:
            if ex.ex_id not in risks:
                continue
            cols, tbls, _ = risks[ex.ex_id]
            mets.append(evaluate_route(ex, cols, tbls, cfg.route))
        per_example[name] = mets
        summary[name] = {
            "recall": round(float(np.mean([m["recall"] for m in mets])), 4),
            "table_recall": round(float(np.mean([m["table_recall"] for m in mets])), 4),
            "filter_rate": round(float(np.mean([m["filter_rate"] for m in mets])), 4),
            "precision": round(float(np.mean([m["precision"] for m in mets])), 4),
            "escalation_rate": round(float(np.mean([m["escalation_rate"] for m in mets])), 4),
            "full_recall_rate": round(float(np.mean([m["recall"] >= 0.999 for m in mets])), 4),
        }
        print(f"{name:15s} " + " ".join(f"{k}={v}" for k, v in summary[name].items()))

    # 显著性：各变体 vs full（recall 维度）
    tests = []
    if "full" in per_example:
        base_recall = [m["recall"] for m in per_example["full"]]
        base_filter = [m["filter_rate"] for m in per_example["full"]]
        for name, mets in per_example.items():
            if name == "full":
                continue
            r = paired_bootstrap(base_recall, [m["recall"] for m in mets])
            r["p_signflip"] = sign_flip_test(base_recall, [m["recall"] for m in mets])
            f = paired_bootstrap(base_filter, [m["filter_rate"] for m in mets])
            tests.append({"variant": name, "recall_diff": r["diff"], "recall_ci": [r["ci_lo"], r["ci_hi"]],
                          "p_boot": r["p_boot"], "p_signflip": r["p_signflip"],
                          "filter_diff": f["diff"]})

    out = {"backend": a.backend, "n": len(examples), "tau": [a.tau_lo, a.tau_hi],
           "summary": summary, "tests_vs_full": tests}
    os.makedirs(RESULTS, exist_ok=True)
    path = a.out or os.path.join(RESULTS, f"ablation_{a.backend}_n{a.n}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
