"""Schema Scaling Benchmark（开题报告 实验 7 / 创新点 5）。

公开数据集的库太小（Spider 平均 27 列），测不出"企业级大库"的压力。
这里把多个 Spider 库拼成一个库（表名加库名前缀避免歧义），
构造 100 / 500 / 1000 / 5000 / 10000 列的合成企业库，gold SQL 同步改写。

看三件事随规模的变化：recall 掉多少、过滤率还能保持多少、延迟怎么涨。
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
from jevschema.data_loader import (  # noqa: E402
    RAW_DIR, build_scaling_benchmark, load_spider_split, load_spider_schemas,
)
from jevschema.pipeline import (  # noqa: E402
    PipelineConfig, RiskConfig, RouteConfig, SummaryConfig,
    evaluate_route, load_risk, save_risk, score_examples,
)

RESULTS = os.path.join(_ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="laya")
    ap.add_argument("--sizes", default="100,500,1000,5000")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-lo", type=float, default=0.10)
    ap.add_argument("--tau-hi", type=float, default=0.90)
    ap.add_argument("--prefilter-keep-tables", type=int, default=40)
    ap.add_argument("--out", default="")
    ap.add_argument("--views", default="ask", help="必须与微调时的问法一致")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--batch-tokens", type=int, default=16384,
                    help="每批的 token 预算；max_len 加大时必须同步调小，否则 OOM")
    ap.add_argument("--max-len", type=int, default=512,
                    help="512 时 state 只装得下 ~317 token；百列以上的库必须加大，"
                         "否则右侧被截断、尾部列根本没进模型视野")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    pool = load_spider_split(os.path.join(RAW_DIR, "spider_val.parquet"), schemas,
                             limit=(a.n or None), seed=a.seed, shuffle=False)
    print(f"base examples: {len(pool)}")

    cfg = PipelineConfig(
        summary=SummaryConfig(prefilter_keep_tables=a.prefilter_keep_tables),
        risk=RiskConfig(views=tuple(a.views.split(",")), table_level=True),
        route=RouteConfig(tau_lo=a.tau_lo, tau_hi=a.tau_hi),
    )

    rows = []
    for size in sizes:
        exs = build_scaling_benchmark(pool, schemas, target_columns=size, seed=a.seed)
        avg_cols = float(np.mean([e.schema.n_columns() for e in exs]))
        # 缓存键必须带上粗筛设置：否则换 prefilter_keep_tables 会静默复用旧分数
        # （实测踩过：关掉粗筛的那次直接复用了粗筛后的结果，数字一模一样）
        tag = (f"{a.backend}_scale{size}_n{a.n}_v{a.views}_L{a.max_len}"
               f"_pf{a.prefilter_keep_tables}")
        rp = os.path.join(RESULTS, f"risk_{tag}.jsonl")
        t0 = time.time()
        if os.path.exists(rp):
            risks = load_risk(rp)
            scoring_s = None
        else:
            kw = {"model_path": a.model_path} if a.model_path else {}
            kw["max_len"] = a.max_len
            # 注意力是 O(L²)：max_len 4096 时默认 batch_tokens=16384 会直接 OOM
            # （实测要分配 4 GiB 失败），长序列必须把批按 token 数调小。
            kw["batch_tokens"] = a.batch_tokens
            be = build_backend(a.backend, **kw)
            scored = score_examples(be, exs, cfg, every=25)
            save_risk(scored, rp)
            close_backend(be)
            risks = load_risk(rp)
            scoring_s = time.time() - t0
        mets = []
        for ex in exs:
            if ex.ex_id not in risks:
                continue
            cols, tbls, _ = risks[ex.ex_id]
            mets.append(evaluate_route(ex, cols, tbls, cfg.route))
        row = {
            "target_columns": size,
            "actual_columns": round(avg_cols, 1),
            "tables": round(float(np.mean([e.schema.n_tables() for e in exs])), 1),
            "recall": round(float(np.mean([m["recall"] for m in mets])), 4),
            "table_recall": round(float(np.mean([m["table_recall"] for m in mets])), 4),
            "filter_rate": round(float(np.mean([m["filter_rate"] for m in mets])), 4),
            "precision": round(float(np.mean([m["precision"] for m in mets])), 4),
            "kept_columns": round(float(np.mean([m["kept_columns"] for m in mets])), 1),
            "scoring_s_total": round(scoring_s, 1) if scoring_s else None,
            "scoring_ms_per_example": round(1000 * scoring_s / max(1, len(exs)), 1) if scoring_s else None,
        }
        rows.append(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()))

    path = a.out or os.path.join(RESULTS, f"scaling_{a.backend}_n{a.n}.json")
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"backend": a.backend, "n": a.n, "tau": [a.tau_lo, a.tau_hi], "rows": rows},
                  f, indent=2)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
