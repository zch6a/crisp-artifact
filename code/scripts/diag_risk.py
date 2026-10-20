"""风险问法（typed question 措辞）诊断。

System-One 模型对问法极其敏感：同一个语义，换成陈述句、换成"是否需要"、
换成"删掉会不会崩"，AUROC 能差 0.2。这一步就是在定 prompt 之前先量出来，
而不是凭感觉挑一个。

输出：每个模板在 N 个样本上的 AUROC / 正负样本均值 / 分数分布。
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

from jevschema.backends import build_backend  # noqa: E402
from jevschema.data_loader import RAW_DIR, load_spider_split, load_spider_schemas  # noqa: E402
from jevschema.metrics import auroc  # noqa: E402
from jevschema.summary import build_state, SummaryConfig  # noqa: E402

TEMPLATES = {
    "A_risk_question": (
        "If the column `{t}.{c}` were removed from this database schema, would writing the "
        "correct SQL to answer the question fail or become substantially harder?"
    ),
    "B_risk_statement": (
        "Removing the column `{t}.{c}` from the schema would make it impossible to write the "
        "correct SQL for the question."
    ),
    "C_required": (
        "The column `{t}.{c}` is required to write the correct SQL that answers the question."
    ),
    "D_asks_for": (
        "Does the question ask for information that is stored in the column `{t}.{c}`?"
    ),
    "E_needed": (
        "Answering the question requires using the column `{t}.{c}` in the SQL query."
    ),
    "F_appear": (
        "Will the column `{t}.{c}` appear in the correct SQL query that answers the question?"
    ),
    "G_without": (
        "The SQL query that answers the question cannot be written correctly without the "
        "column `{t}.{c}`."
    ),
    "H_deleted": (
        "Suppose the column `{t}.{c}` is deleted from the database. It would then be impossible "
        "to write a correct SQL query that answers the question."
    ),
    "I_implied": (
        "Is the column `{t}.{c}` mentioned or implied by the question?"
    ),
    "J_value": (
        "Does the question ask about values that are stored in the column `{t}.{c}`?"
    ),
}

COMBOS = {
    "D+A": ("D_asks_for", "A_risk_question"),
    "D+E": ("D_asks_for", "E_needed"),
    "D+H": ("D_asks_for", "H_deleted"),
    "D+E+A": ("D_asks_for", "E_needed", "A_risk_question"),
    "D+J": ("D_asks_for", "J_value"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="thisthat")
    ap.add_argument("--quant", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--templates", default="")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    examples = load_spider_split(os.path.join(RAW_DIR, "spider_val.parquet"), schemas,
                                 limit=a.n, seed=0, shuffle=True)
    kw = {"quant": a.quant} if a.quant else {}
    backend = build_backend(a.backend, **kw)

    names = a.templates.split(",") if a.templates else list(TEMPLATES)
    results = {}
    store: dict[str, np.ndarray] = {}
    for name in names:
        tmpl = TEMPLATES[name]
        p, y = [], []
        for ex in examples:
            state, tables = build_state(ex.question_block(), ex.schema, SummaryConfig())
            pairs = [(t, c) for (t, c) in ex.schema.columns() if t in set(tables)]
            texts = [tmpl.format(t=t, c=c) for t, c in pairs]
            gold = ex.gold.flat_pairs()
            ps = []
            for i in range(0, len(texts), a.batch_size):
                ps.extend(backend.judge_batch(state, texts[i:i + a.batch_size]))
            p.extend(ps)
            y.extend([1 if pr in gold else 0 for pr in pairs])
        p = np.asarray(p)
        y = np.asarray(y)
        store[name] = p.copy()
        pos, neg = p[y == 1], p[y == 0]
        results[name] = {
            "auroc": round(float(auroc(p, y)), 4),
            "pos_mean": round(float(pos.mean()), 4) if len(pos) else None,
            "neg_mean": round(float(neg.mean()), 4) if len(neg) else None,
            "gap": round(float(pos.mean() - neg.mean()), 4) if len(pos) and len(neg) else None,
            "mean": round(float(p.mean()), 4),
            "p10": round(float(np.percentile(p, 10)), 4),
            "p90": round(float(np.percentile(p, 90)), 4),
            "n_pairs": int(len(p)),
        }
        print(f"{name:18s} auroc={results[name]['auroc']:.4f} "
              f"pos={results[name]['pos_mean']} neg={results[name]['neg_mean']} "
              f"gap={results[name]['gap']} mean={results[name]['mean']}")

    # 多视角组合：同一批分数直接做 mean / max，不需要再跑模型
    combos = {}
    for cname, parts in COMBOS.items():
        if not all(q in store for q in parts):
            continue
        stack = np.vstack([store[q] for q in parts])
        for how in ("mean", "max"):
            agg = stack.mean(0) if how == "mean" else stack.max(0)
            combos[f"{cname}:{how}"] = {
                "auroc": round(float(auroc(agg, y)), 4),
                "pos_mean": round(float(agg[y == 1].mean()), 4),
                "neg_mean": round(float(agg[y == 0].mean()), 4),
            }
    for k, v in sorted(combos.items(), key=lambda kv: -kv[1]["auroc"])[:10]:
        print(f"{k:18s} auroc={v['auroc']:.4f} pos={v['pos_mean']} neg={v['neg_mean']}")
    results.update(combos)

    path = a.out or os.path.join(_ROOT, "results", f"diag_templates_{a.backend}_{a.n}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"backend": a.backend, "n": a.n, "templates": results}, f, indent=2)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
