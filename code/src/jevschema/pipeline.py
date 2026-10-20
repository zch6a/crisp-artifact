"""端到端流水线：question -> System-One 风险 -> 路由 -> 闭包 -> 精筛 -> 选中的 schema。

这一层不发明任何算法，只把 summary / risk / router / refine 串起来，
并把每一步的可观测数据（分数、标签、延迟、token）留在 record 里，
供 metrics 与校准脚本复用。实验脚本只跟 pipeline 打交道。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .backends.base import DecisionBackend
from .data_loader import Example
from .fusion import FusionConfig, fuse
from .metrics import selection_metrics
from .refine import Refiner, build_refiner
from .risk import RiskConfig, RiskOutput, estimate
from .router import RouteConfig, RouteResult, route
from .summary import SummaryConfig, build_state

Pair = Tuple[str, str]


@dataclass
class PipelineConfig:
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    refiner: str = "keep-all"
    refiner_kwargs: Dict = field(default_factory=dict)
    calibrator: Optional[object] = None      # ProbabilityCalibrator
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class Record:
    ex_id: str
    db_id: str
    metrics: Dict[str, float]
    kept: List[Pair] = field(default_factory=list)
    dropped: List[Pair] = field(default_factory=list)
    escalated: List[Pair] = field(default_factory=list)
    closure_added: List[Pair] = field(default_factory=list)
    n_columns: int = 0
    n_gold: int = 0
    state_tokens: int = 0
    latency_s: float = 0.0
    n_questions: int = 0

    # 校准数据：(table, column, risk, 是否 gold)
    pairs: List[Tuple[str, str, float, int]] = field(default_factory=list)


class JevSchemaPipeline:
    def __init__(self, backend: DecisionBackend, cfg: Optional[PipelineConfig] = None,
                 refiner: Optional[Refiner] = None):
        self.backend = backend
        self.cfg = cfg or PipelineConfig()
        self.refiner = refiner or build_refiner(self.cfg.refiner, **self.cfg.refiner_kwargs)

    # ---------------------------------------------------------------- 单条
    def run_example(self, ex: Example) -> Record:
        t0 = time.time()
        state, tables = build_state(ex.question_block(), ex.schema, self.cfg.summary)
        risk_out = estimate(self.backend, state, ex.schema, tables, self.cfg.risk)

        if self.cfg.calibrator is not None:
            keys = list(risk_out.col_risk.keys())
            if keys:
                cal = self.cfg.calibrator.apply([risk_out.col_risk[k] for k in keys])
                risk_out.col_risk = {k: float(v) for k, v in zip(keys, cal)}

        res = route(risk_out, ex.schema, self.cfg.route)

        # 精筛：只处理"不确定"的候选
        refined = self.refiner.refine(ex, sorted(res.escalated)) if res.escalated else set()
        res.kept |= refined
        if refined:
            for t, _c in refined:
                res.kept_tables.add(t)
        final_kept = res.all_kept()

        gold = ex.gold
        mets = selection_metrics(final_kept, res.kept_tables, ex.schema, gold,
                                 n_escalated=len(res.escalated))
        gold_flat = gold.flat_pairs() if gold else set()
        pairs = [(t, c, float(risk_out.col_risk.get((t, c), 0.0)),
                  1 if (t, c) in gold_flat else 0) for (t, c) in risk_out.col_risk]

        return Record(
            ex_id=ex.ex_id, db_id=ex.db_id, metrics=mets.as_dict(),
            kept=sorted(final_kept), dropped=sorted(res.dropped)[:5000],
            escalated=sorted(res.escalated), closure_added=sorted(res.closure_added),
            n_columns=ex.schema.n_columns(), n_gold=len(gold.requirements) if gold else 0,
            state_tokens=self.backend.n_tokens(state), latency_s=time.time() - t0,
            n_questions=risk_out.n_questions, pairs=pairs,
        )

    # ---------------------------------------------------------------- 批量
    def run(self, examples: Sequence[Example], verbose: bool = True,
            every: int = 10) -> List[Record]:
        out: List[Record] = []
        for i, ex in enumerate(examples):
            rec = self.run_example(ex)
            out.append(rec)
            if verbose and (i + 1) % every == 0:
                m = rec.metrics
                print(f"[{i+1}/{len(examples)}] recall={m['recall']:.3f} "
                      f"filter={m['filter_rate']:.3f} kept={m['kept_columns']}/{m['total_columns']} "
                      f"lat={rec.latency_s:.2f}s", flush=True)
        return out


# ----------------------------------------------------------------------------- 打分/复用
def score_examples(backend: DecisionBackend, examples: Sequence[Example],
                   cfg: PipelineConfig, verbose: bool = True,
                   every: int = 10) -> List[Tuple[Example, RiskOutput]]:
    """只做 System-One 打分，不做路由。

    单独拆出来是因为阈值要在同一批分数上扫很多组：打分一次几十分钟，
    扫阈值几毫秒。把两件事绑在一起跑会被 GPU 时间逼着只试一组阈值。
    """
    out: List[Tuple[Example, RiskOutput]] = []
    for i, ex in enumerate(examples):
        state, tables = build_state(ex.question_block(), ex.schema, cfg.summary)
        r = estimate(backend, state, ex.schema, tables, cfg.risk)
        if cfg.fusion.lexical != "none":
            r.col_risk = fuse(ex, r.col_risk, cfg.fusion)
        out.append((ex, r))
        if verbose and (i + 1) % every == 0:
            print(f"[score {i+1}/{len(examples)}] state_tok~{len(state)//4} "
                  f"cols_scored={len(r.col_risk)}", flush=True)
    return out


def save_risk(scored: Sequence[Tuple[Example, RiskOutput]], path: str) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex, r in scored:
            f.write(json.dumps({
                "ex_id": ex.ex_id, "db_id": ex.db_id,
                "state_chars": len(r.state),
                "tables": r.tables_in_summary,
                "col_risk": {f"{t}\x1f{c}": p for (t, c), p in r.col_risk.items()},
                "table_risk": r.table_risk,
                "n_questions": r.n_questions,
            }, ensure_ascii=False) + "\n")


def load_risk(path: str) -> Dict[str, Tuple[Dict[Pair, float], Dict[str, float], List[str]]]:
    import json

    out: Dict[str, Tuple[Dict[Pair, float], Dict[str, float], List[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cols = {}
            for k, v in d["col_risk"].items():
                t, c = k.split("\x1f")
                cols[(t, c)] = float(v)
            out[d["ex_id"]] = (cols, d.get("table_risk", {}), d.get("tables", []))
    return out


def evaluate_route(ex: Example, col_risk: Dict[Pair, float], table_risk: Dict[str, float],
                   cfg: RouteConfig, refiner: Optional[Refiner] = None) -> Dict[str, float]:
    """用已有分数跑一次路由（不再调用模型），返回该例的指标。

    refiner 非空时，"不确定"的候选会真的走一次 System-Two 精筛，
    精筛判为不需要的列会被删掉 —— 这才能体现路由省下的预算到底值不值。
    """
    r = RiskOutput(col_risk=dict(col_risk), table_risk=dict(table_risk),
                   tables_in_summary=[t.name for t in ex.schema.tables])
    res = route(r, ex.schema, cfg)
    n_escalated = res.n_escalated or len(res.escalated)
    if refiner is not None and res.escalated:
        keep = refiner.refine(ex, sorted(res.escalated))
        res.kept |= keep
        res.dropped |= (set(res.escalated) - keep)
        res.escalated = set()
    kept = res.all_kept()
    m = selection_metrics(kept, res.kept_tables, ex.schema, ex.gold, n_escalated=n_escalated)
    return m.as_dict()


def sweep_thresholds(examples: Sequence[Example],
                     risks: Dict[str, Tuple[Dict[Pair, float], Dict[str, float], List[str]]],
                     base_cfg: RouteConfig,
                     tau_lo_grid: Sequence[float],
                     tau_hi_grid: Sequence[float]) -> List[Dict]:
    """在已打分的数据上扫阈值网格，返回每组阈值的聚合指标。"""
    rows: List[Dict] = []
    for lo in tau_lo_grid:
        for hi in tau_hi_grid:
            if hi <= lo:
                continue
            cfg = RouteConfig(**{**base_cfg.__dict__, "tau_lo": float(lo), "tau_hi": float(hi)})
            mets = []
            for ex in examples:
                if ex.ex_id not in risks:
                    continue
                cols, tbls, _ = risks[ex.ex_id]
                mets.append(evaluate_route(ex, cols, tbls, cfg))
            if not mets:
                continue
            import numpy as np
            rows.append({
                "tau_lo": float(lo), "tau_hi": float(hi),
                "recall": float(np.mean([m["recall"] for m in mets])),
                "table_recall": float(np.mean([m["table_recall"] for m in mets])),
                "filter_rate": float(np.mean([m["filter_rate"] for m in mets])),
                "precision": float(np.mean([m["precision"] for m in mets])),
                "escalation_rate": float(np.mean([m["escalation_rate"] for m in mets])),
                "full_recall_rate": float(np.mean([m["recall"] >= 0.999 for m in mets])),
            })
    return rows


# ----------------------------------------------------------------------------- 聚合
def aggregate(records: Sequence[Record]) -> Dict[str, float]:
    def col(k: str) -> np.ndarray:
        return np.array([r.metrics[k] for r in records], dtype=float)

    from .metrics import mean_ci
    out: Dict[str, float] = {}
    for k in ("recall", "table_recall", "filter_rate", "precision",
              "escalation_rate", "compression_per_gold"):
        m, lo, hi = mean_ci(col(k))
        out[k] = round(float(m), 4)
        out[f"{k}_ci_lo"] = round(float(lo), 4)
        out[f"{k}_ci_hi"] = round(float(hi), 4)
    out["n_examples"] = float(len(records))
    out["avg_columns"] = round(float(np.mean([r.n_columns for r in records])), 2)
    out["avg_kept"] = round(float(np.mean([len(r.kept) for r in records])), 2)
    out["avg_gold"] = round(float(np.mean([r.n_gold for r in records])), 2)
    out["total_latency_s"] = round(float(sum(r.latency_s for r in records)), 2)
    out["avg_latency_ms"] = round(float(np.mean([r.latency_s for r in records]) * 1000), 1)
    out["total_questions"] = float(sum(r.n_questions for r in records))
    out["schema_token_reduction"] = round(
        1.0 - float(np.mean([len(r.kept) for r in records]))
        / max(1e-9, float(np.mean([r.n_columns for r in records]))), 4)
    # 完美率：全部 gold 都命中的问题占比（比均值更严格，防止被简单题拉高）
    out["full_recall_rate"] = round(float(np.mean([r.metrics["recall"] >= 0.999
                                                   for r in records])), 4)
    return out


def calibration_pairs(records: Sequence[Record]) -> Tuple[np.ndarray, np.ndarray]:
    p, y = [], []
    for r in records:
        for _t, _c, risk, lab in r.pairs:
            p.append(risk)
            y.append(lab)
    return np.asarray(p, dtype=float), np.asarray(y, dtype=int)
