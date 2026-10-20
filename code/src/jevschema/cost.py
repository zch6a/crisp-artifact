"""成本-延迟模型（开题报告 §6.3 / 实验 6）。

三个系统各自的成本口径不一样，硬凑成一个数会误导：
  * 本地开源决策模型：花的是 GPU 时间（$0 API），但要报延迟和显存；
  * Jev API：输入 $0.042/Mtok，输出免费，state 只算一次；
  * LLM 精筛 / 生成：按 endpoint 报价估。

所以这里不合成"总成本"，而是同时给出 (usd, latency, gpu_seconds)，
画帕累托前沿时按部署场景选轴。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class CostReport:
    backend: str = ""
    n_examples: int = 0
    decision_calls: int = 0
    decision_questions: int = 0
    decision_state_tokens: int = 0
    decision_question_tokens: int = 0
    decision_latency_s: float = 0.0
    decision_usd: float = 0.0
    llm_refine_calls: int = 0
    llm_refine_usd: float = 0.0
    llm_refine_latency_s: float = 0.0
    extra: Dict[str, float] = None  # type: ignore[assignment]

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["extra"] = self.extra or {}
        d["usd_total"] = round(self.decision_usd + self.llm_refine_usd, 6)
        d["latency_total_s"] = round(self.decision_latency_s + self.llm_refine_latency_s, 3)
        d["questions_per_example"] = round(
            self.decision_questions / self.n_examples, 2) if self.n_examples else 0.0
        d["latency_per_example_ms"] = round(
            1000 * self.decision_latency_s / self.n_examples, 2) if self.n_examples else 0.0
        return d


def backend_cost(backend, n_examples: int) -> CostReport:
    inner = getattr(backend, "inner", backend)
    s = inner.stats.as_dict()
    usd = 0.0
    if getattr(inner, "name", "") == "jev-api":
        usd = float(getattr(inner, "spent_usd", 0.0))
    if hasattr(backend, "hits"):
        s["cache_hits"] = backend.hits
        s["cache_misses"] = backend.misses
    return CostReport(
        backend=getattr(backend, "name", "?"),
        n_examples=n_examples,
        decision_calls=int(s.get("calls", 0)),
        decision_questions=int(s.get("n_questions", 0)),
        decision_state_tokens=int(s.get("state_tokens", 0)),
        decision_question_tokens=int(s.get("question_tokens", 0)),
        decision_latency_s=float(s.get("latency_s", 0.0)),
        decision_usd=usd,
        extra={k: v for k, v in s.items() if k.startswith("extra_")},
    )


def add_refiner_cost(rep: CostReport, refiner) -> CostReport:
    if hasattr(refiner, "cost"):
        c = refiner.cost
        rep.llm_refine_calls = c.calls
        rep.llm_refine_usd = round(c.usd, 6)
        rep.llm_refine_latency_s = round(c.latency_s, 3)
    return rep


def estimate_jev_cost(n_examples: int, avg_state_tokens: int, avg_questions: int,
                      avg_question_tokens: int, price_per_mtok: float = 0.042) -> float:
    """按官方口径估 Jev 全量费用：state 一次付费，questions 累加。"""
    mtok = n_examples * (avg_state_tokens + avg_questions * avg_question_tokens) / 1e6
    return mtok * price_per_mtok
