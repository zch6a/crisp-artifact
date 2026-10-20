"""System-One 风险估计层（开题报告 §5.3）。

核心是 Schema Reasoning Risk 的问法：不问"这列相关吗"，而问
"删掉这列，后面写 SQL 会不会崩"。每个 schema element 一道 typed question，
同一次请求里共享 state（问题 + schema summary），N 个判断并行。

多视角（multi-view）来自 jev-compass 的发现：单一视角的 judge 有系统性偏差，
"风险"与"必需"两种问法互补，取平均通常优于任一单视角。这里保留开关，
消融实验里可以单独关掉验证它对 Text-to-SQL 是否同样成立。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backends.base import DecisionBackend
from .schema_graph import SchemaGraph


# ----------------------------------------------------------------------------- 问题模板
COL_RISK = (
    "If the column `{t}.{c}` were removed from this database schema, would writing the "
    "correct SQL to answer the question fail or become substantially harder?"
)
COL_NEED = (
    "Does answering the question require the column `{t}.{c}` to be used in the SQL query?"
)
COL_FK = (
    "Is the column `{t}.{c}` part of a foreign key relationship that connects to a table "
    "needed to answer the question?"
)
TBL_RISK = (
    "If the table `{t}` were removed from this database schema, would writing the correct "
    "SQL to answer the question fail or become substantially harder?"
)
TBL_NEED = "Does answering the question require the table `{t}` to be used in the SQL query?"
TBL_ASK = "Does the question ask for information that is stored in the table `{t}`?"

# 诊断（scripts/diag_risk.py，laya 零样本，n=60）给出的排序：
#   value 0.748 > asks_for 0.714 > implied 0.693 > without 0.679 > need 0.669
#   > risk 0.606 > risk_statement 0.545 > deleted 0.516
# "删除风险"式问法在零样本下反而弱于"问题是否在问这列的值"，这是必须记录的实测结果。
COL_ASK = "Does the question ask for information that is stored in the column `{t}.{c}`?"
COL_VALUE = "Does the question ask about values that are stored in the column `{t}.{c}`?"
COL_IMPLIED = "Is the column `{t}.{c}` mentioned or implied by the question?"
COL_WITHOUT = (
    "The SQL query that answers the question cannot be written correctly without the "
    "column `{t}.{c}`."
)

VIEWS = {
    "risk": COL_RISK,
    "need": COL_NEED,
    "fk": COL_FK,
    "ask": COL_ASK,
    "value": COL_VALUE,
    "implied": COL_IMPLIED,
    "without": COL_WITHOUT,
}
TABLE_VIEWS = {
    "risk": TBL_RISK,
    "need": TBL_NEED,
    "ask": TBL_ASK,
}


@dataclass
class RiskConfig:
    views: Tuple[str, ...] = ("risk",)
    combine: str = "mean"                 # mean | max | first
    table_level: bool = True              # 先打分表，再只给入选表的列打分
    table_views: Tuple[str, ...] = ("risk",)
    table_keep_quantile: float = 0.0      # 0 = 不按分位数硬砍表，只做列级路由
    batch_size: int = 48


@dataclass
class RiskOutput:
    col_risk: Dict[Tuple[str, str], float] = field(default_factory=dict)
    col_view: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    table_risk: Dict[str, float] = field(default_factory=dict)
    state: str = ""
    tables_in_summary: List[str] = field(default_factory=list)
    n_questions: int = 0


def _combine(vals: Sequence[float], how: str) -> float:
    v = np.asarray(vals, dtype=float)
    if how == "max":
        return float(v.max())
    if how == "first":
        return float(v[0])
    return float(v.mean())


def score_columns(
    backend: DecisionBackend,
    state: str,
    elements: Sequence[Tuple[str, str]],
    views: Sequence[str] = ("risk",),
    combine: str = "mean",
    batch_size: int = 48,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """对每个 (table, column) 在每个视角下打分，返回 {pair: {view: p}}。"""
    out: Dict[Tuple[str, str], Dict[str, float]] = {e: {} for e in elements}
    for view in views:
        tmpl = VIEWS[view]
        statements = [tmpl.format(t=t, c=c) for t, c in elements]
        ps: List[float] = []
        for i in range(0, len(statements), batch_size):
            chunk = statements[i:i + batch_size]
            ps.extend(backend.judge_batch(state, chunk))
        for e, p in zip(elements, ps):
            out[e][view] = float(p)
    return out


def score_tables(
    backend: DecisionBackend,
    state: str,
    tables: Sequence[str],
    views: Sequence[str] = ("risk",),
    combine: str = "mean",
    batch_size: int = 48,
) -> Dict[str, float]:
    per_view: Dict[str, Dict[str, float]] = {v: {} for v in views}
    for view in views:
        tmpl = TABLE_VIEWS[view]
        statements = [tmpl.format(t=t) for t in tables]
        ps: List[float] = []
        for i in range(0, len(statements), batch_size):
            ps.extend(backend.judge_batch(state, statements[i:i + batch_size]))
        per_view[view] = dict(zip(tables, ps))
    return {t: _combine([per_view[v][t] for v in views], combine) for t in tables}


def estimate(
    backend: DecisionBackend,
    state: str,
    schema: SchemaGraph,
    tables: Sequence[str],
    cfg: RiskConfig,
) -> RiskOutput:
    """完整的表级 + 列级风险估计。"""
    out = RiskOutput(state=state, tables_in_summary=list(tables))
    table_list = [t for t in schema.table_names() if t in set(tables)]

    col_elements: List[Tuple[str, str]] = []
    if cfg.table_level and table_list:
        out.table_risk = score_tables(backend, state, table_list, cfg.table_views,
                                      cfg.combine, cfg.batch_size)
        # 表级只是"要不要看这张表的列"的门控：低风险表直接整表跳过，
        # 这是把 O(|S|) 的打分压到 O(|候选|) 的关键一步。
        keep = set(table_list)
        if cfg.table_keep_quantile > 0 and out.table_risk:
            vals = np.array([out.table_risk[t] for t in table_list])
            thr = float(np.quantile(vals, cfg.table_keep_quantile))
            keep = {t for t in table_list if out.table_risk[t] >= thr}
        for t in schema.tables:
            if t.name in keep:
                col_elements.extend((t.name, c.name) for c in t.columns)
    else:
        col_elements = [(t, c) for (t, c) in schema.columns() if t in set(tables)]

    if col_elements:
        per_view = score_columns(backend, state, col_elements, cfg.views, cfg.combine, cfg.batch_size)
        for e, vs in per_view.items():
            out.col_view[e] = vs
            out.col_risk[e] = _combine([vs[v] for v in cfg.views], cfg.combine)
    out.n_questions = len(table_list) * len(cfg.table_views) + len(col_elements) * len(cfg.views)
    return out
