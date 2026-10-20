"""不确定性路由 + 外键闭包 + 阈值标定（开题报告 §5.4 / §5.7）。

三路分流：
    risk < tau_lo  -> 直接删除（System-One 有把握）
    risk > tau_hi  -> 直接保留
    其余           -> 交给 System-Two（LLM 精筛）

阈值不抄 Jev 默认值，在 pilot 数据上按"允许丢多少 gold"反解（见 pick_tau_lo），
校准不好时用 isotonic regression 重新映射概率。这两步是论文里"校准实验"的可执行版本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .risk import RiskOutput
from .schema_graph import SchemaGraph

Pair = Tuple[str, str]


@dataclass
class RouteConfig:
    tau_lo: float = 0.10
    tau_hi: float = 0.90
    use_fk_closure: bool = True
    fk_depth: int = 2
    escalate: str = "keep"       # keep | drop | llm
    always_keep_pk: bool = True
    min_keep_columns: int = 1
    keep_whole_tables: bool = False
    table_select: str = "none"       # none | topk | frac
    """先选表再定列。

    Spider 平均 5.3 张表、gold 只用 1.5 张；表级选择的候选数少、信号强，
    比逐列打分可靠得多。选完表之后是否整表保留由 keep_whole_tables 决定。
    """
    top_k_tables: int = 2
    table_frac: float = 0.3
    """frac 模式：保留分数 >= table_frac * 本例最高表分 的表。"""
    """保留命中表的**全部列**。

    动机：gold SQL 里有一类列是纯粹的 join 路径列和同表 filter 列，
    问题里根本不提（如 teaches.course_id）。逐列打分必然漏掉它们。
    表命中就整表保留，是用少量过滤率换召回的直接手段 —— 属于"表级链接"
    这一工业界常见做法，作为对照点必须测。
    """


@dataclass
class RouteResult:
    kept: Set[Pair] = field(default_factory=set)
    dropped: Set[Pair] = field(default_factory=set)
    escalated: Set[Pair] = field(default_factory=set)
    kept_tables: Set[str] = field(default_factory=set)
    closure_added: Set[Pair] = field(default_factory=set)
    risk: Dict[Pair, float] = field(default_factory=dict)
    table_risk: Dict[str, float] = field(default_factory=dict)
    n_escalated: int = 0
    """落在不确定带里的列数。

    必须在 escalate="keep"/"drop" 清空 escalated **之前**记下来：
    否则升级率恒为 0，而这个数字恰恰是本方法最核心的成本指标
    （"有多少列值得交给他 System-Two"）。
    """

    def all_kept(self) -> Set[Pair]:
        return self.kept | self.escalated | self.closure_added


def route(risk_out: RiskOutput, schema: SchemaGraph, cfg: RouteConfig) -> RouteResult:
    res = RouteResult(risk=dict(risk_out.col_risk), table_risk=dict(risk_out.table_risk))
    scored = set(risk_out.col_risk.keys())

    for pair, r in risk_out.col_risk.items():
        if r < cfg.tau_lo:
            res.dropped.add(pair)
        elif r > cfg.tau_hi:
            res.kept.add(pair)
        else:
            res.escalated.add(pair)

    if cfg.table_select in ("topk", "frac"):
        # 表分 = 该表所有列分的最大值（列分可能已与词法信号融合）
        best: Dict[str, float] = {}
        for (t, _c), r in risk_out.col_risk.items():
            best[t] = max(best.get(t, 0.0), r)
        if best:
            if cfg.table_select == "topk":
                order = sorted(best, key=lambda t: -best[t])
                keep_t = set(order[: cfg.top_k_tables])
            else:
                hi = max(best.values())
                keep_t = {t for t, v in best.items() if v >= cfg.table_frac * hi}
            res.kept = {p for p in res.kept if p[0] in keep_t}
            res.escalated = {p for p in res.escalated if p[0] in keep_t}
            res.dropped |= {p for p in schema.columns() if p[0] not in keep_t}
            res.kept_tables = set(keep_t)
            if cfg.keep_whole_tables:
                for t in keep_t:
                    tbl = schema.table(t)
                    if tbl:
                        for c in tbl.columns:
                            res.kept.add((tbl.name, c.name))
                            res.dropped.discard((tbl.name, c.name))

    # 没被打分的元素（表级门控掉 / 粗筛没进摘要）一律记为删除。
    # 这很重要：否则过滤率会被"偷偷不打分"虚高。
    for pair in schema.columns():
        if pair not in scored:
            res.dropped.add(pair)

    res.n_escalated = len(res.escalated)
    if cfg.escalate == "keep":
        res.kept |= res.escalated
        res.escalated = set()
    elif cfg.escalate == "drop":
        res.dropped |= res.escalated
        res.escalated = set()

    for t, _c in list(res.kept) + list(res.escalated):
        res.kept_tables.add(t)

    if cfg.use_fk_closure:
        _fk_closure(res, schema, cfg)

    if cfg.keep_whole_tables:
        for t in list(res.kept_tables):
            tbl = schema.table(t)
            if not tbl:
                continue
            for c in tbl.columns:
                res.closure_added.add((tbl.name, c.name))
                res.dropped.discard((tbl.name, c.name))

    if cfg.always_keep_pk:
        for t in schema.tables:
            if t.name in res.kept_tables:
                for c in t.columns:
                    if c.is_pk:
                        res.closure_added.add((t.name, c.name))
                        res.dropped.discard((t.name, c.name))

    if len(res.all_kept()) < cfg.min_keep_columns:
        # 极端情况下（全库被打低分）保底：保住最高分的若干列，避免生成阶段零上下文
        ranked = sorted(risk_out.col_risk.items(), key=lambda kv: -kv[1])
        for pair, _r in ranked[: cfg.min_keep_columns]:
            res.kept.add(pair)
            res.dropped.discard(pair)
            res.kept_tables.add(pair[0])
    return res


def _fk_closure(res: RouteResult, schema: SchemaGraph, cfg: RouteConfig) -> None:
    """确定性外键闭包：被选中表沿外键扩展 depth 跳，补上连接列与对端主键。"""
    if not res.kept_tables:
        return
    closure_tables = schema.fk_closure(res.kept_tables, max_depth=cfg.fk_depth)
    lower = {t.lower(): t for t in schema.table_names()}
    for tl in closure_tables:
        t = lower.get(tl)
        if t is None:
            continue
        tbl = schema.table(t)
        if tbl is None:
            continue
        for fk in schema.fks:
            if fk.src_table.lower() == tl:
                if (fk.src_table, fk.src_col) not in res.all_kept():
                    res.closure_added.add((fk.src_table, fk.src_col))
                res.closure_added.add((fk.dst_table, fk.dst_col))
                res.dropped.discard((fk.src_table, fk.src_col))
                res.dropped.discard((fk.dst_table, fk.dst_col))
            elif fk.dst_table.lower() == tl:
                res.closure_added.add((fk.src_table, fk.src_col))
                res.closure_added.add((fk.dst_table, fk.dst_col))
                res.dropped.discard((fk.src_table, fk.src_col))
                res.dropped.discard((fk.dst_table, fk.dst_col))
        res.kept_tables.add(tbl.name)


# ----------------------------------------------------------------------------- 阈值标定
def pick_tau_lo(scores: Sequence[float], labels: Sequence[int], max_gold_loss: float = 0.01,
                grid: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """按"允许丢掉多少 gold 列"反解 tau_lo。

    返回 (tau_lo, 在该阈值下真实丢失的 gold 比例)。用 grid 搜索而不是解析解，
    因为概率是离散分布的（this-that 的输出集中在少数几个值上）。
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    total_gold = max(1, int(y.sum()))
    grid = list(grid) if grid is not None else list(np.round(np.arange(0.02, 0.62, 0.02), 3))
    best_t, best_loss = 0.0, 0.0
    for t in grid:
        loss = float(y[s < t].sum()) / total_gold
        if loss <= max_gold_loss:
            best_t, best_loss = float(t), loss
    return best_t, best_loss


def pick_tau_hi(scores: Sequence[float], labels: Sequence[int], min_precision: float = 0.90,
                grid: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """按"高分区直接保留要多纯"反解 tau_hi。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    grid = list(grid) if grid is not None else list(np.round(np.arange(0.40, 0.99, 0.02), 3))
    best_t, best_p = 0.99, 0.0
    for t in grid:
        m = s > t
        if m.sum() == 0:
            continue
        prec = float(y[m].mean())
        if prec >= min_precision:
            best_t, best_p = float(t), prec
            break
    return best_t, best_p


def isotonic_map(scores: Sequence[float], labels: Sequence[int]):
    """isotonic regression 重标定；返回 (fitted model, 标定后的分数)。"""
    from sklearn.isotonic import IsotonicRegression

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(s, y)
    return ir, ir.predict(s)


class ProbabilityCalibrator:
    """把原始分数映射成校准概率。fit 在 pilot 集上做，apply 在全量上用。

    为什么必须分开：在同一批数据上 fit 再算 ECE 会低估 ECE（过拟合），
    论文里要报的是 held-out 上的 ECE。
    """

    def __init__(self, method: str = "none"):
        self.method = method
        self.model = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]):
        if self.method == "isotonic":
            self.model, _ = isotonic_map(scores, labels)
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            s = np.asarray(scores, dtype=float).reshape(-1, 1)
            y = np.asarray(labels, dtype=int)
            self.model = LogisticRegression(C=1e3).fit(s, y)
        return self

    def apply(self, scores: Sequence[float]) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self.model is None:
            return s
        if self.method == "isotonic":
            return self.model.predict(s)
        return self.model.predict_proba(s.reshape(-1, 1))[:, 1]
