"""schema 选中集合的结构化扩展（top-k 之后的确定性补全）。

两代机制：

v1（expand_kept）—— 只有 FK 闭包 / 整表保留两种粗粒度模式。
v2（closure）—— 失败诊断（scripts/failure_diag.py，BIRD-Dev 全量 1534）给出的结论是：
    A_词汇断链  99 题（占失败 69.7%）：gold 列名与 question+evidence 零词汇重叠，
                embedding 也排不进前列（satscores.NumTstTakr ← "number of test takers"）
    D_表级漏召   0 题：gold 列所在的表**从来没有被整体漏掉**
也就是说断的不是"表"，是"表内某一列的别名"。所以 v2 加了一条
**入选表内补齐（table padding）**：一张表只要有任意一列进了 top-k，
就把这张表里分数次高的若干列也带上。它和 FK 闭包互补：
    FK 闭包   补的是跨表连接键（横向）
    表内补齐  补的是表内同义别名（纵向）
两者都是确定性规则，零模型调用、零 LLM token —— 这正是论文要讲的
"把 schema linking 从昂贵的迭代探索变成一次性批量判断"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

Pair = Tuple[str, str]


@dataclass
class ClosureConfig:
    """结构补全的旋钮（互相独立，可叠加）。

    k（主键必留）和 s（小表整表）是 per-DB 归因后补的两条：
      - california_schools 上 20 处漏列全部落在已选中的表里，且 satscores 只有 11 列
        却只保留了几列 → 小表该整表保留；
      - european_football_2 上 14 处漏列里 10 处是 `Match.id`（115 列表的主键），
        而它没被 FK 闭包覆盖 → 入选表的主键该无条件保留。
    """
    fk: bool = True          # 外键闭包：补上选中表的外键端点列
    pad: int = 0             # 入选表内再补 pad 列（按打分顺序取该表内次高的）
    small: int = 0           # 入选表里总列数 ≤ small 的，整表保留
    frac: float = 0.0        # 比例补齐：入选表保留 ceil(frac × 该表列数) 列
    pk: bool = False         # 入选表的主键无条件保留（连接键的兜底）

    def tag(self) -> str:
        parts = ["f" if self.fk else "x"]
        if self.pad:
            parts.append(f"p{self.pad}")
        if self.small:
            parts.append(f"s{self.small}")
        if self.frac:
            parts.append(f"r{self.frac}")
        if self.pk:
            parts.append("k1")
        return "".join(parts)


_CFG_RE = re.compile(r"^(f\d|x\d)?(p\d+)?(s\d+)?(r[\d.]+)?(k[01])?$")


def parse_closure(spec: str) -> ClosureConfig:
    """解析 'f1p5s10k1' / 'f1r0.5' / 'f0' 这类简写。空串 -> 全关（等价于 none）。"""
    spec = (spec or "").strip()
    if not spec or spec in ("none", "f0"):
        return ClosureConfig(fk=False)
    m = _CFG_RE.match(spec)
    if not m:
        raise ValueError(f"无法解析补全配置: {spec!r}（形如 f1p5s10k1 / f1r0.5）")
    fk = bool(int(m.group(1)[1:])) if m.group(1) else False
    pad = int(m.group(2)[1:]) if m.group(2) else 0
    small = int(m.group(3)[1:]) if m.group(3) else 0
    frac = float(m.group(4)[1:]) if m.group(4) else 0.0
    pk = bool(int(m.group(5)[1:])) if m.group(5) else False
    return ClosureConfig(fk=fk, pad=pad, small=small, frac=frac, pk=pk)


def closure(schema, order: Sequence[Pair], k: int, cfg: ClosureConfig) -> List[Pair]:
    """打分排序后的确定性补全。

    order —— 按分数降序排列的**全部列**（不只是 top-k），补齐时要从里面取次高列。
    k     —— top-k 截断值（这就是"预算"）。
    """
    kept: List[Pair] = list(order[: max(1, min(len(order), int(k)))])
    if not kept:
        return kept
    by_table: Dict[str, List[Pair]] = {}
    for p in order:
        by_table.setdefault(p[0], []).append(p)

    if cfg.pad or cfg.small or cfg.frac:
        have: Set[str] = {p[0] for p in kept}
        add: List[Pair] = []
        for t in have:
            cols = by_table.get(t, [])
            if not cols:
                continue
            sel = sum(1 for p in kept if p[0] == t)
            if cfg.small and len(cols) <= cfg.small:
                add.extend(cols)
                continue
            need = sel
            if cfg.frac > 0:
                need = max(need, int(-(-cfg.frac * len(cols) // 1)))  # ceil
            if cfg.pad:
                need = max(need, sel + cfg.pad)
            add.extend(cols[: min(len(cols), need)])
        kept = list(dict.fromkeys(list(kept) + list(add)))

    if cfg.pk:
        have = {p[0] for p in kept}
        add = [(t.name, c.name) for t in schema.tables if t.name in have
               for c in t.columns if c.is_pk]
        kept = list(dict.fromkeys(list(kept) + list(add)))

    if cfg.fk:
        kept = _fk_close(schema, kept)
        if cfg.pk:
            # FK 闭包会带进新表，新表的主键同样要留，否则连接键还是断的
            have = {p[0] for p in kept}
            add = [(t.name, c.name) for t in schema.tables if t.name in have
                   for c in t.columns if c.is_pk]
            kept = list(dict.fromkeys(list(kept) + list(add)))
    valid = {(t.name, c.name) for t in schema.tables for c in t.columns}
    return [p for p in kept if p in valid]


def _fk_close(schema, kept: Sequence[Pair]) -> List[Pair]:
    out: Set[Pair] = set(kept)
    tabs = {t.lower() for t, _ in out}
    for f in schema.fks:
        if f.src_table.lower() in tabs:
            out.add((f.dst_table, f.dst_col))
            out.add((f.src_table, f.src_col))
        elif f.dst_table.lower() in tabs:
            out.add((f.src_table, f.src_col))
            out.add((f.dst_table, f.dst_col))
    valid = {(t.name, c.name) for t in schema.tables for c in t.columns}
    return [p for p in out if p in valid]


def expand_kept(schema, kept: list, mode: str) -> list:
    """旧接口（v1），保留给既有脚本。

    fk    —— 补上选中表的外键端点列。JOIN 键几乎从不在问题里被提到，
             纯语义打分必然漏掉它们，而漏一个 join key 整题 SRR 就归零。
    table —— 选中列所属表整表保留。
    """
    if mode == "none" or not kept:
        return list(kept)
    out = set(kept)
    if "table" in mode:
        tabs = {t for t, _ in kept}
        for t in schema.tables:
            if t.name in tabs:
                out |= {(t.name, c.name) for c in t.columns}
    if "fk" in mode:
        out = set(_fk_close(schema, list(out)))
    valid = {(t.name, c.name) for t in schema.tables for c in t.columns}
    return [p for p in out if p in valid]
