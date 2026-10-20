"""从 gold SQL 抽"必须用到的 schema 元素"，作为风险估计的监督信号。

监督信号的形式不是"相关列集合"，而是**一组需求**：

    requirement = {(t1, c1), (t2, c2), ...}

每个需求代表 SQL 里出现的一次列引用。大多数时候它是单元素（列能被唯一解析到表）；
当 SQL 没写表限定符、且该列名在多个表里都存在时，需求就是多元素集合 ——
这时候"命中其中任意一个"就算满足。这样定义避免在歧义列上人为惩罚方法，
也不会人为放宽：recall 的分母是需求数，不是列数。

为什么不用 `sqlglot.optimizer.qualify`：它遇到 Spider 里那些不规范的 SQL
（缺表限定、子查询别名、SQLite 方言函数）会抛错，抛错就丢样本。这里的解析是
best-effort：先别名，再唯一匹配，最后退化到"全部候选"，并如实记录置信度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .schema_graph import SchemaGraph

Pair = Tuple[str, str]


@dataclass
class GoldSet:
    requirements: List[Set[Pair]] = field(default_factory=list)
    tables: Set[str] = field(default_factory=set)
    ambiguous: int = 0
    unresolved: int = 0
    parse_ok: bool = True

    # ------------------------------------------------------------------ metrics
    def flat_pairs(self) -> Set[Pair]:
        out: Set[Pair] = set()
        for req in self.requirements:
            out |= req
        return out

    def recall(self, selected_pairs: Set[Pair]) -> float:
        if not self.requirements:
            return 1.0
        hit = sum(1 for req in self.requirements if req & selected_pairs)
        return hit / len(self.requirements)

    def table_recall(self, selected_tables: Set[str]) -> float:
        if not self.tables:
            return 1.0
        low = {t.lower() for t in selected_tables}
        return sum(1 for t in self.tables if t.lower() in low) / len(self.tables)


def _norm(x: str) -> str:
    return (x or "").strip().strip("`\"[]").lower()


# ----------------------------------------------------------------------------- sqlglot 路径
def _extract_with_sqlglot(sql: str, schema: SchemaGraph) -> Optional[GoldSet]:
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return None

    try:
        tree = sqlglot.parse_one(sql, read="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return None
    if tree is None:
        return None

    alias2table: Dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        name = _norm(t.name)
        if not name:
            continue
        if not schema.table(name):
            # 子查询别名 / CTE：无法映射到基表，记录后跳过
            alias2table[_norm(t.alias) or name] = None  # type: ignore[assignment]
            continue
        alias2table[_norm(t.alias) or name] = schema.table(name).name  # type: ignore[union-attr]

    scope_tables: List[str] = [v for v in alias2table.values() if v]
    reqs: List[Set[Pair]] = []
    tables: Set[str] = set()
    ambiguous = 0
    unresolved = 0

    for c in tree.find_all(exp.Column):
        cname = _norm(c.name)
        if not cname or cname == "*":
            continue
        qual = _norm(c.table)
        cands: Set[Pair] = set()
        if qual and qual in alias2table:
            real = alias2table[qual]
            if real and schema.has_column(real, cname):
                cands.add((real, _real_col(schema, real, cname)))
        elif qual and schema.table(qual) and schema.has_column(qual, cname):
            cands.add((schema.table(qual).name, _real_col(schema, qual, cname)))
        else:
            pool = scope_tables or schema.table_names()
            for t in pool:
                if schema.has_column(t, cname):
                    cands.add((t, _real_col(schema, t, cname)))
        if not cands:
            unresolved += 1
            continue
        if len(cands) > 1:
            ambiguous += 1
        reqs.append(cands)
        tables |= {t for t, _ in cands}

    for t in tree.find_all(exp.Table):
        real = alias2table.get(_norm(t.alias) or _norm(t.name))
        if real:
            tables.add(real)
    return GoldSet(reqs, tables, ambiguous, unresolved, True)


def _real_col(schema: SchemaGraph, table: str, col: str) -> str:
    t = schema.table(table)
    if not t:
        return col
    for c in t.columns:
        if c.name.lower() == col.lower():
            return c.name
    return col


# ----------------------------------------------------------------------------- 正则兜底
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _extract_regex(sql: str, schema: SchemaGraph) -> GoldSet:
    """sqlglot 不可用/解析失败时的兜底：按 token 匹配列名，表限定能带就带。"""
    all_cols: Dict[str, List[str]] = {}
    for t, c in schema.columns():
        all_cols.setdefault(c.lower(), []).append(t)
    reqs: List[Set[Pair]] = []
    tables: Set[str] = set()
    ambiguous = 0
    unresolved = 0
    body = re.sub(r"'[^']*'", " ", sql)
    toks = [t for t in _IDENT.findall(body)]
    upper = {"SELECT", "FROM", "WHERE", "GROUP", "ORDER", "BY", "AND", "OR", "JOIN", "ON",
             "AS", "IN", "NOT", "NULL", "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT",
             "HAVING", "LIMIT", "DESC", "ASC", "INNER", "LEFT", "RIGHT", "OUTER", "CASE",
             "WHEN", "THEN", "ELSE", "END", "LIKE", "BETWEEN", "IS", "CAST", "EXISTS", "ALL"}
    tbl_alias: Dict[str, str] = {}
    for i, tk in enumerate(toks):
        if tk.upper() in ("FROM", "JOIN") and i + 1 < len(toks):
            nxt = toks[i + 1]
            if schema.table(nxt):
                tbl_alias[nxt.lower()] = schema.table(nxt).name
                tables.add(schema.table(nxt).name)
                if i + 2 < len(toks) and toks[i + 2].upper() == "AS" and i + 3 < len(toks):
                    tbl_alias[toks[i + 3].lower()] = schema.table(nxt).name
    for i, tk in enumerate(toks):
        low = tk.lower()
        if tk.upper() in upper:
            continue
        if low not in all_cols:
            continue
        prev = toks[i - 1].lower() if i else ""
        if prev in tbl_alias and schema.has_column(tbl_alias[prev], low):
            reqs.append({(tbl_alias[prev], _real_col(schema, tbl_alias[prev], low))})
            tables.add(tbl_alias[prev])
            continue
        cands = {(t, _real_col(schema, t, low)) for t in all_cols[low]}
        if len(cands) > 1:
            ambiguous += 1
        reqs.append(cands)
        tables |= {t for t, _ in cands}
    return GoldSet(reqs, tables, ambiguous, unresolved, False)


def extract_gold(sql: str, schema: SchemaGraph, prefer_sqlglot: bool = True) -> GoldSet:
    if prefer_sqlglot:
        g = _extract_with_sqlglot(sql, schema)
        if g is not None and g.requirements:
            return g
    return _extract_regex(sql, schema)


def gold_stats(golds: Sequence[GoldSet]) -> Dict[str, float]:
    n = len(golds) or 1
    return {
        "n": float(len(golds)),
        "parse_ok_rate": sum(g.parse_ok for g in golds) / n,
        "avg_requirements": sum(len(g.requirements) for g in golds) / n,
        "avg_tables": sum(len(g.tables) for g in golds) / n,
        "ambiguous_rate": sum(1 for g in golds if g.ambiguous) / n,
        "unresolved_rate": sum(1 for g in golds if g.unresolved) / n,
    }
