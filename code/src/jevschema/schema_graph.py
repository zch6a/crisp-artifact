"""Schema graph: the typed object model every other layer talks to.

设计原则（与开题报告 §5.2.1 对齐）：
    DatabaseMetadata -> SchemaMetadata -> TableMetadata -> ColumnMetadata / ForeignKeyMetadata
只保留 Jev / 决策模型需要的字段（表名、列名、类型、主键、外键、可选示例值），
丢掉索引、默认值、存储引擎等对风险判断没有信息量的内容。

本地拿到的 Spider schema 源（`richardr1126/spider-schema`）只有类型、没有示例值，
因此 `sample_values` 允许为空；等拿到带 sqlite 的产物后再回填。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class Column:
    name: str
    dtype: str = "text"
    is_pk: bool = False
    sample_values: List[str] = field(default_factory=list)

    def key(self) -> str:
        return self.name.lower()


@dataclass
class Table:
    name: str
    columns: List[Column] = field(default_factory=list)

    def col_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def has(self, col: str) -> bool:
        return col.lower() in {c.name.lower() for c in self.columns}


@dataclass
class ForeignKey:
    src_table: str
    src_col: str
    dst_table: str
    dst_col: str


@dataclass
class SchemaGraph:
    db_id: str
    tables: List[Table] = field(default_factory=list)
    fks: List[ForeignKey] = field(default_factory=list)

    # ---------------------------------------------------------------- index
    def __post_init__(self):
        self._tindex: Dict[str, Table] = {t.name.lower(): t for t in self.tables}

    def table(self, name: str) -> Optional[Table]:
        return self._tindex.get(name.lower())

    def table_names(self) -> List[str]:
        return [t.name for t in self.tables]

    def n_tables(self) -> int:
        return len(self.tables)

    def n_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    def columns(self) -> List[Tuple[str, str]]:
        """[(table, column)] 全量列，顺序稳定。"""
        return [(t.name, c.name) for t in self.tables for c in t.columns]

    def col_key(self, table: str, column: str) -> str:
        return f"{table.lower()}.{column.lower()}"

    def has_column(self, table: str, column: str) -> bool:
        t = self.table(table)
        return bool(t and t.has(column))

    def col_type(self, table: str, column: str) -> str:
        t = self.table(table)
        if not t:
            return "text"
        for c in t.columns:
            if c.name.lower() == column.lower():
                return c.dtype
        return "text"

    # ---------------------------------------------------------------- graph
    def fk_neighbors(self, table: str) -> List[Tuple[str, str, str, str]]:
        """返回与该表相连的所有外键边 (src_t, src_c, dst_t, dst_c)，方向归一到 table 出发。"""
        out = []
        tl = table.lower()
        for fk in self.fks:
            if fk.src_table.lower() == tl:
                out.append((fk.src_table, fk.src_col, fk.dst_table, fk.dst_col))
            elif fk.dst_table.lower() == tl:
                out.append((fk.dst_table, fk.dst_col, fk.src_table, fk.src_col))
        return out

    def fk_closure(self, tables: Iterable[str], max_depth: int = 2) -> Set[str]:
        """从给定表出发沿外键做传递闭包，返回表名集合（含起点）。"""
        seen: Set[str] = {t.lower() for t in tables}
        frontier = list(seen)
        for _ in range(max_depth):
            new: Set[str] = set()
            for t in frontier:
                for _s, _sc, d, _dc in self.fk_neighbors(t):
                    if d.lower() not in seen:
                        new.add(d.lower())
            if not new:
                break
            seen |= new
            frontier = list(new)
        lower2real = {t.name.lower(): t.name for t in self.tables}
        return {lower2real.get(x, x) for x in seen}

    def sqlglot_schema(self) -> Dict[str, Dict[str, str]]:
        return {t.name: {c.name: c.dtype for c in t.columns} for t in self.tables}

    # ---------------------------------------------------------------- text
    def summary(
        self,
        tables: Optional[Sequence[str]] = None,
        with_fk: bool = True,
        with_samples: int = 0,
        max_cols_per_table: Optional[int] = None,
    ) -> str:
        """开题报告 §5.2 的 Schema Summary：表/列/类型/主键/外键/（可选）示例值。"""
        lines: List[str] = []
        for t in self.tables:
            if tables is not None and t.name.lower() not in {x.lower() for x in tables}:
                continue
            cols = t.columns if max_cols_per_table is None else t.columns[:max_cols_per_table]
            col_txt = []
            for c in cols:
                tag = c.dtype or "text"
                if c.is_pk:
                    tag += ", PK"
                if with_samples and c.sample_values:
                    ex = ", ".join(str(v) for v in c.sample_values[:with_samples])
                    tag += f", e.g. {ex}"
                col_txt.append(f"{c.name} ({tag})")
            lines.append(f"Table {t.name}: " + "; ".join(col_txt))
            if max_cols_per_table is not None and len(t.columns) > max_cols_per_table:
                lines.append(f"  ... ({len(t.columns) - max_cols_per_table} more columns)")
        if with_fk and self.fks:
            fk_txt = "; ".join(
                f"{f.src_table}.{f.src_col} -> {f.dst_table}.{f.dst_col}" for f in self.fks
            )
            lines.append("Foreign keys: " + fk_txt)
        return "\n".join(lines)


# ----------------------------------------------------------------------------- parsers
_TYPE_ALIASES = {
    "number": "number",
    "text": "text",
    "real": "number",
    "int": "number",
    "integer": "number",
    "float": "number",
    "varchar": "text",
    "date": "text",
    "time": "text",
    "bool": "text",
}


def _parse_columns(block: str) -> List[Column]:
    """`Perpetrator_ID (number) , People_ID (number)` -> [Column...]"""
    cols: List[Column] = []
    for part in block.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(?P<name>[^(]+?)\s*\((?P<dtype>[^)]*)\)\s*$", part)
        if m:
            name = m.group("name").strip()
            dtype = m.group("dtype").strip().lower()
        else:
            name, dtype = part, "text"
        cols.append(Column(name=name, dtype=_TYPE_ALIASES.get(dtype, dtype or "text")))
    return cols


def parse_spider_schema_row(row: Dict) -> SchemaGraph:
    """解析 `richardr1126/spider-schema` 的一行。"""
    db_id = row["db_id"]
    tables: List[Table] = []
    for tbl in row.get("Schema (values (type))", "").split("|"):
        tbl = tbl.strip()
        if not tbl or ":" not in tbl:
            continue
        name, colblock = tbl.split(":", 1)
        tables.append(Table(name=name.strip(), columns=_parse_columns(colblock)))

    for pk_part in row.get("Primary Keys", "").split("|"):
        if ":" not in pk_part:
            continue
        tname, cols = pk_part.split(":", 1)
        t = next((x for x in tables if x.name.lower() == tname.strip().lower()), None)
        if not t:
            continue
        pk_names = {c.strip().lower() for c in cols.split(",")}
        for c in t.columns:
            if c.name.lower() in pk_names:
                c.is_pk = True

    fks: List[ForeignKey] = []
    for fk_part in row.get("Foreign Keys", "").split("|"):
        fk_part = fk_part.strip()
        m = re.match(
            r"^(?P<st>[^:]+)\s*:\s*(?P<sc>[^=]+?)\s+equals\s+(?P<dt>[^:]+)\s*:\s*(?P<dc>.+)$",
            fk_part,
        )
        if not m:
            continue
        fks.append(
            ForeignKey(
                src_table=m.group("st").strip(),
                src_col=m.group("sc").strip(),
                dst_table=m.group("dt").strip(),
                dst_col=m.group("dc").strip(),
            )
        )
    return SchemaGraph(db_id=db_id, tables=tables, fks=fks)


def load_spider_schemas(path: str) -> Dict[str, SchemaGraph]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out: Dict[str, SchemaGraph] = {}
    for r in rows:
        g = parse_spider_schema_row(r)
        out[g.db_id] = g
    return out


def merge_schemas(db_id: str, graphs: Sequence[SchemaGraph], prefix: bool = True) -> SchemaGraph:
    """把多个库拼成一个超大 schema —— Schema Scaling Benchmark 的构造原语。

    prefix=True 时给表名加库名前缀，模拟企业库里"同一个逻辑名出现在多个域"的现实情况，
    同时保证列名解析不歧义。
    """
    tables: List[Table] = []
    fks: List[ForeignKey] = []
    for g in graphs:
        pfx = (g.db_id + "_") if prefix else ""
        for t in g.tables:
            tables.append(
                Table(
                    name=pfx + t.name,
                    columns=[Column(c.name, c.dtype, c.is_pk, list(c.sample_values)) for c in t.columns],
                )
            )
        for f in g.fks:
            fks.append(
                ForeignKey(pfx + f.src_table, f.src_col, pfx + f.dst_table, f.dst_col)
            )
    return SchemaGraph(db_id=db_id, tables=tables, fks=fks)
