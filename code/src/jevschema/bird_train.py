"""BIRD **train** 域内训练数据（来自 arcwise/bird_sql_annotated 的 train_0711.jsonl）。

为什么需要它：微调模型只在 Spider 上训过，直接搬到 BIRD-Dev 是跨集零样本迁移，
AUROC 从 0.976 掉到 0.89 —— 掉的那部分不是方法问题，是域偏移。
论文要冲 SRR，就必须在**目标基准的 train 分割**上训练；在 dev 上训练是泄漏，不能做。

这份数据的价值在于它同时给了两样东西：
  * `input`  —— 完整 DDL（CREATE TABLE + 列注释），可直接解析成 SchemaGraph；
  * `output` —— 人工/模型标注的 "Input Columns" 段，形如 `movie_id::ratings`，
                即"这道题真正要用到的 (table, column)"。
比从 gold SQL 反解更干净：标注里包含 JOIN 键（`movie_id::ratings` 这种
只出现在 ON 子句里的列），而 SQL 解析经常把它们判成 unresolved。

⚠️ 口径风险：标注列是"解题会用到的列"，与 gold-SQL 抽取的"需求集合"不完全等价
（标注可能包含中间步骤列）。因此训练用它、评测一律用 BIRD-Dev 的 gold SQL，
两者不一致的部分会作为噪声拉低上限，这一点要在论文里说明。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .data_loader import Example
from .gold_labels import GoldSet
from .schema_graph import Column, ForeignKey, SchemaGraph, Table
from .paths import DATA

DEFAULT_PATH = os.environ.get(
    "BIRD_TRAIN_ANNOTATED",
    os.path.join(DATA, "raw", "bird", "arcwise", "train_0711.jsonl"))

_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+[`\"\[]?(?P<name>[\w]+)[`\"\]]?\s*\((?P<body>.*?)\n\);",
                       re.S | re.I)
_CREATE_RE = re.compile(r"^CREATE\s+TABLE\s+[`\"\[]?(?P<name>[\w]+)[`\"\]]?\s*\(.*$", re.I)
_FK_RE = re.compile(r"Foreign\s+keys?:\s*(?P<ref>[\w]+\.[\w]+)", re.I)
_GOLD_RE = re.compile(r"^\s*(?P<col>[\w]+)::(?P<tbl>[\w]+)\s*$")

_SKIP_PREFIX = ("primary key", "foreign key", "unique", "constraint", "check")


def _split_top_level(body: str) -> List[str]:
    """按逗号切分，但跳过括号内部（VARCHAR(10)、DECIMAL(10,2)）。"""
    out, buf, depth = [], [], 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [s.strip() for s in out if s.strip()]


def parse_ddl(ddl: str, db_id: str) -> SchemaGraph:
    """把 CREATE TABLE 块解析成 SchemaGraph。列注释里的 Foreign keys 也抽成 FK。

    ⚠️ 用逐行状态机而不是 `CREATE TABLE ... (.*?)\\);` 正则：
    后者在大 DDL（几千列）上会因 `.*?` 回溯退化到近乎 O(n²)，实测解析 6000 条
    样本能卡住十几分钟。线性扫描是这里唯一可接受的实现。
    """
    tables: List[Table] = []
    fks: List[ForeignKey] = []
    cur: Optional[Table] = None
    pending_comment = ""
    for raw in ddl.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            pending_comment += " " + line[2:]
            continue
        if cur is None:
            m = _CREATE_RE.match(line)
            if m:
                cur = Table(name=m.group("name"))
                pending_comment = ""
            continue
        # 表体结束：`);` 或单独的 `)`
        if line.startswith(")"):
            tables.append(cur)
            cur = None
            pending_comment = ""
            continue
        low = line.lower()
        if low.startswith(_SKIP_PREFIX):
            pending_comment = ""
            continue
        # `col_name TYPE...` —— 名字可能带引号；也可能只有列名（无类型）
        mm = re.match(r"^[`\"\[]?(?P<col>[\w]+)[`\"\]]?\s*(?P<rest>.*)$", line)
        if not mm:
            pending_comment = ""
            continue
        col = mm.group("col")
        rest = (mm.group("rest") or "").strip().rstrip(",").strip()
        # 只取类型的第一个 token，并剥掉括号里的长度/精度
        dtype = "text"
        if rest:
            tok0 = rest.split()[0]
            cut = tok0.find("(")
            dtype = (tok0[:cut] if cut > 0 else tok0).lower() or "text"
        cur.columns.append(Column(name=col, dtype=dtype))
        fm = _FK_RE.search(pending_comment)
        if fm:
            rt, rc = fm.group("ref").split(".")
            fks.append(ForeignKey(cur.name, col, rt, rc))
        pending_comment = ""
    if cur is not None:
        tables.append(cur)
    return SchemaGraph(db_id=db_id, tables=tables, fks=fks)


def parse_gold_columns(output: str) -> Set[Tuple[str, str]]:
    """从标注文本里抠 `col::table` 列表。"""
    body = output
    if "Input Columns" in body:
        body = body.split("Input Columns", 1)[1]
    out: Set[Tuple[str, str]] = set()
    for line in body.split("\n"):
        m = _GOLD_RE.match(line)
        if m:
            out.add((m.group("tbl"), m.group("col")))
    return out


def _db_id_of(ddl: str) -> str:
    """同一个库的 DDL 必须映射到同一个 db_id，否则 schema 缓存会失效。"""
    names = tuple(sorted(set(m.group("name").lower()
                             for m in _TABLE_RE.finditer(ddl))))
    h = hashlib.md5("|".join(names).encode()).hexdigest()[:10]
    return f"bt_{h}"


def _clean_question(instruction: str) -> str:
    return " ".join(instruction.split()).strip()


def load_bird_train(
    path: str = DEFAULT_PATH,
    limit: Optional[int] = None,
    min_gold: int = 1,
    max_columns: int = 4000,
    log_every: int = 2000,
) -> Tuple[List[Example], Dict[str, int]]:
    """返回 (examples, stats)。DB 的 DDL 只解析一次并缓存。"""
    schemas: Dict[str, SchemaGraph] = {}
    out: List[Example] = []
    stats = {"total": 0, "no_gold": 0, "no_schema": 0, "gold_unmatched": 0, "kept": 0}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(out) >= limit:
                break
            try:
                d = json.loads(line)
            except Exception:                       # noqa: BLE001
                continue
            stats["total"] += 1
            ddl = d.get("input", "")
            db_id = _db_id_of(ddl)
            if db_id not in schemas:
                g = parse_ddl(ddl, db_id)
                if not g.tables:
                    stats["no_schema"] += 1
                    continue
                schemas[db_id] = g
            sch = schemas[db_id]
            if sch.n_columns() > max_columns:
                continue
            gold_cols = parse_gold_columns(d.get("output", ""))
            if len(gold_cols) < min_gold:
                stats["no_gold"] += 1
                continue
            # 标注里可能有 schema 中不存在的 (t, c) —— 直接丢弃，避免把噪声当正样本
            real = {(t, c) for (t, c) in gold_cols if sch.has_column(t, c)}
            if not real:
                stats["gold_unmatched"] += 1
                continue
            stats["kept"] += 1
            ex = Example(
                ex_id=f"bt-{i}",
                db_id=db_id,
                question=_clean_question(d.get("instruction", "")),
                sql="",
                schema=sch,
            )
            ex.gold = GoldSet(requirements=[{p} for p in sorted(real)],
                              tables={t for t, _ in real}, parse_ok=True)
            out.append(ex)
            if log_every and stats["kept"] % log_every == 0:
                print(f"  parsed {stats['kept']} examples, {len(schemas)} dbs", flush=True)
    stats["n_dbs"] = len(schemas)
    stats["avg_columns"] = (sum(e.schema.n_columns() for e in out) / len(out)) if out else 0.0
    stats["avg_gold"] = (sum(len(e.gold.requirements) for e in out) / len(out)) if out else 0.0
    return out, stats


def db_overlap_with_dev(train_dbs: Sequence[Example], dev_dbs: Sequence[str]) -> Set[str]:
    """泄漏自检：train 的库名集合与 BIRD-Dev 的 11 个库是否有交集。"""
    tr = {e.db_id for e in train_dbs}
    return tr & set(dev_dbs)
