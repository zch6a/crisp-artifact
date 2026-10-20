"""BIRD-Dev 装载：从官方 tables.json + sqlite 构造 Example 流。

为什么必须切到 BIRD：本课题要对标的 AutoLink / EviLink / LinkAlign / EllieSQL / BAP-SQL
全部在 BIRD-Dev（或 Spider 2.0-Lite）上报数，而原先的 Spider dev 平均只有 27 列，
与之并排放进一张表属于口径错误。BIRD-Dev 11 个库、11~199 列（均值 ~73），
且带 `evidence`（外部知识），是唯一能和这批基线对齐的公开集。

数据源（下载一次，放到 data/ 下）：
    data/raw/bird/dev.json            1534 题，含 gold SQL / evidence / difficulty
    data/raw/bird/dev_tables.json      11 个库的 schema（Spider 格式）
    data/bird_dev/dev_databases/<db>/<db>.sqlite   真实库（执行 EX 用）
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .data_loader import Example
from .gold_labels import GoldSet, extract_gold
from .schema_graph import Column, ForeignKey, SchemaGraph, Table
from .paths import DATA, BIRD_DB_DIR

BIRD_DIR = os.environ.get("JEVSCHEMA_BIRD", os.path.join(DATA, "raw", "bird"))
BIRD_SQLITE_DIR = os.environ.get("JEVSCHEMA_BIRD_SQLITE", BIRD_DB_DIR)


# ----------------------------------------------------------------------------- schema
def schema_from_tables_row(row: Dict) -> SchemaGraph:
    """解析 BIRD `dev_tables.json` 的一行（Spider 风格的 tables.json）。

    用 `*_original` 而非 `*_names`：后者是给人看的带空格别名（"gas stations"），
    gold SQL 里写的是原名。用别名会导致 gold 解析大面积 unresolve。
    """
    db_id = str(row["db_id"])
    tnames = list(row["table_names_original"])
    tables: List[Table] = [Table(name=t) for t in tnames]

    ctypes = list(row.get("column_types", []))
    for idx, (t_idx, cname) in enumerate(row["column_names_original"]):
        if t_idx < 0:            # [-1, "*"]
            continue
        if t_idx >= len(tables):
            continue
        dtype = ctypes[idx] if idx < len(ctypes) else "text"
        tables[t_idx].columns.append(Column(name=str(cname), dtype=str(dtype)))

    # primary_keys 可能是单个索引，也可能是复合主键的索引列表
    for pk in row.get("primary_keys", []):
        for idx in (pk if isinstance(pk, list) else [pk]):
            if not isinstance(idx, int) or idx >= len(row["column_names_original"]):
                continue
            t_idx, cname = row["column_names_original"][idx]
            if t_idx < 0 or t_idx >= len(tables):
                continue
            for c in tables[t_idx].columns:
                if c.name == cname:
                    c.is_pk = True

    fks: List[ForeignKey] = []
    col_of = {i: (row["column_names_original"][i][0], row["column_names_original"][i][1])
              for i in range(len(row["column_names_original"]))}
    for a, b in row.get("foreign_keys", []):
        if a not in col_of or b not in col_of:
            continue
        (st, sc), (dt, dc) = col_of[a], col_of[b]
        if st < 0 or dt < 0:
            continue
        fks.append(ForeignKey(tnames[st], sc, tnames[dt], dc))
    return SchemaGraph(db_id=db_id, tables=tables, fks=fks)


def load_bird_schemas(path: Optional[str] = None) -> Dict[str, SchemaGraph]:
    path = path or os.path.join(BIRD_DIR, "dev_tables.json")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["db_id"]: schema_from_tables_row(r) for r in rows}


def enrich_sample_values(
    schemas: Dict[str, SchemaGraph],
    sqlite_dir: str = BIRD_SQLITE_DIR,
    k: int = 3,
) -> int:
    """从真实 sqlite 里抽每列前 k 个 distinct 值，回填到 schema。

    BIRD 的列名经常是缩略/歧义的（`Free Meal Count (K-12)`），只看名字很难判断
    语义；示例值是官方评测里 `evidence` 之外最有用的消歧信号。
    返回成功回填的库数（读不到 sqlite 就跳过，不阻断流程）。
    """
    ok = 0
    for db_id, g in schemas.items():
        path = os.path.join(sqlite_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(path):
            continue
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            con.text_factory = lambda b: b.decode("utf-8", "ignore")
            for t in g.tables:
                for c in t.columns:
                    try:
                        cur = con.execute(
                            f'SELECT DISTINCT "{c.name}" FROM "{t.name}" '
                            f'WHERE "{c.name}" IS NOT NULL LIMIT {k}'
                        )
                        vals = [str(r[0])[:40] for r in cur.fetchall()]
                    except Exception:
                        vals = []
                    c.sample_values = vals
            con.close()
            ok += 1
        except Exception:
            continue
    return ok


# ----------------------------------------------------------------------------- questions
def load_bird_dev(
    limit: Optional[int] = None,
    schemas: Optional[Dict[str, SchemaGraph]] = None,
    seed: int = 0,
    shuffle: bool = False,
    with_gold: bool = True,
    db_filter: Optional[Sequence[str]] = None,
) -> List[Example]:
    """BIRD-Dev 1534 题。默认 shuffle=False —— 与 Spider 侧口径一致：
    抽样必须是确定前缀，否则跨脚本的样本集对不上，显著性检验就失效了。"""
    if schemas is None:
        schemas = load_bird_schemas()
    path = os.path.join(BIRD_DIR, "dev.json")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not shuffle:
        rows = list(rows)
    else:
        rows = list(rows)
        random.Random(seed).shuffle(rows)
    out: List[Example] = []
    for r in rows:
        db = str(r["db_id"])
        if db_filter and db not in set(db_filter):
            continue
        sch = schemas.get(db)
        if sch is None:
            continue
        ex = Example(
            ex_id=f"bird-{r.get('question_id')}",
            db_id=db,
            question=str(r["question"]),
            sql=str(r["SQL"]),
            schema=sch,
            evidence=str(r.get("evidence", "") or ""),
            difficulty=str(r.get("difficulty", "") or ""),
        )
        if with_gold:
            ex.gold = extract_gold(ex.sql, sch)
        out.append(ex)
        if limit and len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------------- 统计
def schema_stats(schemas: Dict[str, SchemaGraph]) -> List[Dict]:
    rows = []
    for db, g in sorted(schemas.items()):
        rows.append({
            "db_id": db,
            "n_tables": g.n_tables(),
            "n_columns": g.n_columns(),
            "n_fks": len(g.fks),
        })
    return rows


def gold_quality(examples: Sequence[Example]) -> Dict[str, float]:
    """gold 解析质量自检：BIRD 的 SQL 比 Spider 复杂（CTE、嵌套、反引号列名），
    未解析率太高说明 gold 口径不可信，必须先修再报数。"""
    n = len(examples) or 1
    amb = sum(1 for e in examples if e.gold and e.gold.ambiguous)
    unr = sum(1 for e in examples if e.gold and e.gold.unresolved)
    empty = sum(1 for e in examples if e.gold is not None and not e.gold.requirements)
    ok = sum(1 for e in examples if e.gold and e.gold.parse_ok)
    return {
        "n": float(len(examples)),
        "parse_ok_rate": ok / n,
        "ambiguous_rate": amb / n,
        "unresolved_rate": unr / n,
        "empty_requirement_rate": empty / n,
        "avg_requirements": sum(len(e.gold.requirements) for e in examples if e.gold) / n,
        "avg_gold_tables": sum(len(e.gold.tables) for e in examples if e.gold) / n,
    }
