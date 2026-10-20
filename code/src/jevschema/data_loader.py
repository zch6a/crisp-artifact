"""数据集装载：把 Spider / BIRD / Schema-Scaling 统一成 Example 流。

Example 是流水线唯一的数据契约：
    db_id, question, evidence, sql, schema(SchemaGraph), gold(GoldSet)

`evidence` 只有 BIRD 有；Spider 为空串。写 prompt 时统一走 `question_block()`。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .gold_labels import GoldSet, extract_gold
from .schema_graph import SchemaGraph, load_spider_schemas, merge_schemas
from .paths import DATA

# 数据落在 D 盘的模型/数据盘，避免 C 盘吃紧
DATA_DIR = os.environ.get("JEVSCHEMA_DATA", DATA)
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


@dataclass
class Example:
    ex_id: str
    db_id: str
    question: str
    sql: str
    schema: SchemaGraph
    evidence: str = ""
    difficulty: str = ""
    gold: Optional[GoldSet] = None

    def question_block(self) -> str:
        if self.evidence:
            return f"Question: {self.question}\nEvidence: {self.evidence}"
        return f"Question: {self.question}"


# ----------------------------------------------------------------------------- Spider
def load_spider_split(
    parquet_path: str,
    schemas: Dict[str, SchemaGraph],
    limit: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = False,
    with_gold: bool = True,
) -> List[Example]:
    df = pd.read_parquet(parquet_path)
    rows = df.to_dict("records")
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    out: List[Example] = []
    for i, r in enumerate(rows):
        db = str(r["db_id"])
        sch = schemas.get(db)
        if sch is None:
            continue
        ex = Example(
            ex_id=f"spider-{i}",
            db_id=db,
            question=str(r["question"]),
            sql=str(r["query"]),
            schema=sch,
        )
        if with_gold:
            ex.gold = extract_gold(ex.sql, sch)
        out.append(ex)
    return out


def load_spider(limit: Optional[int] = None, split: str = "dev", **kw) -> List[Example]:
    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    path = os.path.join(RAW_DIR, f"spider_{'val' if split == 'dev' else 'train'}.parquet")
    return load_spider_split(path, schemas, limit=limit, **kw)


# ----------------------------------------------------------------------------- BIRD
def load_bird_mini(
    limit: Optional[int] = None,
    schemas: Optional[Dict[str, SchemaGraph]] = None,
    **kw,
) -> List[Example]:
    """BIRD mini-dev（sqlite 变体）。

    注意：官方发布包里的数据库文件与 schema 走 Google Drive / OSS，本机网络拿不到，
    因此 schemas 需要外部提供。缺 schema 时抛明确错误，而不是静默返回空列表 ——
    静默空列表会让下游实验"跑通了但样本数 0"，这种伪成功最难查。
    """
    path = os.path.join(RAW_DIR, "bird_mini_dev_sqlite.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"BIRD mini-dev 未下载: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if schemas is None:
        raise ValueError("BIRD 需要显式提供 schemas（db_id -> SchemaGraph）")
    out: List[Example] = []
    for r in rows[: limit or len(rows)]:
        sch = schemas.get(str(r["db_id"]))
        if sch is None:
            continue
        ex = Example(
            ex_id=f"bird-{r.get('question_id')}",
            db_id=str(r["db_id"]),
            question=str(r["question"]),
            sql=str(r["SQL"]),
            schema=sch,
            evidence=str(r.get("evidence", "")),
            difficulty=str(r.get("difficulty", "")),
        )
        ex.gold = extract_gold(ex.sql, sch)
        out.append(ex)
    return out


# ----------------------------------------------------------------------------- Schema Scaling
def build_scaling_benchmark(
    examples: Sequence[Example],
    schemas: Dict[str, SchemaGraph],
    target_columns: int,
    seed: int = 0,
    with_gold: bool = True,
) -> List[Example]:
    """把每个问题的目标库塞进一个"企业级大库"里，构造开题报告 §6.1 的 scaling benchmark。

    做法：以目标库为种子，随机并入其它库的表直到列数达到 target_columns。
    并入的表是纯干扰项（distractor），gold 不变 —— 这正好复现"企业库里
    99% 的列与当前问题无关"的场景。
    """
    rng = random.Random(seed)
    all_dbs = sorted(schemas.keys())
    out: List[Example] = []
    for ex in examples:
        base = ex.schema
        parts = [base]
        n_cols = base.n_columns()
        pool = [d for d in all_dbs if d != base.db_id]
        rng.shuffle(pool)
        for d in pool:
            if n_cols >= target_columns:
                break
            g = schemas[d]
            parts.append(g)
            n_cols += g.n_columns()
        merged = merge_schemas(f"{base.db_id}__scaled{target_columns}", parts, prefix=True)
        # gold SQL 里的表名带上了库名前缀，重抽一次 gold
        new_sql = _prefix_sql_tables(ex.sql, base.db_id, base)
        new_ex = Example(
            ex_id=f"{ex.ex_id}-s{target_columns}",
            db_id=merged.db_id,
            question=ex.question,
            sql=new_sql,
            schema=merged,
            evidence=ex.evidence,
            difficulty=ex.difficulty,
        )
        if with_gold:
            new_ex.gold = extract_gold(new_sql, merged)
        out.append(new_ex)
    return out


def _prefix_sql_tables(sql: str, db_id: str, base: SchemaGraph) -> str:
    """给 gold SQL 里的表名加库名前缀，使其与 merged schema 对齐。"""
    names = sorted({t.name for t in base.tables}, key=len, reverse=True)
    out = sql
    for t in names:
        out = re_table(out, t, f"{db_id}_{t}")
    return out


import re  # noqa: E402  (放在函数之后仅供 _prefix_sql_tables 使用，保持上面可读性)


def re_table(sql: str, old: str, new: str) -> str:
    pattern = re.compile(rf"(?<![\w.]){re.escape(old)}(?![\w])")
    return pattern.sub(new, sql)


# ----------------------------------------------------------------------------- 缓存
def save_examples(examples: Sequence[Example], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"ex_id": ex.ex_id, "db_id": ex.db_id,
                                "question": ex.question, "sql": ex.sql,
                                "evidence": ex.evidence}, ensure_ascii=False) + "\n")
