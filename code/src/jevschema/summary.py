"""Schema Summary 构造（开题报告 §5.2）。

两件事：
1. 摘要压缩：只保留表名/列名/类型/PK/FK，丢掉索引、约束、默认值等；
2. 预算控制：state 超出 token 预算时，先做表级粗筛（BM25 / embedding / 无），
   再对入选表做完整摘要。这是 Jev 在 5000+ 列上还能用的前提。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .schema_graph import SchemaGraph


# ----------------------------------------------------------------------------- 粗筛
_SUFFIXES = ("ization", "ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s")


def _stem(w: str) -> str:
    """极简词干化：Spider 的问题里名词常带复数/动名词（singers / singer），
    列名却是单数（singer）。不做词干化，BM25 会把这类明显匹配判成 0 分。"""
    if len(w) > 4:
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
    return w


def bm25_scores(question: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """极小 BM25 实现：不依赖 rank_bm25，行为可控（小写 + 蛇形拆分 + 词干化）。"""
    import math
    import re

    def tok(s: str) -> List[str]:
        out: List[str] = []
        for raw in re.findall(r"[a-z0-9_]+", s.lower()):
            out.extend(p for p in raw.split("_") if p)
        return [_stem(p) for p in out]

    q = tok(question)
    dl = [len(tok(d)) for d in docs]
    avgdl = (sum(dl) / len(dl)) if dl else 1.0
    df: Dict[str, int] = {}
    tfs: List[Dict[str, int]] = []
    for d in docs:
        tf: Dict[str, int] = {}
        for t in tok(d):
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)
        for t in tf:
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    out = []
    for tf, L in zip(tfs, dl):
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            denom = tf[t] + k1 * (1 - b + b * L / max(avgdl, 1e-9))
            s += idf * tf[t] * (k1 + 1) / max(denom, 1e-9)
        out.append(s)
    return out


def prefilter_tables(schema: SchemaGraph, question: str, keep_tables: int) -> List[str]:
    """BM25 粗筛：把候选表砍到 keep_tables 个。永远保留 gold 无关性未知的兜底 ——
    粗筛是**效率**手段，不是过滤手段，因此它的输出只影响"进不进 System-One 打分"，
    被粗筛掉的表仍会被记为 dropped（计入过滤率）。
    """
    docs = [f"{t.name} " + " ".join(c.name for c in t.columns) for t in schema.tables]
    scores = bm25_scores(question, docs)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    return [schema.tables[i].name for i in order[:keep_tables]]


# ----------------------------------------------------------------------------- 摘要
@dataclass
class SummaryConfig:
    max_state_tokens: int = 2400
    chars_per_token: int = 4
    with_fk: bool = True
    with_samples: int = 0
    prefilter: str = "bm25"          # bm25 | none
    prefilter_keep_tables: int = 40


def build_state(question_block: str, schema: SchemaGraph, cfg: SummaryConfig,
                tables: Optional[Sequence[str]] = None) -> tuple[str, List[str]]:
    """返回 (state 文本, 实际进摘要的表名列表)。"""
    if tables is None:
        if cfg.prefilter == "none" or schema.n_tables() <= cfg.prefilter_keep_tables:
            tables = schema.table_names()
        else:
            tables = prefilter_tables(schema, question_block, cfg.prefilter_keep_tables)
    tables = list(tables)
    body = schema.summary(tables=tables, with_fk=cfg.with_fk, with_samples=cfg.with_samples)
    budget_chars = cfg.max_state_tokens * cfg.chars_per_token - len(question_block) - 64

    # 摘要超预算：按"每表列数从少到多"逐表削减列数，而不是整表丢掉 ——
    # 整表丢掉会直接抹掉 gold 表，削减列数至少保留表名线索。
    if len(body) > budget_chars and tables:
        # 粗略估计：每张表一行大约 60 字符/列 -> 反推每表保留多少列
        per_table = max(4, (budget_chars // max(1, len(tables))) // 60)
        body = schema.summary(tables=tables, with_fk=cfg.with_fk, max_cols_per_table=per_table)
    state = (
        "You are helping to decide which parts of a database schema are needed to answer a "
        "question with SQL.\n"
        f"{question_block}\n"
        "Database schema:\n"
        f"{body}\n"
    )
    return state, tables
