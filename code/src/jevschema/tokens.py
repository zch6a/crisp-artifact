"""分阶段 token 计量（论文的"成本维度"）。

必须把 token 拆成两段报，否则本课题的主张无法被验证：
    linking_tokens —— schema linking 阶段（决策模型 state + N 个问题）
    gen_tokens     —— SQL 生成阶段（prompt = 问题 + evidence + 过滤后的 schema）
基线的 74.96K / 123.30K tokens/query 是"linking + 若干轮生成"的总量，
混在一起报等于把自己的优势藏起来（我们砍的是 gen 段）。

口径统一用 cl100k_base（GPT-4 系 tokenizer）。
决策模型侧的"等效 token"也换算成 cl100k —— 不是因为它真的按这个计费，
而是只有统一口径才能和 LLM 侧的成本放在同一张帕累托图上看。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:                                   # pragma: no cover
    _ENC = None


def n_tokens(text: str) -> int:
    """统一 tokenizer。tiktoken 不可用时退化到 chars/4（并显式标注口径已变）。"""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


TOKENIZER = "cl100k_base" if _ENC is not None else "chars/4"


@dataclass
class StageTokens:
    """一次查询的分阶段 token 账。"""
    # --- schema linking 段 ---
    link_state_tokens: int = 0        # 共享 state（问题 + evidence + schema summary）
    link_question_tokens: int = 0     # N 条 typed question 之和（Jev 口径下逐条计费）
    link_llm_calls: int = 0           # 该段调用次数（0 = 决策模型，无 LLM 轮次）
    # --- SQL 生成段 ---
    gen_prompt_tokens: int = 0        # 过滤后 schema 拼进 prompt 的长度
    gen_output_tokens: int = 0
    gen_llm_calls: int = 0
    # --- 参考系 ---
    full_schema_prompt_tokens: int = 0  # 不压缩、全 schema 直喂时的 prompt 长度

    @property
    def linking_tokens(self) -> int:
        return self.link_state_tokens + self.link_question_tokens

    @property
    def gen_tokens(self) -> int:
        return self.gen_prompt_tokens + self.gen_output_tokens

    @property
    def total_tokens(self) -> int:
        return self.linking_tokens + self.gen_tokens

    @property
    def gen_saving_vs_full(self) -> float:
        """相对"全 schema 直喂"的生成段节省比例。分母是生成段，不是总量。"""
        if not self.full_schema_prompt_tokens:
            return 0.0
        return 1.0 - self.gen_prompt_tokens / self.full_schema_prompt_tokens

    def as_dict(self) -> Dict:
        return {
            "linking_tokens": self.linking_tokens,
            "gen_tokens": self.gen_tokens,
            "total_tokens": self.total_tokens,
            "link_state_tokens": self.link_state_tokens,
            "link_question_tokens": self.link_question_tokens,
            "link_llm_calls": self.link_llm_calls,
            "gen_prompt_tokens": self.gen_prompt_tokens,
            "gen_output_tokens": self.gen_output_tokens,
            "gen_calls": self.gen_llm_calls,
            "full_schema_prompt_tokens": self.full_schema_prompt_tokens,
            "gen_saving_vs_full": round(self.gen_saving_vs_full, 4),
            "total_saving_vs_full": round(
                1.0 - self.total_tokens / (self.full_schema_prompt_tokens + self.gen_output_tokens), 4
            ) if self.full_schema_prompt_tokens else 0.0,
            "tokenizer": TOKENIZER,
        }


# ----------------------------------------------------------------------------- 生成端 prompt
GEN_PROMPT_TMPL = """You are a SQLite expert. Given the database schema below, write ONE SQL query that answers the question.

Database: {db_id}
Schema:
{schema}

{evidence}Question: {question}

Return only the SQL query, no explanation.
"""


def build_gen_prompt(db_id: str, schema_text: str, question: str, evidence: str = "") -> str:
    ev = f"Evidence: {evidence}\n" if evidence else ""
    return GEN_PROMPT_TMPL.format(db_id=db_id, schema=schema_text,
                                  question=question, evidence=ev)


def schema_prompt_text(schema, tables: Optional[Sequence[str]] = None,
                       columns: Optional[Sequence[Tuple[str, str]]] = None,
                       with_fk: bool = True, with_samples: int = 0) -> str:
    """把（可能已过滤的）schema 渲染成给生成模型看的 DDL 式文本。

    columns 给定则只输出这些列；否则输出 tables（或全库）的所有列。
    """
    keep_cols = None
    if columns is not None:
        keep_cols = {(t.lower(), c.lower()) for t, c in columns}
    keep_tables = {t.lower() for t in tables} if tables is not None else None
    lines: List[str] = []
    for t in schema.tables:
        if keep_tables is not None and t.name.lower() not in keep_tables:
            continue
        cols = []
        for c in t.columns:
            if keep_cols is not None and (t.name.lower(), c.name.lower()) not in keep_cols:
                continue
            tag = c.dtype or "text"
            if c.is_pk:
                tag += " PRIMARY KEY"
            if with_samples and c.sample_values:
                tag += f" -- e.g. {', '.join(str(v) for v in c.sample_values[:with_samples])}"
            cols.append(f"  {c.name} {tag}")
        if not cols and keep_cols is not None:
            continue
        lines.append(f"CREATE TABLE {t.name} (\n" + ",\n".join(cols) + "\n);")
    if with_fk:
        seen = set()
        for f in schema.fks:
            lt = {x.lower() for x in schema.table_names()}
            if keep_tables is not None:
                if f.src_table.lower() not in keep_tables or f.dst_table.lower() not in keep_tables:
                    continue
            key = (f.src_table.lower(), f.src_col.lower(), f.dst_table.lower(), f.dst_col.lower())
            if key in seen or f.src_table.lower() not in lt:
                continue
            seen.add(key)
            lines.append(f"-- FK: {f.src_table}.{f.src_col} -> {f.dst_table}.{f.dst_col}")
    return "\n".join(lines)


def make_stage_tokens(schema, question: str, evidence: str = "",
                      kept_columns: Optional[Sequence[Tuple[str, str]]] = None,
                      state_text: str = "", questions: Sequence[str] = (),
                      output_sql: str = "", with_samples: int = 0) -> StageTokens:
    """把一次查询的两段 token 一次性算清。kept_columns=None 表示"全 schema 基线"。"""
    if kept_columns is None:
        schema_txt = schema_prompt_text(schema, with_samples=with_samples)
        gen_prompt = build_gen_prompt(schema.db_id, schema_txt, question, evidence)
        return StageTokens(
            gen_prompt_tokens=n_tokens(gen_prompt),
            gen_output_tokens=n_tokens(output_sql),
            gen_llm_calls=1,
            full_schema_prompt_tokens=n_tokens(gen_prompt),
        )
    # 过滤后：只保留至少有一列入选的表
    tables = sorted({t for t, _ in kept_columns})
    schema_txt = schema_prompt_text(schema, tables=tables, columns=kept_columns,
                                    with_samples=with_samples)
    gen_prompt = build_gen_prompt(schema.db_id, schema_txt, question, evidence)
    full_txt = schema_prompt_text(schema, with_samples=with_samples)
    full_prompt = build_gen_prompt(schema.db_id, full_txt, question, evidence)
    return StageTokens(
        link_state_tokens=n_tokens(state_text),
        link_question_tokens=sum(n_tokens(q) for q in questions),
        gen_prompt_tokens=n_tokens(gen_prompt),
        gen_output_tokens=n_tokens(output_sql),
        gen_llm_calls=1,
        full_schema_prompt_tokens=n_tokens(full_prompt),
    )
