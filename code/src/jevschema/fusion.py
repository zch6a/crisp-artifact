"""词法/语义信号与 System-One 风险的融合。

为什么需要它：Spider 的问题里列名经常原词出现，BM25 的 AUROC（0.80）高于
零样本决策模型（0.60~0.71）。把两者按"各自先做样本内归一、再加权求和"融合，
既保留词汇匹配的确定性，也保留决策模型对"同义改写/隐含列"的判断。

归一化必须在**样本内**做：决策模型和 BM25 的绝对分数在不同库之间量纲差异很大，
直接相加等于让量纲大的那一方单独做决定（jev-compass 里也踩过这个坑）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .data_loader import Example
from .summary import bm25_scores
from .paths import MODELS

Pair = Tuple[str, str]


# Some local Hugging Face caches contain a broken (0-byte) config.json;
# 直接用 HF repo id 加载会抛 JSONDecodeError。默认走已验证的本地目录，
# 需要换模型时用环境变量覆盖。
DEFAULT_EMB_MODEL = os.environ.get(
    "JEVSCHEMA_EMB_MODEL", os.path.join(MODELS, "miniLM"))


@dataclass
class FusionConfig:
    lexical: str = "bm25"        # bm25 | embedding | none
    alpha: float = 0.3           # 词法信号权重（0 = 纯决策模型）
    normalize: str = "max"       # max | minmax | rank | none（只作用于词法分数）
    embedding_model: str = DEFAULT_EMB_MODEL


def _normalize(vals: Dict[Pair, float], how: str) -> Dict[Pair, float]:
    if how == "none" or not vals:
        return dict(vals)
    ks = list(vals)
    v = np.asarray([vals[k] for k in ks], dtype=float)
    if how == "max":
        # 除以样本内最大值：0 保持 0。BM25 里 80% 的列得分为 0，
        # 用 rank/percentile 归一会把这片并列值抬到分布中部，
        # 等于把"毫无词汇重叠"伪装成"中等相关"，过滤率会直接归零。
        hi = float(v.max())
        out = v / hi if hi > 0 else np.zeros_like(v)
    elif how == "rank":
        from scipy.stats import rankdata

        r = rankdata(v, method="average") - 1.0
        n = max(1, len(v) - 1)
        out = r / n
    else:  # minmax
        lo, hi = float(v.min()), float(v.max())
        out = (v - lo) / max(1e-9, hi - lo)
    return {k: float(x) for k, x in zip(ks, out)}


_CACHE: Dict[str, object] = {}


def lexical_scores(ex: Example, cfg: FusionConfig) -> Dict[Pair, float]:
    if cfg.lexical == "none":
        return {}
    if cfg.lexical == "both":
        # 两条词法信号各自的量纲不同：BM25 是稀疏词频分（大量 0，用 max 缩放），
        # 余弦相似度是密集窄区间值（用 minmax 拉伸）。先各自归一再平均，
        # 直接相加等于让区间宽的一方单独决定。
        bm = _normalize(_bm25(ex), "max")
        emb = _normalize(_embed(ex, cfg), "minmax")
        return {k: 0.5 * bm.get(k, 0.0) + 0.5 * emb.get(k, 0.0) for k in bm}
    if cfg.lexical == "bm25":
        return _bm25(ex)
    if cfg.lexical == "embedding":
        return _embed(ex, cfg)
    raise ValueError(f"未知词法信号: {cfg.lexical}")


def _bm25(ex: Example) -> Dict[Pair, float]:
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_', ' ')} {c} {c.replace('_', ' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def _embed(ex: Example, cfg: FusionConfig) -> Dict[Pair, float]:
    from sentence_transformers import SentenceTransformer

    if "emb" not in _CACHE:
        _CACHE["emb"] = SentenceTransformer(cfg.embedding_model)
    m = _CACHE["emb"]
    pairs = ex.schema.columns()
    docs = [f"{t.replace('_', ' ')} {c.replace('_', ' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    emb = m.encode([q] + docs, normalize_embeddings=True, show_progress_bar=False)
    sims = emb[1:] @ emb[0]
    return {p: float(s) for p, s in zip(pairs, sims)}


def fuse(ex: Example, risk: Dict[Pair, float], cfg: FusionConfig) -> Dict[Pair, float]:
    """把风险分数与词法分数融合。

    决策模型的输出本身就在 [0,1]（校准概率），所以**原样保留**；
    只对词法分数做样本内缩放。反过来给两边都做归一会出事：BM25 里
    80% 的列得分恰好为 0，rank/percentile 类归一会把这片并列值抬到 0.5 附近，
    等于把"毫无词汇重叠"伪装成"中等相关"，过滤率直接归零。
    """
    if cfg.lexical == "none" or cfg.alpha <= 0:
        return dict(risk)
    lex = lexical_scores(ex, cfg)
    lex_n = _normalize(lex, cfg.normalize)
    out: Dict[Pair, float] = {}
    for k, r in risk.items():
        l = lex_n.get(k)
        out[k] = float(r) if l is None else (1.0 - cfg.alpha) * float(r) + cfg.alpha * l
    return out
