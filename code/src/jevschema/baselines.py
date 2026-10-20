"""基线（开题报告 §6.2）。

公平比较的关键：**在相同过滤率下比召回**，而不是各报各的 top-k。
因此每个基线只暴露"给每个列打一个分数"，过滤率的扫描统一交给
metrics.recall_at_filter_rate。这样 BM25 / embedding / cross-encoder / Jev-Schema
在同一条横轴上可比。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .data_loader import Example
from .summary import bm25_scores

Pair = Tuple[str, str]


class Baseline:
    name = "baseline"
    scored_per_example: int = 0

    def score(self, ex: Example) -> Dict[Pair, float]:
        raise NotImplementedError


class FullSchemaBaseline(Baseline):
    """Baseline 1/6：全量 schema 直送（工业现状）。过滤率固定为 0。"""

    name = "full-schema"

    def score(self, ex: Example) -> Dict[Pair, float]:
        return {p: 1.0 for p in ex.schema.columns()}


class BM25Baseline(Baseline):
    """Baseline 3：关键词检索，列名 + 表名当文档。"""

    name = "bm25"

    def score(self, ex: Example) -> Dict[Pair, float]:
        pairs = ex.schema.columns()
        docs = [f"{t} {t.replace('_', ' ')} {c} {c.replace('_', ' ')}" for t, c in pairs]
        q = ex.question + (" " + ex.evidence if ex.evidence else "")
        s = bm25_scores(q, docs)
        return dict(zip(pairs, s))


class RandomBaseline(Baseline):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, ex: Example) -> Dict[Pair, float]:
        pairs = ex.schema.columns()
        return {p: float(v) for p, v in zip(pairs, self.rng.random(len(pairs)))}


class EmbeddingBaseline(Baseline):
    """Baseline 2：question/schema embedding 相似度（工业界主流做法）。"""

    name = "embedding"

    def __init__(self, model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self._m = None

    def _load(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_path, device=self.device)
        return self._m

    def score(self, ex: Example) -> Dict[Pair, float]:
        m = self._load()
        pairs = ex.schema.columns()
        q = ex.question + (" " + ex.evidence if ex.evidence else "")
        docs = [f"{t.replace('_', ' ')} {c.replace('_', ' ')}" for t, c in pairs]
        emb = m.encode([q] + docs, normalize_embeddings=True, show_progress_bar=False)
        sims = emb[1:] @ emb[0]
        return {p: float(s) for p, s in zip(pairs, sims)}


class CrossEncoderBaseline(Baseline):
    """Baseline 4：(question, column) 交叉编码器 —— 最主要竞争对手。

    权重来自 scripts/train_cross_encoder.py 在 Spider train 上训练得到的模型；
    没有权重时抛错，不静默退化（静默退化会让人以为 cross-encoder 很弱）。
    """

    name = "cross-encoder"

    def __init__(self, model_path: str, max_len: int = 128, batch_size: int = 64,
                 device: Optional[str] = None):
        self.model_path = model_path
        self.max_len = max_len
        self.batch_size = batch_size
        self._tok = None
        self._model = None
        self._dev = device

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._dev = self._dev or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._dev).eval()
        return self._tok, self._model

    def score(self, ex: Example) -> Dict[Pair, float]:
        import torch
        tok, model = self._load()
        pairs = ex.schema.columns()
        q = ex.question + (" " + ex.evidence if ex.evidence else "")
        texts = [f"{q} [SEP] table {t}, column {c}" for t, c in pairs]
        out: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = tok(batch, truncation=True, max_length=self.max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(self._dev) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            if logits.shape[-1] == 1:
                p = torch.sigmoid(logits.squeeze(-1))
                out.extend([float(x) for x in p])
            else:
                p = torch.softmax(logits, -1)[:, 1]
                out.extend([float(x) for x in p])
        return dict(zip(pairs, out))


def build_baseline(name: str, **kw) -> Baseline:
    name = (name or "bm25").lower()
    if name == "full":
        return FullSchemaBaseline()
    if name == "bm25":
        return BM25Baseline()
    if name == "random":
        return RandomBaseline(kw.get("seed", 0))
    if name == "embedding":
        return EmbeddingBaseline(kw.get("model_path", "sentence-transformers/all-MiniLM-L6-v2"))
    if name in ("cross-encoder", "ce"):
        if not kw.get("model_path"):
            raise ValueError("cross-encoder 需要 model_path（先跑 train_cross_encoder.py）")
        return CrossEncoderBaseline(kw["model_path"])
    raise ValueError(f"未知基线: {name}")


# ----------------------------------------------------------------------------- 统一评测
def evaluate_at_filter_rates(baseline: Baseline, examples: Sequence[Example],
                             filter_rates: Sequence[float]) -> Dict[str, Dict[float, float]]:
    """返回 {filter_rate: mean_recall}；同时给出每例的分数以便后续做显著性检验。"""
    per_rate: Dict[float, List[float]] = {fr: [] for fr in filter_rates}
    for ex in examples:
        if ex.gold is None:
            continue
        scores = baseline.score(ex)
        pairs = list(scores.keys())
        s = np.array([scores[p] for p in pairs], dtype=float)
        gold_flat = ex.gold.flat_pairs()
        y = np.array([1 if p in gold_flat else 0 for p in pairs], dtype=int)
        for fr in filter_rates:
            rec, _thr = _recall_at(s, y, fr)
            per_rate[fr].append(rec)
    return {fr: float(np.mean(v)) for fr, v in per_rate.items()}


def _recall_at(s: np.ndarray, y: np.ndarray, target_filter: float) -> Tuple[float, float]:
    from .metrics import recall_at_filter_rate
    return recall_at_filter_rate(s, y, target_filter)
