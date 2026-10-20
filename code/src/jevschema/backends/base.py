"""决策后端契约。

所有"System-One 判断"都收敛到一个接口：

    decide(state, questions) -> List[Decision]

state 是共享上下文（问题 + schema summary），questions 是一批 typed question。
一次调用 = 一次（或几次）forward，N 个问题并行 —— 这正是 Jev / this-that / laya
相对 cross-encoder 的效率优势来源，也是实验 2 要验证的东西。

契约之外的东西一律放在 stats 里：调用次数、question 次数、state token、question token、
墙钟延迟。成本模型只依赖 stats，不依赖具体后端，这样换后端不改实验代码。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TypedQuestion:
    text: str
    options: Tuple[str, ...] = ("no", "yes")
    qtype: str = "noul"          # noul | choice | score
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.options) < 2:
            raise ValueError("need >= 2 options")


@dataclass
class Decision:
    probs: np.ndarray            # shape [n_options], 已 softmax，顺序 = question.options
    qtype: str = "noul"

    @property
    def p_yes(self) -> float:
        """noul/二值题取 P(第二个选项)；多选项题取 1 - P(第一个选项)。"""
        if len(self.probs) == 2:
            return float(self.probs[1])
        return float(1.0 - self.probs[0])

    @property
    def confidence(self) -> float:
        """最大概率作为置信度（用于 uncertainty routing 的辅助信号）。"""
        return float(np.max(self.probs))


class Stats:
    """线程安全的计数器。成本/延迟/帕累托的全部原始数据来源。"""

    def __init__(self):
        self.calls = 0
        self.n_questions = 0
        self.state_tokens = 0
        self.question_tokens = 0
        self.latency_s = 0.0
        self.extra: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record(self, n_questions: int, latency: float, state_tokens: int = 0,
               question_tokens: int = 0, **extra):
        with self._lock:
            self.calls += 1
            self.n_questions += n_questions
            self.latency_s += latency
            self.state_tokens += state_tokens
            self.question_tokens += question_tokens
            for k, v in extra.items():
                self.extra[k] = self.extra.get(k, 0.0) + float(v)

    def as_dict(self) -> Dict:
        return {
            "calls": self.calls,
            "n_questions": self.n_questions,
            "state_tokens": self.state_tokens,
            "question_tokens": self.question_tokens,
            "latency_s": round(self.latency_s, 3),
            **{f"extra_{k}": v for k, v in self.extra.items()},
        }


class DecisionBackend:
    """后端基类。子类实现 `_decide`。"""

    name = "base"
    supports_shared_context_batch = True
    max_state_tokens = 2048
    max_questions_per_pass = 64

    def __init__(self):
        self.stats = Stats()

    # ---------------------------------------------------------------- 子类实现
    def _decide(self, state: str, questions: Sequence[TypedQuestion]) -> List[np.ndarray]:
        raise NotImplementedError

    # ---------------------------------------------------------------- 公共 API
    def decide(self, state: str, questions: Sequence[TypedQuestion]) -> List[Decision]:
        qs = list(questions)
        if not qs:
            return []
        t0 = time.time()
        probs = self._decide(state, qs)
        self.stats.record(len(qs), time.time() - t0,
                          state_tokens=self.n_tokens(state),
                          question_tokens=sum(self.n_tokens(q.text) for q in qs))
        return [Decision(p, q.qtype) for p, q in zip(probs, qs)]

    def judge_batch(self, state: str, statements: Sequence[str]) -> List[float]:
        """便捷接口：一批 yes/no 陈述 -> P(命题为真)。"""
        qs = [TypedQuestion(s, ("no", "yes"), "noul") for s in statements]
        return [d.p_yes for d in self.decide(state, qs)]

    # ---------------------------------------------------------------- 辅助
    def n_tokens(self, text: str) -> int:
        """没有 tokenizer 的后端按 4 字符/token 估算（只用于成本模型，不用于截断）。"""
        return max(1, len(text) // 4)

    def reset_stats(self):
        self.stats = Stats()


class CachedBackend(DecisionBackend):
    """按 (backend, state, question) 精确缓存的判断后端。

    存在的唯一理由：付费 API 绝不允许因为重跑脚本而重复计费；
    消融实验之间也必须严格可比。key 用 sha1，避免长 state 撑爆内存。
    """

    def __init__(self, inner: DecisionBackend, cache_path: str):
        super().__init__()          # 必须有 stats：外层同样要能被成本模型读
        self.inner = inner
        self.name = inner.name      # 成本报表按真实后端归类，不按"cached"
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        self.hits = 0
        self.misses = 0
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["p"]
            except Exception as e:
                print(f"[warn] 缓存读取失败，按空缓存继续: {e}")
        self._fh = open(cache_path, "a", encoding="utf-8")

    @property
    def supports_shared_context_batch(self):  # type: ignore[override]
        return self.inner.supports_shared_context_batch

    @property
    def inner_stats(self):
        return self.inner.stats

    def n_tokens(self, text: str) -> int:     # 用真实后端的 tokenizer 计数
        return self.inner.n_tokens(text)

    @staticmethod
    def _key(backend: str, state: str, q: TypedQuestion) -> str:
        raw = json.dumps([backend, state, q.text, list(q.options), q.qtype], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _decide(self, state: str, questions: Sequence[TypedQuestion]) -> List[np.ndarray]:
        out: List[Optional[np.ndarray]] = [None] * len(questions)
        pending: List[int] = []
        for i, q in enumerate(questions):
            k = self._key(self.inner.name, state, q)
            if k in self.cache:
                out[i] = np.asarray(self.cache[k], dtype=float)
                self.hits += 1
            else:
                pending.append(i)
        if pending:
            self.misses += len(pending)
            # 走 public decide：token/延迟统计记在 inner 上，缓存命中则完全不记成本
            got = self.inner.decide(state, [questions[i] for i in pending])
            for i, d in zip(pending, got):
                p = np.asarray(d.probs, dtype=float)
                out[i] = p
                k = self._key(self.inner.name, state, questions[i])
                self.cache[k] = p.tolist()
                self._fh.write(json.dumps({"k": k, "p": p.tolist()}, ensure_ascii=False) + "\n")
            self._fh.flush()
        # stats 记在 inner 上（真实发生的调用），外层只记缓存命中数
        return [o for o in out]  # type: ignore[misc]

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass


class MockBackend(DecisionBackend):
    """占位后端：全部返回 0.5。只用于冒烟测试，禁止用于任何结论。"""

    name = "mock"
    supports_shared_context_batch = False

    def _decide(self, state, questions):
        return [np.full(len(q.options), 1.0 / len(q.options)) for q in questions]


class RandomBackend(DecisionBackend):
    """随机基线：给"路由到底有没有用"提供下限参照。"""

    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def _decide(self, state, questions):
        out = []
        for q in questions:
            p = self.rng.dirichlet(np.ones(len(q.options)))
            out.append(p)
        return out
