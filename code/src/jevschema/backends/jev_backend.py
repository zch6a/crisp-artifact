"""真实 Jev API 后端（付费，默认 dry-run）。

硬约束（与本项目一贯的付费纪律一致）：
1. 密钥只从 configs/jev_api.json 或环境变量读，**永远不打印**；
2. 未显式给 `--live` 时只 dry-run（打印将要发的请求，不发）；
3. 强制套 CachedBackend，重跑脚本绝不重复计费；
4. 每次调用前由调用方打印预估条数与费用，获批后小批执行。

计费口径（官方）：输入 $0.042 / 1M tokens，输出免费；一次请求共享 state，
N 个问题只付一次 state tokens。因此成本模型必须把 state/question 分开记，
否则会严重高估 —— 这也是本后端把 tokens 拆成两份统计的原因。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import DecisionBackend, TypedQuestion

PRICE_IN_PER_MTOK = 0.042      # USD / 1M input tokens
PRICE_OUT_PER_MTOK = 0.0       # 官方：输出免费

CONFIG_CANDIDATES = [
    os.environ.get("JEVSCHEMA_JEV_CONFIG", ""),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "..", "..", "configs", "jev_api.json"),
]


def _load_cfg() -> Dict[str, Any]:
    for p in CONFIG_CANDIDATES:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


class JevBackend(DecisionBackend):
    name = "jev-api"
    supports_shared_context_batch = True
    max_state_tokens = 8192

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, dry_run: bool = True,
                 max_questions_per_pass: int = 64, timeout: int = 60):
        super().__init__()
        cfg = _load_cfg()
        self.endpoint = endpoint or cfg.get("endpoint") or os.environ.get("JEV_ENDPOINT", "")
        self.api_key = api_key or cfg.get("api_key") or os.environ.get("JEV_API_KEY", "")
        self.model = model or cfg.get("model") or os.environ.get("JEV_MODEL", "")
        self.dry_run = dry_run
        self.max_questions_per_pass = max_questions_per_pass
        self.timeout = timeout
        self.protocol = cfg.get("protocol", "systemone")
        self.spent_usd = 0.0
        if not self.endpoint:
            raise ValueError("Jev endpoint 未配置（configs/jev_api.json 或 JEV_ENDPOINT）")

    # ---------------------------------------------------------------- 请求构造
    def _payload(self, state: str, questions: Sequence[TypedQuestion]) -> Dict[str, Any]:
        return {
            "model": self.model,
            "state": state,
            "questions": [
                {"text": q.text, "type": q.qtype, "options": list(q.options)} for q in questions
            ],
        }

    def _post(self, payload: Dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse(resp: Any, questions: Sequence[TypedQuestion]) -> List[np.ndarray]:
        """尽力解析；解析不出来就抛错，绝不静默返回 0.5 —— 静默会污染实验结论。"""
        if isinstance(resp, dict):
            arr = resp.get("answers") or resp.get("results") or resp.get("decisions")
        else:
            arr = resp
        if not isinstance(arr, list) or len(arr) < len(questions):
            raise RuntimeError(f"Jev 返回无法解析: {str(resp)[:300]}")
        out = []
        for a, q in zip(arr, questions):
            if isinstance(a, dict):
                probs = a.get("probs") or a.get("distribution") or a.get("p")
                if probs is None and "answer" in a:
                    idx = list(q.options).index(a["answer"]) if a["answer"] in q.options else 0
                    probs = [0.0] * len(q.options)
                    probs[idx] = 1.0
            else:
                probs = a
            p = np.asarray(probs, dtype=float)
            if p.ndim != 1 or p.size == 0:
                raise RuntimeError(f"Jev 单条返回异常: {a}")
            p = p[: len(q.options)]
            out.append(p / p.sum())
        return out

    # ---------------------------------------------------------------- 核心
    def _decide(self, state: str, questions: Sequence[TypedQuestion]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        step = self.max_questions_per_pass
        for i in range(0, len(questions), step):
            chunk = questions[i:i + step]
            payload = self._payload(state, chunk)
            if self.dry_run:
                print(f"[dry-run] would POST {len(chunk)} questions, "
                      f"state_chars={len(state)}, est_cost=${self.estimate_cost_usd(state, chunk):.6f}")
                out.extend([np.full(len(q.options), 1.0 / len(q.options)) for q in chunk])
                continue
            t0 = time.time()
            try:
                resp = self._post(payload)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")[:300]
                raise RuntimeError(f"Jev HTTP {e.code}: {body}") from None
            out.extend(self._parse(resp, chunk))
            cost = self.estimate_cost_usd(state, chunk)
            self.spent_usd += cost
            self.stats.extra["usd"] = self.stats.extra.get("usd", 0.0) + cost
            self.stats.extra["api_latency_s"] = self.stats.extra.get("api_latency_s", 0.0) + (time.time() - t0)
        return out

    # ---------------------------------------------------------------- 计费
    @staticmethod
    def estimate_cost_usd(state: str, questions: Sequence[TypedQuestion],
                          chars_per_token: int = 4) -> float:
        n_state = max(1, len(state) // chars_per_token)
        n_q = sum(max(1, (len(q.text) + sum(len(o) for o in q.options)) // chars_per_token)
                  for q in questions)
        return (n_state + n_q) / 1_000_000 * PRICE_IN_PER_MTOK
