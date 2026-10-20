"""System-Two 端的 SQL 生成器（本地 ollama，$0，可复现）。

端到端 EX 必须有一个**固定的生成器**当控制变量：论文比的是"schema 给得对不对、
给得省不省"，不是"谁家的 LLM 更强"。所以这里刻意用本地开源模型并锁死
temperature=0、固定 prompt 模板，所有被比较的 schema linking 方法共用同一个
生成器与同一套解码参数。

绝对 EX 会低于 GPT-4 系基线（7B 在 BIRD 上就是这个水平），这一点必须在论文里
明说；可比的是**同一生成器下各 linking 方法的相对差**，以及 token-EX 帕累托前沿。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@dataclass
class GenStats:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    errors: int = 0

    def as_dict(self) -> Dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "latency_s": round(self.latency_s, 2), "errors": self.errors,
                "latency_per_call_s": round(self.latency_s / self.calls, 2) if self.calls else 0.0}


class OllamaGenerator:
    def __init__(self, model: str = "qwen2.5-coder:7b", url: str = OLLAMA_URL,
                 temperature: float = 0.0, num_predict: int = 512,
                 timeout_s: float = 180.0, keep_alive: str = "10m"):
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.stats = GenStats()
        self.last_error: Optional[str] = None

    def _post(self, prompt: str, num_predict: Optional[int] = None) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": int(num_predict or self.num_predict),
                "stop": ["\n\n\n", ";END"],
            },
        }
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            payload = json.loads(r.read().decode("utf-8"))
        # ⚠️ ollama 加载失败时是 **HTTP 200 + {"error": "..."}**，不是抛异常。
        # `.get("response","")` 会把它静默变成空串，于是失败被当成"模型回答了空"，
        # 一路传到缓存里 —— 实测把 989 条垃圾写进 gen_*.jsonl 并永久污染后续所有重跑。
        if payload.get("error"):
            raise RuntimeError(f"ollama: {str(payload['error'])[:300]}")
        return payload.get("response", "")

    def generate(self, prompt: str, num_predict: Optional[int] = None) -> str:
        t0 = time.time()
        try:
            out = self._post(prompt, num_predict)
        except Exception as e:                      # noqa: BLE001
            self.stats.errors += 1
            self.stats.calls += 1
            self.stats.latency_s += time.time() - t0
            self.last_error = f"{type(e).__name__}: {e}"
            # ⚠️ 不能返回 "-- ERROR xxx" 这种看起来像 SQL 的字符串：调用方会把它
            # 当成一条（失败的）生成结果**缓存**起来，之后再跑永远命中它。
            # 失败就必须是空串，让调用方有能力识别并拒绝缓存。
            return ""
        self.stats.calls += 1
        self.stats.latency_s += time.time() - t0
        return out

    def generate_sql(self, prompt: str) -> str:
        return extract_sql(self.generate(prompt))


# ----------------------------------------------------------------------------- SQL 抽取
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """从模型回复里抠出第一条 SQL。

    本地小模型经常先写一段解释再给 SQL，或者给多条候选。
    取 ```sql 代码块优先，否则取第一个 SELECT/WITH 开头的语句，
    到分号为止（没有分号就到末尾）。
    """
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    # 去掉前置解释行
    lines = text.split("\n")
    start = 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b", ln, re.I):
            start = i
            break
    body = "\n".join(lines[start:]).strip()
    if ";" in body:
        body = body[: body.index(";") + 1]
    return body.strip().rstrip(";").strip()
