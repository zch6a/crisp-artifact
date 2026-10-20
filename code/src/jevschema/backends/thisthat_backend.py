"""this-that-model-1.0 后端（本地 RTX 5070，$0 成本）。

官方设计：一次 forward，取每个回答槽位（"Answer k: (" 最后一个 token）的 hidden state，
与该题选项的 label token 做内积后 softmax。没有解码循环，所以：
  * 不问"生成什么"，只问"选项里哪个概率高"；
  * 同一 state 下的 N 个判断共享一次 forward —— 对应 Jev-Schema 的
    "一个问题 × 上百个候选列"，这是实验 2 要测的效率优势。

权重位置：models/this-that-model-1.0（env THISTHAT_MODEL_PATH 可覆盖）。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .base import DecisionBackend, TypedQuestion
from ..paths import MODELS

DEFAULT_PATH = os.environ.get("THISTHAT_MODEL_PATH", os.path.join(MODELS, "this-that-model-1.0"))
LETTERS = "ABCDEFGHIJ"


class _PromptBuilder:
    """复刻官方 state_first 布局：Context -> 每题 Options -> Answer k: ( """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._labels: Optional[List[int]] = None

    def label_ids(self) -> List[int]:
        if self._labels is None:
            ids = []
            for ch in LETTERS:
                t = self.tok.encode(ch, add_special_tokens=False)
                if len(t) != 1:
                    raise RuntimeError(f"label {ch!r} 不是单 token: {t}")
                ids.append(t[0])
            self._labels = ids
        return self._labels

    def build(self, state: str, questions: Sequence[TypedQuestion],
              max_state_tokens: int) -> tuple[List[int], List[int], List[int]]:
        tok = self.tok
        multi = len(questions) > 1
        ids = tok.encode("Context:\n" + state, add_special_tokens=False)[:max_state_tokens]
        slots: List[int] = []
        n_options: List[int] = []
        for k, q in enumerate(questions):
            head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:"
            ids += tok.encode(head, add_special_tokens=False)
            ids += tok.encode("".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(q.options[:10])),
                              add_special_tokens=False)
            ids += tok.encode(f"\nAnswer{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False)
            slots.append(len(ids) - 1)
            n_options.append(min(len(q.options), 10))
        return ids, slots, n_options


class ThisThatBackend(DecisionBackend):
    name = "this-that-1.0"
    supports_shared_context_batch = True

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 max_state_tokens: int = 3072, max_questions_per_pass: int = 48,
                 quant: Optional[str] = None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path or DEFAULT_PATH
        self.max_state_tokens = max_state_tokens
        self.max_questions_per_pass = max_questions_per_pass

        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            qcfg.disable_expert_buffers = True
            try:  # transformers 5.x 的 allocator 预热在显存紧张时会自己 OOM，纯优化可跳
                import transformers.modeling_utils as _mu
                if hasattr(_mu, "caching_allocator_warmup"):
                    _mu.caching_allocator_warmup = lambda *a, **k: None
            except Exception:
                pass
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, quantization_config=qcfg, device_map=str(self.dev),
                low_cpu_mem_usage=True)
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, dtype=dtype, device_map=str(self.dev), low_cpu_mem_usage=True)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype)
                self.model.to(self.dev)
        self.model.eval()
        self.builder = _PromptBuilder(self.tok)
        self._label_ids = torch.tensor(self.builder.label_ids(), device=self.dev)

    # ---------------------------------------------------------------- 核心
    @torch.no_grad()
    def _forward(self, state: str, questions: Sequence[TypedQuestion]) -> List[np.ndarray]:
        ids, slots, n_opts = self.builder.build(state, questions, self.max_state_tokens)
        t = torch.tensor([ids], device=self.dev)
        hidden = self.model.model(input_ids=t).last_hidden_state
        h = hidden[0, slots, :]
        w = self.model.lm_head.weight[self._label_ids]
        logits = F.linear(h, w).float()
        mask = torch.arange(len(self._label_ids), device=self.dev)[None, :] >= torch.tensor(
            n_opts, device=self.dev)[:, None]
        logits = logits.masked_fill(mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return [p[:n] for p, n in zip(probs, n_opts)]

    def _decide(self, state: str, questions: Sequence[TypedQuestion]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        step = self.max_questions_per_pass
        for i in range(0, len(questions), step):
            out.extend(self._forward(state, questions[i:i + step]))
        return out

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))
