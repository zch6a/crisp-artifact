"""Laya 后端：ModernBERT-large 编码器 + Jev 风格的 typed decision head（开源，$0）。

Laya 与 Jev 同源的设计：noul / choice / score 三种题型，每个选项一个 [MASK] marker，
head 对 marker 打分后 softmax，没有解码循环。官方配置里带一组 post-hoc 拟合的
temperature（按题型/选项数），用它做 temperature scaling 之后概率才是校准过的 ——
这一点对开题报告"实验 1：校准"是硬需求，所以这里默认启用。

权重：models/laya（model.safetensors 842MB + encoder/config.json + tokenizer），
推理代码复用官方 rl_common.py。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .base import DecisionBackend, TypedQuestion
from ..paths import MODELS

DEFAULT_PATH = os.environ.get("LAYA_MODEL_PATH", os.path.join(MODELS, "laya"))
QTYPES = {"choice": 0, "score": 1, "noul": 2}


def stream_load_state_dict(model, path: str, strict: bool = False) -> int:
    """逐张量读取 safetensors 直接写进模型参数。

    为什么不用 `load_file` + `load_state_dict`：那会先在内存里物化一整份
    与文件等大的 state dict，等于把峰值内存翻倍。本机提交内存只剩 ~4.6GB 时，
    842MB 的权重能用前者加载、用后者必抛 OSError 1455（页面文件太小）。
    """
    from safetensors import safe_open

    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    n = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            key = name
            if key not in params:
                # 兼容 prefix 差异（如 "model.encoder." / 缺 "module."）
                alt = key.replace("model.", "", 1) if key.startswith("model.") else f"model.{key}"
                if alt in params:
                    key = alt
                elif not strict:
                    continue
                else:
                    raise KeyError(name)
            t = f.get_tensor(name)
            dst = params[key].data
            if dst.shape != t.shape:
                if not strict:
                    continue
                raise ValueError(f"{name}: shape {tuple(t.shape)} != {tuple(dst.shape)}")
            dst.copy_(t.to(dst.dtype))
            del t
            n += 1
    return n


class LayaBackend(DecisionBackend):
    name = "laya"
    supports_shared_context_batch = True
    max_state_tokens = 320          # max_len 512 里 state 能占的大致上限

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 max_len: int = 512, batch_tokens: int = 16384, use_temperature: bool = True):
        super().__init__()
        self.model_path = model_path or DEFAULT_PATH
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)
        import rl_common  # type: ignore

        self.rl = rl_common
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.cfg = rl_common.load_cfg(os.path.join(self.model_path, "rl_agent_config.json"))
        self.tok = AutoTokenizer.from_pretrained(os.path.join(self.model_path, "tokenizer"))
        self.model = rl_common.build_model(self.cfg, encoder_dir=os.path.join(self.model_path, "encoder"))
        stream_load_state_dict(self.model, os.path.join(self.model_path, "model.safetensors"))
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.dev).eval()
        self.max_len = int(max_len)
        self.head_max_len = int(self.cfg.get("head_max_len", 192))
        self.batch_tokens = batch_tokens
        self.use_temperature = use_temperature
        self.temperature = list(self.cfg.get("temperature", [1.0, 1.0, 1.0]))
        self.temp_by_options = self.cfg.get("temperature_by_options", {})
        self.pad_id = self.tok.pad_token_id or 0

    # ---------------------------------------------------------------- 温度
    def _temp_for(self, qtype: int, n_options: int) -> float:
        if not self.use_temperature:
            return 1.0
        bucket = self.rl.temp_bucket(qtype, n_options) if hasattr(self.rl, "temp_bucket") else ""
        if bucket and bucket in self.temp_by_options:
            return float(self.temp_by_options[bucket])
        return float(self.temperature[qtype]) if qtype < len(self.temperature) else 1.0

    # ---------------------------------------------------------------- 编码
    def _items(self, state: str, questions: Sequence[TypedQuestion]) -> List[Dict]:
        items = []
        for q in questions:
            qtype = QTYPES.get(q.qtype, 2)
            qq = {"t": q.qtype, "ins": q.text, "crit": {}}
            ids, markers = self.rl.build_sequence(self.tok, state, qq, self.max_len, self.head_max_len)
            items.append({
                "ids": ids, "markers": markers, "target": [0.0] * len(markers),
                "qtype": qtype, "label": 0, "episode": False, "ep_step": 0,
                "rec_uid": -1, "n_options": len(q.options),
            })
        return items

    # ---------------------------------------------------------------- 核心
    @torch.no_grad()
    def _decide(self, state: str, questions: Sequence[TypedQuestion]) -> List[np.ndarray]:
        items = self._items(state, questions)
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        out: List[Optional[np.ndarray]] = [None] * len(items)
        i = 0
        while i < len(order):
            j, L = i, 0
            while j < len(order):
                L = max(L, len(items[order[j]]["ids"]))
                if L * (j - i + 1) > self.batch_tokens:
                    break
                j += 1
            j = max(j, i + 1)
            sel = [items[t] for t in order[i:j]]
            b = self.rl.collate_items([sel], self.pad_id)
            logits, _act = self.model(
                b["input_ids"].to(self.dev), b["attention_mask"].to(self.dev),
                b["marker_pos"].to(self.dev), b["marker_mask"].to(self.dev),
                b["qtype"].to(self.dev))
            logits = logits.float().cpu()
            for r, idx in enumerate(order[i:j]):
                k = len(items[idx]["markers"])
                lg = logits[r, :k]
                t = self._temp_for(items[idx]["qtype"], k)
                p = torch.softmax(lg / max(t, 1e-6), -1).numpy()
                out[idx] = p / p.sum()
            i = j
        return [o for o in out]  # type: ignore[misc]

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))
