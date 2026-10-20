"""System-Two 精筛层（开题报告 §5.5）。

被 System-One 判为"不确定"的候选交到这里。三种实现：

1. KeepAllRefiner   —— 保底基线：不确定就全留（等价于不做精筛，用于消融）
2. EmbeddingRefiner —— 本地 embedding 重排，$0，给"不用 LLM 能到哪"一个参照
3. LLMRefiner       —— OpenAI 兼容端点的真 LLM 精筛；默认 dry-run + 强制估费

LLMRefiner 的付费纪律与 Jev 后端一致：调用前打印条数与预估费用，
未 --live 不发请求；结果走磁盘缓存，重跑不重复计费。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .data_loader import Example
from .paths import CACHE

Pair = Tuple[str, str]


class Refiner:
    name = "base"

    def refine(self, ex: Example, candidates: Sequence[Pair]) -> Set[Pair]:
        raise NotImplementedError

    def reset_stats(self) -> None:
        pass


class KeepAllRefiner(Refiner):
    name = "keep-all"

    def refine(self, ex, candidates):
        return set(candidates)


class DropAllRefiner(Refiner):
    """消融用：不确定就全删，测"System-One 单独够不够"。"""

    name = "drop-all"

    def refine(self, ex, candidates):
        return set()


class EmbeddingRefiner(Refiner):
    """用 sentence-transformers 把候选列与问题做相似度重排，取 top-fraction。

    这是一个"便宜但不糊涂"的对照：它证明精筛层带来的增益有多少来自
    LLM 的推理，有多少只是因为做了排序。
    """

    name = "embedding"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 keep_fraction: float = 0.5, device: Optional[str] = None,
                 model_path: Optional[str] = None):
        self.keep_fraction = keep_fraction
        self.model_path = model_path or model_name
        self._model = None
        self.device = device
        self.calls = 0

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path, device=self.device)
        return self._model

    def refine(self, ex: Example, candidates: Sequence[Pair]) -> Set[Pair]:
        if not candidates:
            return set()
        m = self._load()
        self.calls += 1
        q = ex.question + (" " + ex.evidence if ex.evidence else "")
        docs = [f"{t} {c}".replace("_", " ") for t, c in candidates]
        emb = m.encode([q] + docs, normalize_embeddings=True, show_progress_bar=False)
        sims = (emb[1:] @ emb[0]).tolist()
        order = sorted(range(len(candidates)), key=lambda i: -sims[i])
        k = max(1, int(round(self.keep_fraction * len(candidates))))
        return {candidates[i] for i in order[:k]}


@dataclass
class LLMCost:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    latency_s: float = 0.0


class LLMRefiner(Refiner):
    """OpenAI 兼容 chat 端点的精筛层。默认 dry-run。"""

    name = "llm"

    PROMPT = (
        "You are helping select the minimal set of database columns needed to write a correct SQL "
        "query.\n\n"
        "Question: {question}\n"
        "{evidence}"
        "Schema:\n{schema}\n\n"
        "Candidate columns (uncertain):\n{candidates}\n\n"
        "For each candidate answer only YES if the column is required to write the correct SQL, "
        "otherwise NO. Reply with one line per candidate in the form `table.column: YES/NO`."
    )

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini", dry_run: bool = True,
                 price_in_per_mtok: float = 0.15, price_out_per_mtok: float = 0.60,
                 cache_path: Optional[str] = None, timeout: int = 90,
                 max_candidates_per_call: int = 120):
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "..", "..", "configs", "llm.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        self.endpoint = (endpoint or cfg.get("endpoint")
                         or os.environ.get("OPENAI_BASE_URL", ""))
        self.api_key = api_key or cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or cfg.get("model", "gpt-4o-mini")
        self.dry_run = dry_run
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.timeout = timeout
        self.max_candidates_per_call = max_candidates_per_call
        self.cost = LLMCost()
        self.cache_path = cache_path or os.path.join(
            os.environ.get("JEVSCHEMA_CACHE", CACHE), "llm_refine.jsonl")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.cache: Dict[str, List[str]] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.cache[rec["k"]] = rec["keep"]
        self._fh = open(self.cache_path, "a", encoding="utf-8")

    # ---------------------------------------------------------------- 计费
    def estimate(self, ex: Example, candidates: Sequence[Pair]) -> LLMCost:
        schema_txt = ex.schema.summary()
        prompt = self.PROMPT.format(question=ex.question,
                                    evidence=(f"Evidence: {ex.evidence}\n" if ex.evidence else ""),
                                    schema=schema_txt,
                                    candidates="\n".join(f"{t}.{c}" for t, c in candidates))
        n_in = max(1, len(prompt) // 4)
        n_out = max(1, len(candidates) * 4)
        return LLMCost(prompt_tokens=n_in, completion_tokens=n_out,
                       usd=n_in / 1e6 * self.price_in + n_out / 1e6 * self.price_out, calls=1)

    # ---------------------------------------------------------------- 请求
    def _post(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0}
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(reply: str, candidates: Sequence[Pair]) -> Set[Pair]:
        return LLMRefiner._parse(reply, candidates)[0]

    @staticmethod
    def _parse_detailed(reply: str, candidates: Sequence[Pair]) -> Tuple[Set[Pair], int]:
        """解析 LLM 的逐行回答，返回 (保留集合, 成功匹配的行数)。

        为什么必须写得这么防御：模型几乎一定会加装饰 —— 反引号、markdown 列表符号、
        甚至只回答了前几条。旧实现要求行必须精确等于 `table.column: YES`，
        实测回复是 `` `singer.Singer_ID: YES` ``，反引号让 key 匹配失败、返回空集，
        于是**整批候选被判为删除**，recall 从 0.956 掉到 0.730 ——
        这不是"LLM 判断差"，是解析失败。宁可多写几行清洗，也不能静默删光。
        """
        keep: Set[Pair] = set()
        low = {f"{t}.{c}".lower(): (t, c) for t, c in candidates}
        col_only: Dict[str, List[Pair]] = {}
        for t, c in candidates:
            col_only.setdefault(c.lower(), []).append((t, c))
        matched = 0
        for raw in reply.splitlines():
            line = raw.strip().strip("`").strip()
            line = re.sub(r"^[-*•]\s*", "", line).strip()
            if ":" not in line:
                continue
            key, ans = line.rsplit(":", 1)
            key = key.strip().strip("`").strip().lower()
            matched += 1
            if not ans.strip().strip("`").strip().upper().startswith("YES"):
                continue
            if key in low:
                keep.add(low[key])
            elif key in col_only and len(col_only[key]) == 1:
                keep.add(col_only[key][0])
            else:
                for name, pair in low.items():          # 行里有 t.c 子串也算命中
                    if name in key:
                        keep.add(pair)
        return keep, matched

    def refine(self, ex: Example, candidates: Sequence[Pair]) -> Set[Pair]:
        if not candidates:
            return set()
        candidates = list(candidates)[: self.max_candidates_per_call]
        schema_txt = ex.schema.summary()
        prompt = self.PROMPT.format(question=ex.question,
                                    evidence=(f"Evidence: {ex.evidence}\n" if ex.evidence else ""),
                                    schema=schema_txt,
                                    candidates="\n".join(f"{t}.{c}" for t, c in candidates))
        key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        if key in self.cache:
            return {tuple(x) for x in self.cache[key]}  # type: ignore[misc]
        if self.dry_run:
            est = self.estimate(ex, candidates)
            print(f"[dry-run][llm-refine] {len(candidates)} candidates, "
                  f"~{est.prompt_tokens} in-tokens, est ${est.usd:.5f}")
            return set(candidates)
        t0 = time.time()
        try:
            reply = self._post(prompt)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM refine HTTP {e.code}: {e.read().decode('utf-8','ignore')[:200]}")
        keep = self._parse(reply, candidates)
        self.cost.calls += 1
        self.cost.latency_s += time.time() - t0
        self.cache[key] = [list(x) for x in keep]
        self._fh.write(json.dumps({"k": key, "keep": [list(x) for x in keep]},
                                  ensure_ascii=False) + "\n")
        self._fh.flush()
        return keep

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass


class OllamaRefiner(Refiner):
    """本地 ollama 作为 System-Two（$0，可离线复现）。

    存在的意义：让"不确定 -> 交给 LLM"这条路真的跑起来，而不是只报告
    escalation rate 就完事。用本地模型既不用付费 API，也能给出真实延迟。
    默认模型 qwen2.5-coder:7b（SQL 场景更合适）；要快可以换 :1.5b。
    """

    name = "ollama"

    PROMPT = (
        "You are helping select the minimal set of database columns needed to write a correct "
        "SQL query.\n\n"
        "Question: {question}\n"
        "Schema (candidate tables only):\n{schema}\n\n"
        "For each candidate column below, answer YES only if the correct SQL for this question "
        "must use that column, otherwise answer NO.\n{candidates}\n\n"
        "Reply with EXACTLY one line for EACH candidate, in this format (no backticks, "
        "no bullets, no explanations):\n"
        "table.column: YES\n"
        "table.column: NO\n"
        "...\n"
        "Do not skip any candidate. Do not add any other text."
    )

    def __init__(self, model: str = "qwen2.5-coder:7b",
                 endpoint: str = "http://localhost:11434", dry_run: bool = False,
                 timeout: int = 120, cache_path: Optional[str] = None,
                 max_candidates: int = 60, num_predict: int = 512,
                 include_types: bool = True):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.dry_run = dry_run
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.num_predict = num_predict
        self.include_types = include_types
        self.calls = 0
        self.latency_s = 0.0
        self.prompt_chars = 0
        self.parse_failures = 0      # 解析失败、回退到全保留的次数（必须报告）
        self.cache_path = cache_path or os.path.join(
            os.environ.get("JEVSCHEMA_CACHE", CACHE),
            f"ollama_refine_{model.replace(':', '_')}.jsonl")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.cache: Dict[str, List[str]] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.cache[rec["k"]] = rec["keep"]
        self._fh = open(self.cache_path, "a", encoding="utf-8")

    def _post(self, prompt: str, num_predict: Optional[int] = None) -> str:
        import urllib.request

        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"temperature": 0.0,
                               "num_predict": int(num_predict or self.num_predict)}}
        req = urllib.request.Request(self.endpoint + "/api/chat",
                                    data=json.dumps(payload).encode("utf-8"),
                                    headers={"Content-Type": "application/json"}, method="POST")
        # 注意：payload 里 num_predict 由调用方按候选数放大（见 refine）
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return (data.get("message") or {}).get("content", "") or data.get("response", "")

    def refine(self, ex: Example, candidates: Sequence[Pair]) -> Set[Pair]:
        if not candidates:
            return set()
        cands = list(candidates)[: self.max_candidates]
        tables = sorted({t for t, _ in cands})
        schema_txt = ex.schema.summary(tables=tables, with_fk=True)
        lines = []
        for t, c in cands:
            if self.include_types:
                lines.append(f"- {t}.{c} ({ex.schema.col_type(t, c)})")
            else:
                lines.append(f"- {t}.{c}")
        prompt = self.PROMPT.format(question=ex.question, schema=schema_txt,
                                    candidates="\n".join(lines))
        # 实测：不写"必须输出 N 行"，7B 模型只答 1 行就停（回退保险会整批保留，
        # 等于白花一次 LLM 调用）。写上数量后三个模型都能答全。
        prompt += (f"\nThere are exactly {len(cands)} candidates above. Your reply must "
                   f"contain exactly {len(cands)} lines, one per candidate, in the same order.")
        # 输出预算必须随候选数放大：默认 512 在 60 个候选时会被截断。
        num_predict = max(self.num_predict, 16 * len(cands) + 128)
        key = hashlib.sha1((self.model + "\n" + prompt).encode("utf-8")).hexdigest()
        if key in self.cache:
            return {tuple(x) for x in self.cache[key]}  # type: ignore[misc]
        if self.dry_run:
            print(f"[dry-run][ollama:{self.model}] {len(cands)} candidates, "
                  f"{len(prompt)} chars")
            return set(cands)
        t0 = time.time()
        reply = self._post(prompt, num_predict)
        self.calls += 1
        self.latency_s += time.time() - t0
        self.prompt_chars += len(prompt)
        keep = self._safe_refine(reply, cands)
        self.cache[key] = [list(x) for x in keep]
        self._fh.write(json.dumps({"k": key, "keep": [list(x) for x in keep]},
                                  ensure_ascii=False) + "\n")
        self._fh.flush()
        return keep

    @staticmethod
    def _parse(reply: str, candidates: Sequence[Pair]) -> Set[Pair]:
        return LLMRefiner._parse(reply, candidates)

    def _safe_refine(self, reply: str, cands: List[Pair]) -> Set[Pair]:
        """解析 + 失败保险。

        保险规则：匹配到的行数不足候选数的一半，就判为解析失败并**回退到全保留**。
        理由：解析失败的后果是整批被删（recall 崩 ~0.22），而全保留只是没拿到收益；
        宁可拿不到收益，也不能静默毁掉结果。失败次数单独计数，必须出现在报告里。
        """
        keep, matched = LLMRefiner._parse_detailed(reply, cands)
        if cands and matched < 0.5 * len(cands):
            self.parse_failures += 1
            return set(cands)
        return keep

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass


def build_refiner(name: str = "keep-all", **kw) -> Refiner:
    name = (name or "keep-all").lower()
    if name in ("keep-all", "keep", "none"):
        return KeepAllRefiner()
    if name == "drop-all":
        return DropAllRefiner()
    if name == "embedding":
        return EmbeddingRefiner(**kw)
    if name == "llm":
        return LLMRefiner(**kw)
    if name == "ollama":
        return OllamaRefiner(**kw)
    raise ValueError(f"未知精筛层: {name}")
