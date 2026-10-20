"""评价指标（开题报告 §6.3）：召回、过滤率、校准、显著性。

刻意不给"综合分"。压缩/过滤这类任务的分数可以被分母游戏刷高
（多留一点就涨召回），所以这里只报告互相牵制的原始量：
    schema recall / filter rate / precision / ECE / cost
把它们放在一张帕累托图上看，比合成一个数字诚实。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------- 召回类
def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


@dataclass
class SelectionMetrics:
    recall: float
    table_recall: float
    filter_rate: float          # 被删掉的列占全库列的比例
    kept_columns: int
    total_columns: int
    escalation_rate: float      # 被路由给 LLM 的比例
    precision: float            # 保留列里真正是 gold 的比例
    compression_per_gold: float # 保留列数 / gold 列数（越低越省）

    def as_dict(self) -> Dict[str, float]:
        return {
            "recall": round(self.recall, 4),
            "table_recall": round(self.table_recall, 4),
            "filter_rate": round(self.filter_rate, 4),
            "kept_columns": self.kept_columns,
            "total_columns": self.total_columns,
            "escalation_rate": round(self.escalation_rate, 4),
            "precision": round(self.precision, 4),
            "compression_per_gold": round(self.compression_per_gold, 3),
        }


def selection_metrics(kept_pairs, kept_tables, schema, gold, n_escalated=0) -> SelectionMetrics:
    total = schema.n_columns()
    kept = set(kept_pairs)
    gold_flat = gold.flat_pairs()
    gold_n = max(1, len(gold.requirements))
    return SelectionMetrics(
        recall=gold.recall(kept),
        table_recall=gold.table_recall(set(kept_tables)),
        filter_rate=safe_div(total - len(kept), total),
        kept_columns=len(kept),
        total_columns=total,
        escalation_rate=safe_div(n_escalated, total),
        precision=safe_div(len(kept & gold_flat), len(kept)),
        compression_per_gold=safe_div(len(kept), gold_n),
    )


# ----------------------------------------------------------------------------- 严格召回（对齐 AutoLink / EviLink 口径）
def strict_recall(gold, kept_pairs) -> float:
    """Strict Recall Rate 的逐题定义：所有 gold 需求都被覆盖才算 1，否则 0。

    AutoLink / EviLink 报的 SRR 是"整题全对"口径，比逐列 recall 严格得多：
    漏一个 JOIN 键整题就废。两种口径必须分开报 ——
    只报软召回会让"每题都漏一点"的方法看起来很好。
    """
    if gold is None or not gold.requirements:
        return 1.0
    return 1.0 if gold.recall(set(kept_pairs)) >= 1.0 - 1e-9 else 0.0


def srr(per_example: Sequence[float]) -> float:
    return float(np.mean(per_example)) if len(per_example) else 0.0


def field_level_recall(gold, kept_pairs) -> float:
    """字段级（列级）召回：覆盖到的 gold 需求比例。分母是需求数不是列数。"""
    if gold is None or not gold.requirements:
        return 1.0
    return float(gold.recall(set(kept_pairs)))


def tep(perf: float, tokens: float, perf_ref: float, tokens_ref: float,
        per_k_tokens: float = 1000.0) -> float:
    """Token Elasticity of Performance（借用 EllieSQL 的指标）。

        TEP = ΔPerformance / ΔToken(K)

    含义：相对参考点，每多花（或少花）1K token 换到多少性能。
    省钱方向（tokens < ref）时若性能不降，TEP 为正且越大越好 ——
    这正是本课题要证明的"砍掉无效 schema 不掉精度"。
    ΔToken 为 0 时返回 inf（不可比），调用方需过滤。
    """
    dtok = (tokens - tokens_ref) / per_k_tokens
    if abs(dtok) < 1e-9:
        return float("nan")
    return (perf - perf_ref) / dtok


def pareto_front(points: Sequence[Tuple[float, float]],
                 maximize_y: bool = True) -> List[int]:
    """给定 (token_cost, perf) 点集，返回帕累托前沿上的下标。
    x 越小越好（成本），y 越大越好（性能）。"""
    idx = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front: List[int] = []
    best = -float("inf")
    for i in idx:
        y = points[i][1] if maximize_y else -points[i][1]
        if y > best + 1e-12:
            front.append(i)
            best = y
    return front


# ----------------------------------------------------------------------------- 校准
def ece(probs: Sequence[float], labels: Sequence[int], bins: int = 10,
        strategy: str = "uniform") -> Tuple[float, List[Dict]]:
    """ECE + reliability diagram 数据。

    两种分箱：uniform（等宽，标准做法）与 quantile（等频，样本少时更稳）。
    两个都返回，避免"挑一个好看的报"。
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.size == 0:
        return 0.0, []
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Dict] = []
    tot = 0.0
    n = len(p)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        if not m.any():
            rows.append({"lo": float(lo), "hi": float(hi), "n": 0, "acc": None, "conf": None})
            continue
        acc = float(y[m].mean())
        conf = float(p[m].mean())
        w = m.sum() / n
        tot += w * abs(acc - conf)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "acc": round(acc, 4), "conf": round(conf, 4)})
    return float(tot), rows


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    return float(np.mean((p - y) ** 2)) if p.size else 0.0


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """阈值无关的排序质量：召回被过滤率掩盖时，AUROC 能说明打分本身好不好。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # 平局取平均秩
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def recall_at_filter_rate(scores: Sequence[float], labels: Sequence[int],
                          target_filter: float) -> Tuple[float, float]:
    """给定过滤率，返回 (达到的 recall, 实际阈值)。用于画 recall-filter 曲线。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.size == 0:
        return 0.0, 0.0
    n_drop = int(round(target_filter * len(s)))
    n_drop = max(0, min(len(s), n_drop))
    if n_drop == 0:
        return float(y.mean()), float(s.min())
    # 并列分数的取舍必须是确定的、无偏的。
    # BM25 里 80% 的列得分恰好为 0，np.argsort 对并列值的顺序是任意的 ——
    # 高过滤率下"删哪些 0 分列"会直接影响 recall，等于让实现细节决定结果。
    # 这里用固定种子洗牌后再排序：可复现，且并列组内是等概率的（无偏）。
    rng = np.random.default_rng(0)
    order = rng.permutation(len(s))
    idx = order[np.argsort(s[order], kind="stable")][:n_drop]   # 分数最低的先删
    keep = np.ones(len(s), dtype=bool)
    keep[idx] = False
    thr = float(s[idx].max()) if n_drop else 0.0
    gold_kept = float(y[keep].sum())
    return safe_div(gold_kept, float(y.sum())), thr


# ----------------------------------------------------------------------------- 显著性
def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                     seed: int = 0) -> Dict[str, float]:
    """配对 bootstrap：返回 diff=b-a 的均值、95% CI 与双侧 p。

    只用这一个检验（加上符号翻转置换检验做交叉验证），不做多重比较时
    不报"显著"；Holm 校正在 compare_systems 里做。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    d = b - a
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    obs = d.mean()
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff": float(obs), "ci_lo": float(lo), "ci_hi": float(hi),
        "p_boot": float(min(1.0, p)), "n": int(n),
    }


def sign_flip_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000,
                   seed: int = 0) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = (d * flips).mean(axis=1)
    return float((np.abs(perm) >= obs).mean())


def holm_correction(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare_systems(results: Dict[str, Sequence[float]], baseline: str,
                    n_boot: int = 10000, seed: int = 0) -> List[Dict]:
    """多个方法 vs 同一基线，带 Holm 校正。"""
    base = np.asarray(results[baseline], dtype=float)
    raw = []
    names = []
    for name, vals in results.items():
        if name == baseline:
            continue
        r = paired_bootstrap(base, np.asarray(vals, dtype=float), n_boot, seed)
        raw.append(r)
        names.append(name)
    adj = holm_correction([r["p_boot"] for r in raw])
    for r, a in zip(raw, adj):
        r["p_holm"] = a
        r["system"] = r.get("system", "")
    for name, r in zip(names, raw):
        r["system"] = name
    return raw


def mean_ci(x: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """均值 + 95% bootstrap CI。

    ⚠️ 原来一次性 `rng.integers(0, len(x), size=(n_boot, len(x)))`，
    在 n=6765 上要分配 10000×6765 的 int64 = **516 MiB**，本机 16GB 内存直接
    `_ArrayMemoryError`（这已经是第二次踩了：上次是在 laya 打分跑完之后，
    只能把调用方的 n_boot 降到 2000 绕过去 —— 治标不治本）。
    现在改成**分块**：每次只分配 chunk×n，峰值内存从 O(n_boot·n) 降到 O(chunk·n)。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    chunk = max(1, min(n_boot, int(4_000_000 // max(1, n))))   # 约 32 MB/块
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        means[done:done + b] = x[idx].mean(axis=1)
        done += b
        del idx
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)
