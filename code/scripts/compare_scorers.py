"""同口径打分器对比：AUROC + recall@filter，Spider dev 全 schema。

为什么必须单独写这个脚本：
    run_pilot 的 recall/filter 走的是**完整路由**（三路阈值 + FK 闭包 + 整表保留 +
    升级到 System-Two），而 baselines.py 走的是"按分数砍掉最低的 k 个"。
    两者不是同一件事，把它们的数字放进一张表属于口径错误 —— 之前就差点这么干。
    这里让每个打分器只产出一个 {pair: score}，然后全部走同一个评测函数。

参与对比的打分器：
    bm25 / embedding        纯词法·语义基线（无监督，$0）
    risk_ft                 微调后的 System-One 风险模型（laya + Spider train gold 监督）
    risk_base               未微调的 laya 零样本（对照组）
    fusion                  (1-α)·risk + α·词法（词法做样本内 max 归一）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.backends import build_backend, close_backend  # noqa: E402
from jevschema.data_loader import RAW_DIR, load_spider_split, load_spider_schemas  # noqa: E402
from jevschema.fusion import FusionConfig, _normalize, lexical_scores  # noqa: E402
from jevschema.metrics import auroc, mean_ci, paired_bootstrap, recall_at_filter_rate  # noqa: E402
from jevschema.risk import score_columns  # noqa: E402
from jevschema.summary import SummaryConfig, build_state  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate(name: str, per_ex: list, examples: list, rates) -> dict:
    """per_ex: 每个例子一个 {pair: score}。全 schema 上算 AUROC 与 recall@rate。"""
    all_s, all_y = [], []
    aucs = []
    per_rate = {r: [] for r in rates}
    for ex, sc in zip(examples, per_ex):
        pairs = list(ex.schema.columns())
        gold = ex.gold.flat_pairs() if ex.gold else set()
        s = np.asarray([float(sc.get(p, 0.0)) for p in pairs], dtype=float)
        y = np.asarray([1 if p in gold else 0 for p in pairs], dtype=int)
        all_s.append(s)
        all_y.append(y)
        # AUROC 必须**逐例算再平均**：不同库的分数本来就不可比，
        # 跨例池化会把"A 库的 0.9 分列"和"B 库的 0.5 分 gold 列"放在一起比，
        # 那是无意义的比较（实测同一批分数：池化 0.796 vs 逐例 0.912）。
        aucs.append(auroc(s, y))
        for r in rates:
            rec, _ = recall_at_filter_rate(s, y, r)
            per_rate[r].append(rec)
    s_all = np.concatenate(all_s) if all_s else np.zeros(0)
    y_all = np.concatenate(all_y) if all_y else np.zeros(0)
    return {
        "auroc": float(np.nanmean(aucs)),
        "auroc_pooled": float(auroc(s_all, y_all)),
        "curve": {str(r): float(np.mean(per_rate[r])) for r in rates},
        "curve_ci": {str(r): [round(float(x), 4) for x in mean_ci(per_rate[r])] for r in rates},
        "_per_example": {str(r): [float(v) for v in per_rate[r]] for r in rates},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="laya")
    ap.add_argument("--model-path", default="", help="微调权重目录")
    ap.add_argument("--views", default="ask")
    ap.add_argument("--combine", default="mean")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512,
                    help="序列总长。512 时 21.9%% 的 dev 样例 state 会被右侧截断"
                         "（p90=499 token，可用余量仅 317）；1024 可做到 0 截断。")
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES))
    ap.add_argument("--alphas", default="0.0,0.2,0.3,0.5,0.7")
    ap.add_argument("--fusion-lexical", default="bm25,embedding")
    ap.add_argument("--normalize", default="max", choices=["max", "rank", "minmax", "none"])
    ap.add_argument("--include-base", action="store_true", help="同时评测未微调的 laya")
    ap.add_argument("--reuse-risk", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rates = [float(x) for x in a.rates.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    views = tuple(a.views.split(","))
    tag = (f"cmp_{a.backend}{'_ft' if a.model_path else ''}_dev{a.n}"
           f"_v{'-'.join(views)}_L{a.max_len}")
    risk_path = a.reuse_risk or os.path.join(RESULTS, f"{tag}.jsonl")

    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    # 取样口径必须和 run_baselines.py 一致，否则两组数字来自不同样本集、根本不能比。
    # 统一为 shuffle=False（取前 N 条，不依赖 shuffle 实现），n<=0 表示全量 dev。
    examples = load_spider_split(os.path.join(RAW_DIR, "spider_val.parquet"), schemas,
                                 limit=a.n if a.n > 0 else None, seed=a.seed, shuffle=False)
    print(f"dev 样例 {len(examples)}，过滤率 {rates}")

    # ---------------------------------------------------------------- 词法 / 语义
    def load_lex(kind: str) -> list:
        """按样例 id 增量缓存词法分数。

        embedding 要加载 sentence-transformers，既慢又在与训练并发时因内存紧张
        segfault（实测）。缓存到"样例 id -> 分数"后，任何 n 都能复用已有部分，
        只补算缺失的样例 —— 换 n 重跑不必重新加载模型。
        """
        cpath = os.path.join(RESULTS, f"lex_{kind}_byid.jsonl")
        store: dict = {}
        if os.path.exists(cpath):
            for line in open(cpath, encoding="utf-8"):
                r = json.loads(line)
                store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
        todo = [ex for ex in examples if ex.ex_id not in store]
        if todo:
            t0 = time.time()
            with open(cpath, "a", encoding="utf-8") as f:
                for ex in todo:
                    sc = lexical_scores(ex, FusionConfig(lexical=kind))
                    store[ex.ex_id] = sc
                    f.write(json.dumps({"id": ex.ex_id,
                                        "s": {"|".join(k): v for k, v in sc.items()}},
                                       ensure_ascii=False) + "\n")
            print(f"  {kind} 新算 {len(todo)} 例 {time.time()-t0:.1f}s -> {cpath}")
        else:
            print(f"  {kind} 全部命中缓存 ({len(examples)} 例)")
        return [store[ex.ex_id] for ex in examples]

    lex_cache: dict = {}
    for kind in [k for k in a.fusion_lexical.split(",") if k]:
        lex_cache[kind] = load_lex(kind)

    out = {"tag": tag, "n": len(examples), "rates": rates, "scorers": {}}

    # ---------------------------------------------------------------- 纯词法基线
    for kind, sc in lex_cache.items():
        out["scorers"][kind] = evaluate(kind, sc, examples, rates)

    # ---------------------------------------------------------------- 风险模型
    scored = {"ft": None, "base": None}
    if os.path.exists(risk_path) and a.reuse_risk:
        raw = [json.loads(l) for l in open(risk_path, encoding="utf-8")]
        scored["ft"] = [{tuple(k.split("|")): v for k, v in r.items()} for r in raw]
        print(f"复用已存风险分数: {risk_path} ({len(raw)} 例)")
    if scored["ft"] is None:
        be = build_backend(a.backend, cached=False, model_path=a.model_path or None,
                           max_len=a.max_len)
        t0 = time.time()
        ft_scores = []
        for i, ex in enumerate(examples):
            state, tables = build_state(ex.question_block(), ex.schema, SummaryConfig())
            cols = [(t, c) for (t, c) in ex.schema.columns() if t in set(tables)]
            pv = score_columns(be, state, cols, views, a.combine, a.batch_size)
            ft_scores.append({e: float(np.mean([v[vk] for vk in views])) for e, v in pv.items()})
            if (i + 1) % 25 == 0:
                print(f"  微调模型打分 {i+1}/{len(examples)} {time.time()-t0:.0f}s", flush=True)
        scored["ft"] = ft_scores
        with open(risk_path, "w", encoding="utf-8") as f:
            for d in ft_scores:
                f.write(json.dumps({"|".join(k): v for k, v in d.items()}, ensure_ascii=False) + "\n")
        close_backend(be)
        print(f"风险分数已存: {risk_path}")
    out["scorers"]["risk_ft"] = evaluate("risk_ft", scored["ft"], examples, rates)

    if a.include_base:
        be0 = build_backend(a.backend, cached=False)
        bs = []
        for ex in examples:
            state, tables = build_state(ex.question_block(), ex.schema, SummaryConfig())
            cols = [(t, c) for (t, c) in ex.schema.columns() if t in set(tables)]
            pv = score_columns(be0, state, cols, views, a.combine, a.batch_size)
            bs.append({e: float(np.mean([v[vk] for vk in views])) for e, v in pv.items()})
        close_backend(be0)
        scored["base"] = bs
        out["scorers"]["risk_base"] = evaluate("risk_base", bs, examples, rates)

    # ---------------------------------------------------------------- 融合
    for kind, lex in lex_cache.items():
        for al in alphas:
            if al <= 0:
                continue
            fused = []
            for ex, sc, lx in zip(examples, scored["ft"], lex):
                ln = _normalize(lx, a.normalize)
                fused.append({k: (1 - al) * v + al * ln.get(k, 0.0) for k, v in sc.items()})
            nm = f"risk+{kind}@{al}"
            out["scorers"][nm] = evaluate(nm, fused, examples, rates)

    # ---------------------------------------------------------------- 显著性（vs embedding）
    if "embedding" in out["scorers"] and "risk_ft" in out["scorers"]:
        sig = {}
        for r in rates:
            k = str(r)
            aa = out["scorers"]["risk_ft"]["_per_example"][k]
            bb = out["scorers"]["embedding"]["_per_example"][k]
            r = paired_bootstrap(aa, bb, n_boot=5000)      # diff = embedding - risk_ft
            sig[k] = {"delta": round(-r["diff"], 4),
                      "ci": [round(-r["ci_hi"], 4), round(-r["ci_lo"], 4)],
                      "p": round(float(r["p_boot"]), 5)}
        out["risk_ft_vs_embedding"] = sig
    for v in out["scorers"].values():
        v.pop("_per_example", None)

    # ---------------------------------------------------------------- 打印
    print("\n== 打分器对比（Spider dev, 全 schema）==")
    print(f"{'scorer':<22}{'AUROC':>8} " + " ".join(f"{r:.1f}" for r in rates))
    for nm, v in sorted(out["scorers"].items(), key=lambda kv: -kv[1]["curve"][str(rates[-1])]):
        row = " ".join(f"{v['curve'][str(r)]:.3f}" for r in rates)
        print(f"{nm:<22}{v['auroc']:>8.4f} {row}")
    if "risk_ft_vs_embedding" in out:
        print("\n== risk_ft - embedding（配对 bootstrap, 5000 次）==")
        for r in rates:
            s = out["risk_ft_vs_embedding"][str(r)]
            print(f"  filter {r:.1f}: Δ={s['delta']:+.4f} CI=[{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}] "
                  f"p={s['p']:.4f}")

    path = a.out or os.path.join(RESULTS, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
