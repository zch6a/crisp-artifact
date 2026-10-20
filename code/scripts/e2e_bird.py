"""BIRD-Dev 端到端：schema linking -> SQL 生成 -> 执行比对，产出 token-EX 帕累托前沿。

论文的主图就是这张：x = 每次查询的总 token，y = EX。
只报 SRR 会被"全留着"刷满，只报 EX 又看不出省了多少 —— 两个轴一起才说明问题。
（SRR 是 schema linking 的代理指标；EX 才是下游真正在乎的。两个必须同时报，
  否则"省了 token"这句话本身没有意义。）

控制变量（锁死，否则比不出东西）：
  * 同一个生成器（本地 ollama，temperature=0，--model 指定）；
  * 同一套 prompt 模板（tokens.build_gen_prompt）；
  * 同一批样本、同一套 gold、同一个执行器；
  * 生成结果按 prompt 哈希缓存，重跑不重复调用（本地模型 $0，但省时间）。
变的只有一件事：喂给生成器的 schema 是哪一块。

被比较的 schema 供给方式：
  full            全 schema 直喂（不压缩；既是上界也是 baseline）
  bm25@K          BM25 top-K + 结构闭包（纯词法基线）
  fused@K         BM25⊕embedding 融合 top-K + 结构闭包（本文方法的固定预算版）
  adapt@z         融合排序 + **预算预测器**逐题定深度 + 结构闭包（自适应版）
  risk_ft@K       System-One 风险模型 top-K（可选，--skip-risk 关掉就不占 GPU）

⚠️ 与旧版的差别：旧版用 selection.expand_kept（v1），这里换成 v2 的 closure，
   --closure 用同一套配置串（f1p5s15k1 之类），与 pareto_closure /
   budget_predictor 完全一致，三张表才能对得上。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)
import jevschema.paths as _P  # noqa: E402  (env CRISP_DATA / CRISP_MODELS)

import numpy as np  # noqa: E402

from jevschema.backends import build_backend, close_backend  # noqa: E402
from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.exec_eval import evaluate_execution  # noqa: E402
from jevschema.fusion import FusionConfig, lexical_scores  # noqa: E402
from jevschema.generator import OllamaGenerator  # noqa: E402
from jevschema.metrics import paired_bootstrap, pareto_front, tep  # noqa: E402
from jevschema.risk import VIEWS, score_columns  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import SummaryConfig, build_state  # noqa: E402
from jevschema.tokens import (TOKENIZER, build_gen_prompt, n_tokens,  # noqa: E402
                              schema_prompt_text)

from budget_predictor import feats, load_prob_model, oracle_m, prob_of  # noqa: E402
from train_lex_classifier import features as lex_features  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


def _cache_path(tag: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{tag}.jsonl")


def score_all(tag: str, examples: list, fn) -> list:
    p = _cache_path(tag)
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    todo = [e for e in examples if e.ex_id not in store]
    if todo and fn is None:
        # 只读模式：缓存缺就是缺，绝不用别的信号顶替（否则两组实验静默变成同一组）
        print(f"  {tag}: 只读缓存，缺 {len(todo)}/{len(examples)} 例 → 置空", flush=True)
        for e in todo:
            store[e.ex_id] = {}
    elif todo:
        t0 = time.time()
        with open(p, "a", encoding="utf-8") as f:
            for ex in todo:
                sc = fn(ex)
                store[ex.ex_id] = sc
                f.write(json.dumps({"id": ex.ex_id,
                                    "s": {"|".join(k): v for k, v in sc.items()}},
                                   ensure_ascii=False) + "\n")
        print(f"  {tag}: 新算 {len(todo)} 例 {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"  {tag}: 命中缓存 ({len(examples)} 例)", flush=True)
    return [store[e.ex_id] for e in examples]


def _norm(d, how="max"):
    v = np.asarray(list(d.values()), dtype=float)
    if how == "max":
        hi = v.max()
        return {k: float(x) for k, x in
                zip(d.keys(), v / hi if hi > 0 else np.zeros_like(v))}
    lo, hi = v.min(), v.max()
    return {k: float(x) for k, x in zip(d.keys(), (v - lo) / max(1e-9, hi - lo))}


def fused_orders(examples, bm, eb):
    """BM25⊕embedding 融合排序（CPU，$0），与 pareto_closure 里的实现一致。"""
    have_emb = any(len(d) > 0 for d in eb)
    out = []
    for e, b, d in zip(examples, bm, eb):
        pairs = e.schema.columns()
        bn = _norm({q: float(b.get(q, 0.0)) for q in pairs}, "max")
        dn = (_norm({q: float(d.get(q, 0.0)) for q in pairs}, "minmax")
              if have_emb else {q: 0.0 for q in pairs})
        s = np.asarray([0.5 * bn[q] + 0.5 * dn[q] for q in pairs])
        out.append((pairs, s, np.argsort(-s, kind="stable")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--closure", default="f1p5s15k1")
    ap.add_argument("--keeps", default="10,15,20,25,35,50")
    ap.add_argument("--zs", default="0.5,1.5,2.5")
    ap.add_argument("--ms", default="5,6,8,10,12,15,18,20,25,30,35,40,50,60")
    ap.add_argument("--skip-risk", action="store_true",
                    help="跳过 System-One 打分（不占 GPU，纯 CPU 跑完）")
    ap.add_argument("--model-path", default=os.path.join(_P.MODELS, "laya_ft_bird"))
    ap.add_argument("--views", default="ask")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--gen-timeout", type=float, default=180.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--laya-tag", default="bird_layapareto_laya_ft_bird_vask_L1024_t6",
                    help="System-One(laya) 打分缓存 tag；留空则不跑该臂。"
                         "默认 tag 是 laya_pareto 跑完的 1534 条，直接命中、不占 GPU。")
    ap.add_argument("--laya-top-tables", type=int, default=6,
                    help="System-One 的表级门控宽度（必须与打分缓存的 tag 一致，"
                         "否则 link token 算错）")
    ap.add_argument("--laya-link", type=float, default=0.0,
                    help=">0 时直接指定 System-One 的 linking token/查询，跳过复算")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    keeps = [int(x) for x in a.keeps.split(",")]
    zs = [float(x) for x in a.zs.split(",")]
    ms = [int(x) for x in a.ms.split(",")]
    cfg = parse_closure(a.closure)
    views = tuple(a.views.split(","))

    schemas = load_bird_schemas()
    # 预算预测器在**全量 dev** 上做 K 折拟合，生成/执行只在 stride 抽出的子集上跑。
    # 否则 307 条样本拟合出来的 R²≈0（全量 1534 是 0.136），adapt@z 会退化成近似常数深度，
    # 端到端里这个系统就白比了。K 折保证每个样本的深度预测都没让它自己参与拟合。
    all_ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    sub_idx = list(range(0, len(all_ex), a.stride)) if a.stride > 1 else list(range(len(all_ex)))
    examples = [all_ex[i] for i in sub_idx]
    print(f"BIRD-Dev E2E: 生成 {len(examples)} 例（stride={a.stride}，"
          f"预算预测器在 {len(all_ex)} 例上 K 折拟合）, 生成器 {a.model}, "
          f"闭包={cfg.tag()}, keeps={keeps}, zs={zs}", flush=True)

    # ---------------------------------------------------------------- 打分
    print("打分（全量，用于拟合；子集用于生成）...", flush=True)
    bm_all = score_all("bird_bm25", all_ex,
                       lambda ex: lexical_scores(ex, FusionConfig(lexical="bm25")))
    eb_all = score_all("bird_embedding", all_ex,
                       lambda ex: lexical_scores(ex, FusionConfig(lexical="embedding")))
    orders_all = fused_orders(all_ex, bm_all, eb_all)
    orders = [orders_all[i] for i in sub_idx]
    bm = [bm_all[i] for i in sub_idx]

    sc_by = {
        "bm25": bm,
        "fused": [{p: float(s[i]) for i, p in enumerate(pairs)}
                  for (pairs, s, o) in orders],
    }

    # ------------------------------------------------- System-One（laya）打分
    # 前面所有"System-One 更强"的证据都只停在 SRR —— 而 SRR 是代理指标，
    # 真正要报的是 EX。这一臂就是把 laya 的分数接进同一条生成链路。
    # 缓存是 laya_pareto 跑全量 1534 留下的，所以这里**不占 GPU**。
    laya_link = 0.0
    if a.laya_tag:
        laya_all = score_all(a.laya_tag, all_ex, None)     # 只读
        laya = [laya_all[i] for i in sub_idx]
        n_have = sum(1 for d in laya if d)
        if n_have != len(laya):
            print(f"  ⚠️ laya 缓存只覆盖 {n_have}/{len(laya)} 例，跳过该臂", flush=True)
        else:
            sc_by["laya"] = laya
            if a.laya_link > 0:
                laya_link = a.laya_link
            else:
                # 复算：state 一次 + 每条 typed question 一次，门控宽度要与打分时一致
                def _pick_tables(ex, top_t):
                    """BM25 先选表（与 laya_pareto.pick_tables_cheap 一致）。"""
                    sc = lexical_scores(ex, FusionConfig(lexical="bm25"))
                    best = {}
                    for (t, c), v in sc.items():
                        if v > best.get(t, -1e9):
                            best[t] = v
                    return [t for t, _ in sorted(best.items(),
                                                 key=lambda kv: -kv[1])[:top_t]]
                lk = []
                for ex in examples:
                    state, tables = build_state(ex.question_block(), ex.schema,
                                                SummaryConfig())
                    keep = (set(_pick_tables(ex, a.laya_top_tables))
                            if a.laya_top_tables > 0 else set(tables))
                    cols = [(t, c) for (t, c) in ex.schema.columns() if t in keep]
                    qs = [VIEWS[v].format(t=t, c=c) for v in views for t, c in cols]
                    lk.append(n_tokens(state) + sum(n_tokens(q) for q in qs))
                laya_link = float(np.mean(lk))
            print(f"  laya linking token: {laya_link:.0f}/查询"
                  f"（表级门控 top{a.laya_top_tables}）", flush=True)

    risk_link = 0.0
    if not a.skip_risk:
        be = build_backend("laya", cached=False, model_path=a.model_path or None,
                           max_len=a.max_len)

        def fn(ex):
            state, tables = build_state(ex.question_block(), ex.schema, SummaryConfig())
            cols = [(t, c) for (t, c) in ex.schema.columns() if t in set(tables)]
            if not cols:
                return {}
            pv = score_columns(be, state, cols, views, "mean", a.batch_size)
            return {e: float(np.mean([v[vk] for vk in views])) for e, v in pv.items()}

        sc_by["risk_ft"] = score_all(f"bird_risk_ft_v{'-'.join(views)}_L{a.max_len}_s0",
                                     examples, fn)
        close_backend(be)
        lk = []
        for ex in examples:
            state, tables = build_state(ex.question_block(), ex.schema, SummaryConfig())
            cols = [(t, c) for (t, c) in ex.schema.columns() if t in set(tables)]
            qs = [VIEWS[v].format(t=t, c=c) for v in views for t, c in cols]
            lk.append(n_tokens(state) + sum(n_tokens(q) for q in qs))
        risk_link = float(np.mean(lk))

    # ---------------------------------------------------------------- 预算预测器
    pm = load_prob_model(os.path.join(RESULTS, "lex_classifier_bird.json"))
    ps = ([prob_of(e, o, pm, lex_features) for e, (pairs, s, o) in zip(all_ex, orders_all)]
          if pm is not None else None)
    X = np.asarray([feats(e, s[o], pairs, ps[i] if ps is not None else None)
                    for i, (e, (pairs, s, o)) in enumerate(zip(all_ex, orders_all))])
    # target = 能达到全召回的最小深度（oracle）。dev 上 K 折，held-out 不参与自己的拟合。
    y = np.asarray([np.log1p(oracle_m(e, pairs, s, o, cfg, ms) or max(ms))
                    for e, (pairs, s, o) in zip(all_ex, orders_all)])
    logm_all = np.zeros(len(all_ex))
    logsig_all = np.zeros(len(all_ex))
    fold = np.random.default_rng(0).integers(0, a.folds, size=len(all_ex))
    for f in range(a.folds):
        tr_i = np.where(fold != f)[0]
        te_i = np.where(fold == f)[0]
        mu, sd = X[tr_i].mean(0), X[tr_i].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Zf = np.hstack([(X[tr_i] - mu) / sd, np.ones((len(tr_i), 1))])
        Zp = np.hstack([(X[te_i] - mu) / sd, np.ones((len(te_i), 1))])
        A = Zf.T @ Zf + a.ridge * np.eye(Zf.shape[1])
        w = np.linalg.solve(A, Zf.T @ y[tr_i])
        logm_all[te_i] = Zp @ w
        wv = np.linalg.solve(A, Zf.T @ np.log(np.maximum((y[tr_i] - Zf @ w) ** 2, 1e-6)))
        logsig_all[te_i] = 0.5 * (Zp @ wv)
    ssr = float(((y - logm_all) ** 2).mean())
    sst = float(max(1e-9, ((y - y.mean()) ** 2).mean()))
    cv_r2 = 1 - ssr / sst
    logm = logm_all[sub_idx]
    logsig = logsig_all[sub_idx]
    print(f"预算预测器: {len(all_ex)} 例 {a.folds} 折交叉, 样本外 R²={cv_r2:.4f}", flush=True)

    # ---------------------------------------------------------------- 组系统
    systems: dict = {}

    def add(name, kept_of, link=0.0):
        """kept_of(i, ex, pairs, order) -> kept 列列表（None = 全 schema）。"""
        d = {"prompt": [], "kept": [], "link": link}
        for i, (ex, (pairs, s, o)) in enumerate(zip(examples, orders)):
            kept = kept_of(i, ex, pairs, o)
            d["kept"].append(kept)
            d["prompt"].append(build_gen_prompt(
                ex.db_id, schema_prompt_text(ex.schema, columns=kept),
                ex.question, ex.evidence))
        systems[name] = d

    def topk_of(k, sc_list):
        def f(i, ex, pairs, o):
            sc = sc_list[i]
            v = np.asarray([float(sc.get(p, 0.0)) for p in pairs], dtype=float)
            order = np.argsort(-v, kind="stable")
            # ⚠️ 必须传**完整**排序：closure 内部自己截 top-k，而 pad 补的是
            # 该表内 top-k **之外**的次高列。传 order[:k] 会让 pad 没有候选，
            # C̄ 和 SRR 双双被压低（实测 k=15 时 C̄ 35.6→25.5、SRR 0.919→0.857）。
            return closure(ex.schema, [pairs[j] for j in order], k, cfg)
        return f

    def adapt_of(z):
        # 用融合分排序（顶部区分度最好），深度由预算预测器给
        def f(i, ex, pairs, o):
            m = int(round(float(np.expm1(np.clip(logm[i] + z * np.exp(logsig[i]), 0, 12)))))
            m = max(ms[0], min(ms[-1], m))
            return closure(ex.schema, [pairs[j] for j in o], m, cfg)   # 同上：传完整排序
        return f

    add("full", lambda i, ex, pairs, o: None)
    for k in keeps:
        add(f"bm25@{k}", topk_of(k, sc_by["bm25"]))
        add(f"fused@{k}", topk_of(k, sc_by["fused"]))
    for z in zs:
        add(f"adapt@z{z}", adapt_of(z))
    if "risk_ft" in sc_by:
        for k in keeps:
            add(f"risk_ft@{k}", topk_of(k, sc_by["risk_ft"]), link=risk_link)
    if "laya" in sc_by:
        for k in keeps:
            add(f"laya@{k}", topk_of(k, sc_by["laya"]), link=laya_link)

    # ---------------------------------------------------------------- 生成
    print("开始生成 SQL ...", flush=True)
    gen = OllamaGenerator(model=a.model, timeout_s=a.gen_timeout)
    gcache_path = os.path.join(CACHE, f"gen_{a.model.replace(':', '_')}.jsonl")
    gcache = {}
    if os.path.exists(gcache_path):
        for line in open(gcache_path, encoding="utf-8"):
            r = json.loads(line)
            gcache[r["h"]] = r["sql"]

    # ⚠️ 先冒烟：ollama 加载失败（CUDA_Host OOM、模型被卸载）时是 HTTP 200 + error
    # 载荷，症状是"每条都秒回空串"。不先探一次，整轮跑完才会发现 EX 全塌，
    # 而且失败结果已经写进缓存、永久污染后续重跑。
    smoke = gen.generate_sql("Answer with a single SQL: SELECT 1;")
    if not smoke.strip():
        raise SystemExit(
            f"❌ 生成器冒烟失败，停止（避免把空结果写进缓存污染后续重跑）\n"
            f"   model={a.model}  last_error={gen.last_error}\n"
            f"   常见原因：ollama 没能加载模型（CUDA_Host 内存不足 / 模型被卸载）。"
            f"先 `ollama ps` 或 curl /api/generate 确认能出结果再跑。")
    print(f"  冒烟 OK: {smoke.strip()[:60]!r}", flush=True)

    def gen_sql(prompt: str) -> str:
        h = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if h in gcache:
            return gcache[h]
        sql = gen.generate_sql(prompt)
        if not sql.strip():
            # 失败不缓存 —— 缓存了就永远命中这条垃圾
            raise RuntimeError(
                f"生成失败且拒绝缓存: {gen.last_error} "
                f"(已失败 {gen.stats.errors}/{gen.stats.calls} 次)")
        gcache[h] = sql
        with open(gcache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"h": h, "sql": sql}, ensure_ascii=False) + "\n")
        return sql

    for key, d in systems.items():
        t0 = time.time()
        d["gen_sql"] = [gen_sql(p) for p in d["prompt"]]
        d["prompt_tokens"] = float(np.mean([n_tokens(p) for p in d["prompt"]]))
        d["output_tokens"] = float(np.mean([n_tokens(s) for s in d["gen_sql"]]))
        print(f"  {key:<16} {len(d['prompt'])} 条 {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- 执行
    print("开始执行比对 ...", flush=True)
    out = {"bench": "BIRD-Dev", "n": len(examples), "generator": a.model,
           "tokenizer": TOKENIZER, "closure": cfg.tag(), "keeps": keeps, "zs": zs,
           "budget_cv_r2": round(cv_r2, 4), "systems": {}}
    gold_cache: dict = {}
    flags_by: dict = {}
    for key, d in systems.items():
        pairs = [(ex.db_id, ex.sql, s, key) for ex, s in zip(examples, d["gen_sql"])]
        fl: list = []
        rep = evaluate_execution(pairs, gold_cache=gold_cache, log_every=100, flags=fl)
        flags_by[key] = fl
        tot = d["prompt_tokens"] + d["output_tokens"] + float(d["link"])
        ncol = [len(k) if k is not None else len(ex.schema.columns())
                for k, ex in zip(d["kept"], examples)]
        srr = [1.0 if (k is None or all(g in set(k) for g in ex.gold.flat_pairs()))
               else 0.0 for k, ex in zip(d["kept"], examples)]
        out["systems"][key] = {
            "ex": round(rep.ex, 4), "ex_strict": round(rep.ex_strict, 4),
            "n": rep.n, "correct": rep.correct, "gold_fail": rep.gold_fail,
            "pred_fail": rep.pred_fail, "errors": rep.errors,
            "srr": round(float(np.mean(srr)), 4),
            "kept_cols": round(float(np.mean(ncol)), 2),
            "prompt_tokens": round(d["prompt_tokens"], 1),
            "output_tokens": round(d["output_tokens"], 1),
            "link_tokens": round(float(d["link"]), 1),
            "total_tokens": round(tot, 1),
        }
        v = out["systems"][key]
        print(f"  {key:<16} EX={v['ex']:.4f} ({rep.correct}/{rep.n - rep.gold_fail}) "
              f"SRR={v['srr']:.4f} C̄={v['kept_cols']:.1f} tok={tot:.0f} "
              f"pred_fail={rep.pred_fail}", flush=True)

    ref = out["systems"].get("full", {})
    if ref:
        for key, v in out["systems"].items():
            if key == "full":
                v["tep_vs_full"] = None
                continue
            t = tep(v["ex"], v["total_tokens"], ref["ex"], ref["total_tokens"])
            v["tep_vs_full"] = None if (t != t) else round(t, 4)
            v["token_saving_vs_full"] = round(1 - v["total_tokens"] / ref["total_tokens"], 4)
    pts = [(v["total_tokens"], v["ex"]) for v in out["systems"].values()]
    names = list(out["systems"].keys())
    out["pareto"] = [{"system": names[i], "tokens": round(pts[i][0], 1),
                      "ex": round(pts[i][1], 4)} for i in pareto_front(pts)]

    print(f"\n== token-EX 帕累托前沿（生成器 {a.model}）==")
    for p in out["pareto"]:
        print(f"  {p['system']:<16} tokens={p['tokens']:>8.0f}  EX={p['ex']:.4f}")

    # ---------------------------------------------------------------- 显著性
    # 压缩系统和 full 的 EX 只差 2-3 条，不做配对检验根本分不出是噪声还是真差异。
    # 这里用配对 bootstrap + McNemar（两套检验互相印证，避免只报好看的那个）。
    def _pair(x, y):
        """只在两边都不是 None（gold 可执行）的样本上比。"""
        return [(u, v) for u, v in zip(x, y) if u is not None and v is not None]

    print(f"\n== 显著性：各系统 vs full（配对 bootstrap + McNemar）==")
    out["tests_vs_full"] = {}
    for key in systems:
        if key == "full" or "full" not in flags_by:
            continue
        xy = _pair(flags_by["full"], flags_by[key])
        if len(xy) < 10:
            continue
        a_ = np.array([u for u, v in xy], dtype=float)
        b_ = np.array([v for u, v in xy], dtype=float)
        pb = paired_bootstrap(a_, b_)
        n01 = sum(1 for u, v in xy if u == 1 and v == 0)   # full 对、压缩错
        n10 = sum(1 for u, v in xy if u == 0 and v == 1)   # full 错、压缩对
        # McNemar 精确二项检验（n01+n10 为条件）
        from math import comb
        ndisc = n01 + n10
        if ndisc > 0:
            k = min(n01, n10)
            p_exact = min(1.0, 2.0 * sum(comb(ndisc, i) for i in range(k + 1))
                          / float(2 ** ndisc))
        else:
            p_exact = 1.0
        out["tests_vs_full"][key] = {
            "n_paired": len(xy), "diff": round(pb["diff"], 4),
            "boot_ci": [round(pb["ci_lo"], 4), round(pb["ci_hi"], 4)],
            "p_boot": round(pb["p_boot"], 4),
            "mcnemar_full_only": n01, "mcnemar_comp_only": n10,
            "p_mcnemar": round(p_exact, 4),
        }
        print(f"  {key:<16} n={len(xy):<5} ΔEX={pb['diff']:+.4f} "
              f"CI=[{pb['ci_lo']:+.4f},{pb['ci_hi']:+.4f}] p_boot={pb['p_boot']:.4f} | "
              f"McNemar {n01}/{n10} p={p_exact:.4f}")
    print("  （Δ<0 是比 full 差。两边 p 都 >0.05 就说明『统计不可区分』，"
          "这时该报的是省了多少 token，不是 EX 涨了多少。）")

    # System-One 到底有没有真的提高 EX？前面所有证据都停在 SRR，这里才是答案。
    # 同一个 k 比（同预算、只换排序器），再按"不比 laya 更贵的 fused"比一次（同成本）。
    if "laya" in sc_by:
        print(f"\n== 显著性：System-One(laya) vs 纯词法融合（同 k / 同成本）==")
        out["tests_laya_vs_fused"] = {}
        for k in keeps:
            lk, fk = f"laya@{k}", f"fused@{k}"
            if lk not in flags_by or fk not in flags_by:
                continue
            xy = _pair(flags_by[fk], flags_by[lk])
            if len(xy) < 10:
                continue
            a_ = np.array([u for u, v in xy], dtype=float)
            b_ = np.array([v for u, v in xy], dtype=float)
            pb = paired_bootstrap(a_, b_)
            out["tests_laya_vs_fused"][f"k={k}"] = {
                "n_paired": len(xy), "diff": round(pb["diff"], 4),
                "boot_ci": [round(pb["ci_lo"], 4), round(pb["ci_hi"], 4)],
                "p_boot": round(pb["p_boot"], 4),
                "fused_ex": out["systems"][fk]["ex"],
                "laya_ex": out["systems"][lk]["ex"],
                "fused_tokens": out["systems"][fk]["total_tokens"],
                "laya_tokens": out["systems"][lk]["total_tokens"],
            }
            print(f"  k={k:<3} fused EX={out['systems'][fk]['ex']:.4f} "
                  f"(tok {out['systems'][fk]['total_tokens']:.0f})  vs  "
                  f"laya EX={out['systems'][lk]['ex']:.4f} "
                  f"(tok {out['systems'][lk]['total_tokens']:.0f})  "
                  f"Δ={pb['diff']:+.4f} p={pb['p_boot']:.4f}")
        print("  （Δ>0 是 System-One 赢。注意 laya 的 token 含 1788 linking 成本，"
              "所以『同 k』不等于『同成本』；真正的同成本对照看上面的总 token。）")

    print(f"\n生成器统计: {gen.stats.as_dict()}")

    path = a.out or os.path.join(RESULTS, f"e2e_bird_n{len(examples)}_"
                                          f"{a.model.replace(':', '-')}_{cfg.tag()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
