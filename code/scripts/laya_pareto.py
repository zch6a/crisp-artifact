"""System-One（laya）打分 vs 纯词法融合打分：同一套结构闭包下的 SRR / token 前沿。

为什么必须跑这个：
    laya 在 BIRD 域内微调后的 dev AUROC = **0.9905**，而 bm25⊕emb 融合是 **0.8772**。
    AUROC 差了 11 个点，但前面已经吃过一次亏 —— **AUROC 高 ≠ top-k 好**
    （逻辑回归打分器 AUROC 0.9058 却比融合的 0.8772 在 SRR@C=35 上更差：
     0.8481 vs 0.8814）。所以不能拿 AUROC 下结论，必须把 laya 的分数
    真正喂进 selection.closure 里，在同一套闭包配置下画出 (token, SRR) 前沿再比。

设计：
    * 打分缓存按 (模型, 模板, max_len) 打 tag —— 不同模型的分数绝不能混进同一个文件，
      否则第二次跑会命中"看起来算过"的分数，两组实验静默变成同一组。
    * 闭包配置串与 pareto_closure / budget_predictor 完全一致，三张表才能对得上。
    * 纯词法侧（bm25、bm25⊕emb）同时跑，作为同口径对照。
    * 表级门控：用便宜的 BM25 先选表，只对入选表的列跑 laya —— 这既省算力，
      也是实际系统里必须做的事（否则 O(全库列数) 的打分在超大库上会输给全 schema 直喂）。

输出：每个 (打分器, 配置, k) 的 (SRR, C̄, tokens)，以及各打分器的帕累托前沿对比。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import jevschema.paths as _P  # noqa: E402  (env CRISP_DATA / CRISP_MODELS)

import numpy as np  # noqa: E402

from jevschema.backends import build_backend, close_backend  # noqa: E402
from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import mean_ci, paired_bootstrap  # noqa: E402
from jevschema.risk import VIEWS, score_columns  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import SummaryConfig, bm25_scores, build_state  # noqa: E402
from jevschema.tokens import make_stage_tokens, n_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")


def _cache_path(tag: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{tag}.jsonl")


def _load(tag, examples, fn):
    p = _cache_path(tag)
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    if fn is None:
        return [store.get(e.ex_id, {}) for e in examples], len(store)
    todo = [e for e in examples if e.ex_id not in store]
    if todo:
        # 必须**流式**写：fn 若返回生成器就逐条落盘。
        # 初版写成 zip(todo, fn(todo)) —— 生成器会被先整个求值，中途挂掉（segfault /
        # 某个超大库卡死）时前面一千条全丢，只能从头再来。laya 打分要跑一小时以上，
        # 不续跑根本跑不完。
        with open(p, "a", encoding="utf-8") as f:
            for ex, sc in zip(todo, fn(todo)):
                store[ex.ex_id] = sc
                f.write(json.dumps({"id": ex.ex_id,
                                    "s": {"|".join(k): v for k, v in sc.items()}},
                                   ensure_ascii=False) + "\n")
                f.flush()
    return [store[e.ex_id] for e in examples], len(store)


def bm25_of(ex):
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def _norm(d, how="max"):
    v = np.asarray(list(d.values()), dtype=float)
    if how == "max":
        hi = v.max()
        return {k: float(x) for k, x in
                zip(d.keys(), v / hi if hi > 0 else np.zeros_like(v))}
    lo, hi = v.min(), v.max()
    return {k: float(x) for k, x in zip(d.keys(), (v - lo) / max(1e-9, hi - lo))}


def pick_tables_cheap(ex, top_t=6, bm=None):
    """用 BM25 先选表：laya 只对入选表的列打分。

    ⚠️ `--top-tables 6` 是之前所有 laya 评测的默认值，但它的**表召回只有 0.9570**
    （T=8 → 0.9915，T=10 → 0.9961）—— 也就是说门控先漏掉 4.3% 的题，
    laya 报的 SRR 上限被这个门控卡死，**不是它排序能力的上限**。默认已改成 10。
    """
    sc = bm if bm is not None else bm25_of(ex)
    best = {}
    for (t, c), v in sc.items():
        if v > best.get(t, -1e9):
            best[t] = v
    return [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_t]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model-path", default=os.path.join(_P.MODELS, "laya_ft_bird"))
    ap.add_argument("--views", default="ask")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--top-tables", type=int, default=10,
                    help="问题门控宽度（对哪些表的列发 typed question）。"
                         "默认从 6 改成 10：T=6 表召回仅 0.9570，是人为天花板")
    ap.add_argument("--state-tables", type=int, default=0,
                    help=">0 时 **state 也门控**：只把 BM25 top-T 表写进 state。"
                         "这是让 System-One 成本与库规模脱钩的唯一办法 —— "
                         "不门控时 state 本身 ≈ 全 schema prompt（比值 0.905）")
    ap.add_argument("--cfgs", default="f1p5s15k1,f1p0s0k0,f1p8k1")
    ap.add_argument("--ks", default="8,10,12,15,18,20,25,30,35,40,50,60")
    ap.add_argument("--gate-cols", type=int, default=0,
                    help=">0 时只对 BM25 前 N 列跑 laya（更强的门控，更省）")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfgs = [parse_closure(g) for g in a.cfgs.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    views = tuple(a.views.split(","))

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(ex)}  模型={a.model_path}  views={views}  "
          f"max_len={a.max_len}  表级门控 top{a.top_tables}", flush=True)

    bm, _ = _load("bird_bm25", ex, lambda todo: [bm25_of(e) for e in todo])
    eb, n_emb = _load("bird_embedding", ex, None)   # 只读：缺就是缺
    have_emb = any(len(d) > 0 for d in eb)

    fused = []
    for e, b, d in zip(ex, bm, eb):
        pairs = e.schema.columns()
        bn = _norm({q: float(b.get(q, 0.0)) for q in pairs}, "max")
        dn = (_norm({q: float(d.get(q, 0.0)) for q in pairs}, "minmax")
              if have_emb else {q: 0.0 for q in pairs})
        fused.append({q: 0.5 * bn[q] + 0.5 * dn[q] for q in pairs})

    # ---------------------------------------------------------------- laya 打分
    tag = (f"bird_layapareto_{os.path.basename(a.model_path)}_"
           f"v{'-'.join(views)}_L{a.max_len}_t{a.top_tables}"
           f"{'_g' + str(a.gate_cols) if a.gate_cols else ''}"
           f"{'_st' + str(a.state_tables) if a.state_tables else ''}")
    cached, n_cached = _load(tag, ex, None)
    if n_cached < len(ex):
        todo = [e for e, c in zip(ex, cached) if not c]
        print(f"laya 打分：{len(todo)} 例待算（已缓存 {n_cached}）", flush=True)
        be = build_backend("laya", cached=False, model_path=a.model_path, max_len=a.max_len)

        def run(todo):
            """生成器：算一条落一条，配合 _load 的流式写，挂了也能续。"""
            for i, e in enumerate(todo):
                b = bm25_of(e)
                qt = (set(pick_tables_cheap(e, a.top_tables, bm=b))
                      if a.top_tables > 0 else None)
                st_t = (pick_tables_cheap(e, a.state_tables, bm=b)
                        if a.state_tables > 0 else None)
                state, tables = build_state(e.question_block(), e.schema,
                                            SummaryConfig(), tables=st_t)
                keep = qt if qt is not None else set(tables)
                cols = [(t, c) for (t, c) in e.schema.columns() if t in keep]
                if a.gate_cols > 0:
                    cols = sorted(cols, key=lambda p: -b.get(p, 0.0))[:a.gate_cols]
                if not cols:
                    yield {}
                    continue
                pv = score_columns(be, state, cols, views, "mean", a.batch_size)
                yield {k: float(np.mean([v[vk] for vk in views])) for k, v in pv.items()}
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(todo)}", flush=True)

        _load(tag, ex, lambda todo: run(todo))
        close_backend(be)
        cached, _ = _load(tag, ex, None)
    laya = cached

    # laya 的 linking token（state 一次 + N 条 typed question）
    link_tok = []
    for e in ex:
        b = bm25_of(e)
        qt = (set(pick_tables_cheap(e, a.top_tables, bm=b))
              if a.top_tables > 0 else None)
        st_t = (pick_tables_cheap(e, a.state_tables, bm=b)
                if a.state_tables > 0 else None)
        state, tables = build_state(e.question_block(), e.schema,
                                    SummaryConfig(), tables=st_t)
        keep = qt if qt is not None else set(tables)
        cols = [(t, c) for (t, c) in e.schema.columns() if t in keep]
        if a.gate_cols > 0:
            cols = sorted(cols, key=lambda p: -b.get(p, 0.0))[:a.gate_cols]
        qs = [VIEWS[v].format(t=t, c=c) for v in views for t, c in cols]
        link_tok.append(n_tokens(state) + sum(n_tokens(q) for q in qs))
    link_mean = float(np.mean(link_tok))
    print(f"laya linking token: {link_mean:.0f}/查询", flush=True)

    # ---------------------------------------------------------------- 前沿
    def front_for(name, sc_list, link=0.0):
        rows = []
        for cfg in cfgs:
            for k in ks:
                srrs, cb, tk = [], [], []
                for e, sc in zip(ex, sc_list):
                    pairs = e.schema.columns()
                    v = np.asarray([float(sc.get(p, 0.0)) for p in pairs], dtype=float)
                    order = np.argsort(-v, kind="stable")
                    # 传完整排序：closure 内部截 top-k，pad 需要从 top-k 之外取次高列
                    kept = closure(e.schema, [pairs[j] for j in order], k, cfg)
                    srrs.append(1.0 if all(g in set(kept)
                                           for g in e.gold.flat_pairs()) else 0.0)
                    cb.append(len(kept))
                    tk.append(make_stage_tokens(e.schema, e.question, e.evidence,
                                                kept_columns=kept).gen_prompt_tokens)
                rows.append({"scorer": name, "cfg": cfg.tag(), "k": k,
                             "srr": float(np.mean(srrs)),
                             "cbar": float(np.mean(cb)),
                             "gen_tokens": float(np.mean(tk)),
                             "total_tokens": float(np.mean(tk)) + link,
                             "ci": [round(x, 4) for x in mean_ci(srrs)[1:3]],
                             "_srr": srrs})
        return rows

    print(f"\n{'打分器':<10}{'配置':<12}{'k':>4}{'C̄':>7}{'tokens':>9}{'SRR':>9}  95%CI")
    all_rows = []
    for name, sc_list, link in (("bm25", bm, 0.0),
                                ("fused", fused, 0.0),
                                ("laya", laya, link_mean)):
        rows = front_for(name, sc_list, link)
        all_rows += rows
        pts = sorted(rows, key=lambda r: (r["total_tokens"], -r["srr"]))
        fr, best = [], -1.0
        for r in pts:
            if r["srr"] > best + 1e-9:
                fr.append(r); best = r["srr"]
        print(f"\n-- {name} 帕累托前沿（{len(fr)} 点，linking={link:.0f} tok）--")
        for r in fr:
            print(f"{name:<10}{r['cfg']:<12}{r['k']:>4}{r['cbar']:>7.1f}"
                  f"{r['total_tokens']:>9.0f}{r['srr']:>9.4f}"
                  f"  [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]")

    # ---------------------------------------------------------------- 等 token 对照
    print("\n== 等 token 对照：laya vs 纯词法（配对 bootstrap）==")
    laya_rows = [r for r in all_rows if r["scorer"] == "laya"]
    for base in ("fused", "bm25"):
        base_rows = [r for r in all_rows if r["scorer"] == base]
        print(f"\n  -- vs {base} --")
        for r in sorted(laya_rows, key=lambda r: r["total_tokens"]):
            cand = [b for b in base_rows if b["total_tokens"] <= r["total_tokens"] + 1e-9]
            if not cand:
                continue
            b = max(cand, key=lambda z: z["srr"])
            # n_boot 降到 2000：默认 10000 会分配 10000×1534 的 int64（117 MB），
            # 本机 16GB 内存在跑完 laya 打分后已经很紧，直接 _ArrayMemoryError。
            pb = paired_bootstrap(b["_srr"], r["_srr"], n_boot=2000)
            flag = "  << laya 赢" if pb["diff"] > 0 and pb["p_boot"] < 0.05 else (
                "  << laya 输" if pb["diff"] < 0 and pb["p_boot"] < 0.05 else "")
            print(f"    laya {r['cfg']:<10} k={r['k']:<3} tok={r['total_tokens']:6.0f} "
                  f"SRR={r['srr']:.4f}  vs {base} {b['cfg']:<10} k={b['k']:<3} "
                  f"tok={b['total_tokens']:6.0f} SRR={b['srr']:.4f}  "
                  f"Δ={pb['diff']:+.4f} p={pb['p_boot']:.4f}{flag}")

    path = a.out or os.path.join(RESULTS, "laya_pareto_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(ex), "model": a.model_path, "views": list(views),
                   "max_len": a.max_len, "top_tables": a.top_tables,
                   "laya_link_tokens": round(link_mean, 1),
                   "rows": [{kk: vv for kk, vv in r.items() if kk != "_srr"}
                            for r in all_rows]}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
