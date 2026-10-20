"""BIRD-Dev 主评测：SRR / 软召回 / 分阶段 token / TEP / 帕累托。

为什么单独开一个脚本而不是复用 compare_scorers.py：
    BIRD 才是能和 AutoLink(97.4% SRR) / EviLink / LinkAlign / EllieSQL / BAP-SQL
    对齐的基准。这里除了召回还必须产出**成本维度**（linking 段 vs 生成段 token），
    否则"省了多少"这件事无法验证 —— 而这正是本课题的核心主张。

参与的打分器：
    bm25 / embedding     无监督基线（$0，作为地板）
    risk_ft              Spider-train 微调的 laya（跨集零样本迁移）
    risk_base            未微调 laya（对照组）
    full                 不压缩（全 schema 直喂），作为 token 参考系

输出的每个方法在每个过滤率上给出：
    srr          —— 整题全对的严格召回率（AutoLink 口径）
    recall       —— 逐需求软召回
    table_recall —— 表级召回
    filter_rate  —— 被删列占比
    link_tokens / gen_tokens / total_tokens / saving
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
import jevschema.paths as _P  # noqa: E402  (env CRISP_DATA / CRISP_MODELS)

import numpy as np  # noqa: E402

from jevschema.backends import build_backend, close_backend  # noqa: E402
from jevschema.bird_data import load_bird_dev, load_bird_schemas, gold_quality, schema_stats  # noqa: E402
from jevschema.fusion import FusionConfig, lexical_scores  # noqa: E402
from jevschema.metrics import (auroc, mean_ci, paired_bootstrap, recall_at_filter_rate,  # noqa: E402
                               strict_recall, tep, pareto_front)
from jevschema.risk import VIEWS, score_columns  # noqa: E402
from jevschema.selection import ClosureConfig, closure, parse_closure  # noqa: E402
from jevschema.summary import SummaryConfig, build_state  # noqa: E402
from jevschema.tokens import TOKENIZER, make_stage_tokens, n_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")
DEFAULT_RATES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


# ----------------------------------------------------------------------------- 缓存
def _cache_path(tag: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{tag}.jsonl")


def load_cached(tag: str, ex_ids: list) -> dict:
    p = _cache_path(tag)
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    return out


def append_cache(tag: str, rows: list) -> None:
    with open(_cache_path(tag), "a", encoding="utf-8") as f:
        for ex_id, sc in rows:
            f.write(json.dumps({"id": ex_id,
                                "s": {"|".join(k): v for k, v in sc.items()}},
                               ensure_ascii=False) + "\n")


def score_all(tag: str, examples: list, fn, log_every: int = 100) -> list:
    """带缓存的批量打分。fn(example) -> {pair: score}。"""
    store = load_cached(tag, [e.ex_id for e in examples])
    todo = [e for e in examples if e.ex_id not in store]
    if todo:
        t0 = time.time()
        buf = []
        for i, ex in enumerate(todo):
            sc = fn(ex)
            store[ex.ex_id] = sc
            buf.append((ex.ex_id, sc))
            if len(buf) >= 20:
                append_cache(tag, buf)
                buf = []
            if (i + 1) % log_every == 0:
                print(f"    {tag} {i+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
        if buf:
            append_cache(tag, buf)
        print(f"  {tag}: 新算 {len(todo)} 例 {time.time()-t0:.1f}s")
    else:
        print(f"  {tag}: 全部命中缓存 ({len(examples)} 例)")
    return [store[e.ex_id] for e in examples]


# ----------------------------------------------------------------------------- 评测
def _legacy_cfg(expand: str) -> ClosureConfig:
    """旧 --expand 字符串映射到 v2 的 ClosureConfig（保持既有脚本行为不变）。"""
    if expand == "none":
        return ClosureConfig(fk=False)
    if expand == "fk":
        return ClosureConfig(fk=True)
    if expand == "table":
        return ClosureConfig(fk=False, frac=1.0)
    if expand == "fk+table":
        return ClosureConfig(fk=True, frac=1.0)
    raise ValueError(f"未知 expand: {expand}")


def _select(sc: dict, pairs: list, budget: int, schema, cfg: ClosureConfig) -> list:
    """top-k → 结构补全。

    budget 是"截断值 k"，不是最终保留列数 —— 表内补齐和 FK 闭包都会往上加列，
    最终列数（C̄）单独统计。两者必须分开报，否则跨策略比较是不公平的。
    """
    s = np.asarray([float(sc.get(p, 0.0)) for p in pairs], dtype=float)
    n_keep = max(1, min(len(pairs), int(budget)))
    order = np.argsort(-s, kind="stable")
    return closure(schema, [pairs[i] for i in order], n_keep, cfg)


def evaluate(name: str, per_ex: list, examples: list, budgets: list,
             with_samples: int = 0, mode: str = "rate", expand: str = "none",
             closure_spec: str = "") -> dict:
    """每个打分器走同一个评测函数：AUROC + 各预算下的 SRR/召回/C̄/token。

    mode=rate  预算是过滤率（按库大小自适应）
    mode=cols  预算是**绝对列数** —— 这才是能和 AutoLink 的 C̄=35.8 对齐的口径，
               因为"过滤 50%"在 11 列的库和 199 列的库里完全不是一回事。

    closure_spec 形如 'f1p5s10'（见 selection.parse_closure）。给了就走 v2 补全，
    没给就退回旧的 expand 字符串（none/fk/table/fk+table）。
    """
    cfg = parse_closure(closure_spec) if closure_spec else _legacy_cfg(expand)
    aucs = []
    per_b = {b: {"recall": [], "srr": [], "tbl": [], "gen": [], "link": [],
                 "kept": [], "state_tok": []} for b in budgets}
    for ex, sc in zip(examples, per_ex):
        pairs = list(ex.schema.columns())
        gold = ex.gold
        gold_flat = gold.flat_pairs() if gold else set()
        s = np.asarray([float(sc.get(p, 0.0)) for p in pairs], dtype=float)
        y = np.asarray([1 if p in gold_flat else 0 for p in pairs], dtype=int)
        aucs.append(auroc(s, y))
        state_text = ex.question_block()
        for b in budgets:
            if mode == "cols":
                n = int(b)
            else:
                n = int(round((1 - float(b)) * len(s)))
            kept = _select(sc, pairs, n, ex.schema, cfg)
            st = make_stage_tokens(ex.schema, ex.question, ex.evidence,
                                   kept_columns=kept, state_text=state_text,
                                   questions=[], with_samples=with_samples)
            per_b[b]["recall"].append(gold.recall(set(kept)) if gold else 1.0)
            per_b[b]["srr"].append(strict_recall(gold, kept))
            per_b[b]["tbl"].append(gold.table_recall({t for t, _ in kept}) if gold else 1.0)
            per_b[b]["gen"].append(st.gen_prompt_tokens)
            per_b[b]["link"].append(st.linking_tokens)
            per_b[b]["kept"].append(len(kept))
            per_b[b]["state_tok"].append(n_tokens(state_text))

    out = {"auroc": float(np.nanmean(aucs)), "rates": {}}
    for b in budgets:
        d = per_b[b]
        out["rates"][str(b)] = {
            "srr": float(np.mean(d["srr"])),
            "srr_ci": [round(float(x), 4) for x in mean_ci(d["srr"])],
            "recall": float(np.mean(d["recall"])),
            "recall_ci": [round(float(x), 4) for x in mean_ci(d["recall"])],
            "table_recall": float(np.mean(d["tbl"])),
            "avg_kept_columns": float(np.mean(d["kept"])),     # C̄，与 AutoLink 同口径
            "gen_tokens": float(np.mean(d["gen"])),
            "link_tokens": float(np.mean(d["link"])),
            "total_tokens": float(np.mean(d["gen"])) + float(np.mean(d["link"])),
            "_srr": d["srr"], "_recall": d["recall"], "_gen": d["gen"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = 全量 1534")
    ap.add_argument("--stride", type=int, default=1,
                    help="等间隔抽样。dev.json 按库排序，取前 N 条会整段落在前几个库上；"
                         "stride>1 才能覆盖全部 11 个库。")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="laya")
    ap.add_argument("--model-path", default=os.path.join(_P.MODELS, "laya_ft_schema"))
    ap.add_argument("--views", default="ask")
    ap.add_argument("--combine", default="mean")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rates", default=",".join(str(r) for r in DEFAULT_RATES))
    ap.add_argument("--mode", default="rate", choices=["rate", "cols"],
                    help="cols = 按绝对列数给预算（与 AutoLink 的 C̄ 同口径）")
    ap.add_argument("--expand", default="none", choices=["none", "fk", "table", "fk+table"],
                    help="top-k 之后的结构化补全：外键端点列 / 整表")
    ap.add_argument("--gate", default="none", choices=["none", "bm25", "risk"],
                    help="表级门控：先选表，只给入选表的列打分。"
                         "没有门控时成本是 O(全库列数) —— 3000 列的库上会比全 schema 直喂还贵，"
                         "门控才是本方法能 scale 的前提。")
    ap.add_argument("--keep-tables", type=int, default=5,
                    help="门控保留多少张表（含最相似的若干张）")
    ap.add_argument("--fuse-alpha", type=float, default=0.0,
                    help=">0 时额外评测 (1-α)·risk + α·BM25（BM25 做样本内 max 归一）。"
                         "Spider 上微调后融合无增益，但那是同域；跨域迁移时融合更可能有用。")
    ap.add_argument("--closure", default="",
                    help="结构补全 v2 简写：f{0,1}p<数>s<数>r<比例>，如 f1p5s10 / f1r0.5。"
                         "给了它就用 selection.closure（含入选表内补齐），忽略 --expand。")
    ap.add_argument("--lexical", default="bm25", choices=["bm25", "embedding", "both"],
                    help="无监督词法信号。both = bm25+embedding 各归一后 0.5/0.5 加权，"
                         "BIRD-Dev 全量上 AUROC 0.8465 -> 0.8772、SRR@C=35 0.8814 -> 0.9074。")
    ap.add_argument("--with-samples", type=int, default=0)
    ap.add_argument("--skip", default="", help="逗号分隔，跳过某些打分器")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    parse_b = (lambda x: int(x)) if a.mode == "cols" else (lambda x: float(x))
    rates = [parse_b(x) for x in a.rates.split(",")]
    budgets = rates
    views = tuple(a.views.split(","))
    skip = {s for s in a.skip.split(",") if s}
    extra = dict(mode=a.mode, expand=a.expand, closure_spec=a.closure)

    schemas = load_bird_schemas()
    if a.with_samples > 0:
        from jevschema.bird_data import enrich_sample_values
        n = enrich_sample_values(schemas, k=a.with_samples)
        print(f"示例值回填: {n}/{len(schemas)} 库")
    examples = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    if a.stride > 1:
        examples = examples[::a.stride]
    print(f"BIRD-Dev 样例 {len(examples)} / {len(schemas)} 库 (stride={a.stride})")
    for r in schema_stats(schemas):
        print("   ", r)
    q = gold_quality(examples)
    print("  gold:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()})

    out = {"bench": "BIRD-Dev", "n": len(examples), "rates": rates,
           "views": list(views), "max_len": a.max_len, "tokenizer": TOKENIZER,
           "with_samples": a.with_samples, "stride": a.stride, "scorers": {}}
    # 截断率必须显式报：state 被右侧截断意味着模型根本没看到后半张 schema，
    # 这时候任何召回数字都只是"信息缺失"的度量，不是方法好坏的度量。
    try:
        from transformers import AutoTokenizer
        _tk = AutoTokenizer.from_pretrained(
            os.path.join(a.model_path or os.path.join(_P.MODELS, "laya_ft_schema"), "tokenizer"))
        room = a.max_len - 52
        lens = [len(_tk.encode(build_state(e.question_block(), e.schema,
                                           SummaryConfig(with_samples=a.with_samples))[0],
                               add_special_tokens=False)) for e in examples]
        out["state_tokens_laya"] = {
            "mean": round(float(np.mean(lens)), 1),
            "p90": float(np.percentile(lens, 90)), "max": float(np.max(lens)),
            "room": room, "truncation_rate": round(float(np.mean([l > room for l in lens])), 4),
        }
        print(f"  state 长度(laya tok): mean {np.mean(lens):.0f} p90 "
              f"{np.percentile(lens,90):.0f} max {np.max(lens)} | room {room} | "
              f"截断率 {out['state_tokens_laya']['truncation_rate']:.3f}")
    except Exception as e:                                    # noqa: BLE001
        print(f"  (截断率统计跳过: {type(e).__name__})")

    # ------------------------------------------------------------ 无监督基线
    lex_kinds = ["bm25", "embedding"] if a.lexical in ("bm25", "embedding") else \
        [a.lexical, "bm25", "embedding"]
    for kind in dict.fromkeys(lex_kinds):
        if kind in skip:
            continue
        tag = f"bird_{kind}"
        out["scorers"][kind] = evaluate(
            kind, score_all(tag, examples,
                            lambda ex: lexical_scores(ex, FusionConfig(lexical=kind))),
            examples, budgets, a.with_samples, **extra)

    # ------------------------------------------------------------ System-One
    backends = []
    if "risk_ft" not in skip:
        backends.append(("risk_ft", a.model_path or None))
    if "risk_base" not in skip:
        backends.append(("risk_base", None))
    sc_store: Dict[str, list] = {}
    for name, mp in backends:
        # tag 必须带模型标识：否则 spider 微调模型和 BIRD 微调模型会写进同一个缓存文件，
        # 第二次跑直接命中"看起来算过"的分数 —— 两组实验数字会静默变成同一组。
        mkey = os.path.basename((mp or a.backend).rstrip("\\/"))[:14] or "base"
        tag = (f"bird_{name}_{mkey}_v{'-'.join(views)}_L{a.max_len}_s{a.with_samples}"
               f"_g{a.gate}{a.keep_tables if a.gate != 'none' else ''}")
        be = build_backend(a.backend, cached=False, model_path=mp, max_len=a.max_len)

        # 门控选出的表（每个样例一份，打分和算 token 都用它，保证口径一致）
        gate_tables: Dict[str, List[str]] = {}

        def pick_tables(ex, be=be):
            if a.gate == "none":
                return ex.schema.table_names()
            if ex.ex_id in gate_tables:
                return gate_tables[ex.ex_id]
            if a.gate == "bm25":
                from jevschema.summary import bm25_scores
                docs = [f"{t.name} " + " ".join(c.name for c in t.columns)
                        for t in ex.schema.tables]
                sc = bm25_scores(ex.question_block(), docs)
                order = sorted(range(len(docs)), key=lambda i: -sc[i])
                keep = [ex.schema.tables[i].name for i in order[:a.keep_tables]]
            else:
                from jevschema.risk import TABLE_VIEWS, score_tables
                state, _ = build_state(ex.question_block(), ex.schema,
                                       SummaryConfig(with_samples=a.with_samples))
                ts = {v: score_tables(be, state, ex.schema.table_names(), [v],
                                      a.combine, a.batch_size)
                      for v in views if v in TABLE_VIEWS}
                if not ts:
                    keep = ex.schema.table_names()
                else:
                    merged = {t: float(np.mean([d[t] for d in ts.values()]))
                              for t in ex.schema.table_names()}
                    keep = sorted(merged, key=lambda t: -merged[t])[:a.keep_tables]
            gate_tables[ex.ex_id] = keep
            return keep

        def fn(ex, be=be):
            state, _ = build_state(ex.question_block(), ex.schema, SummaryConfig(
                with_samples=a.with_samples))
            keep = set(pick_tables(ex))
            cols = [(t, c) for (t, c) in ex.schema.columns() if t in keep]
            if not cols:
                return {}
            pv = score_columns(be, state, cols, views, a.combine, a.batch_size)
            return {e: float(np.mean([v[vk] for vk in views])) for e, v in pv.items()}

        sc = score_all(tag, examples, fn)
        sc_store[name] = sc
        close_backend(be)
        # 补 linking 段 token：state 只付一次，表级问题 + 列级问题逐条累加
        st_list = []
        gate_recall = []
        for ex in examples:
            state, _ = build_state(ex.question_block(), ex.schema,
                                   SummaryConfig(with_samples=a.with_samples))
            keep = set(pick_tables(ex))
            cols = [(t, c) for (t, c) in ex.schema.columns() if t in keep]
            qs = [VIEWS[v].format(t=t, c=c) for v in views for t, c in cols]
            if a.gate == "risk":
                from jevschema.risk import TABLE_VIEWS
                qs += [TABLE_VIEWS[v].format(t=t) for v in views if v in TABLE_VIEWS
                       for t in ex.schema.table_names()]
            st_list.append((state, qs))
            if ex.gold:
                gate_recall.append(ex.gold.table_recall(keep))
        out["scorers"][name] = evaluate(name, sc, examples, budgets, a.with_samples, **extra)
        out["scorers"][name]["gate_table_recall"] = (
            round(float(np.mean(gate_recall)), 4) if gate_recall else None)
        out["scorers"][name]["gate"] = a.gate if a.gate != "none" else None
        for r in budgets:
            link = [n_tokens(s) + sum(n_tokens(q) for q in qs) for s, qs in st_list]
            out["scorers"][name]["rates"][str(r)]["link_tokens"] = float(np.mean(link))
            out["scorers"][name]["rates"][str(r)]["total_tokens"] = (
                float(np.mean(link)) + out["scorers"][name]["rates"][str(r)]["gen_tokens"])

    # ------------------------------------------------------------ 融合
    if a.fuse_alpha > 0 and a.lexical not in skip:
        from jevschema.fusion import _normalize
        # CPU 侧最佳骨干是 bm25+emb 融合（AUROC 0.8772），不是裸 bm25。
        # --lexical both 让 System-One 与纯词法基线共用同一条融合骨干，比较才公平。
        lx_kind = a.lexical
        bm = score_all(f"bird_{lx_kind}", examples,
                       lambda ex: lexical_scores(ex, FusionConfig(lexical=lx_kind)))
        for name, sc in sc_store.items():
            fused = []
            for ex, s, lx in zip(examples, sc, bm):
                ln = _normalize(lx, "max")
                fused.append({k: (1 - a.fuse_alpha) * v + a.fuse_alpha * ln.get(k, 0.0)
                              for k, v in s.items()})
            nm = f"{name}+{lx_kind}@{a.fuse_alpha}"
            out["scorers"][nm] = evaluate(nm, fused, examples, budgets, a.with_samples, **extra)
            # 融合沿用 risk 侧的 linking 成本（BM25 是 $0 的词法计算，不加 token）
            for r in budgets:
                out["scorers"][nm]["rates"][str(r)]["link_tokens"] = \
                    out["scorers"][name]["rates"][str(r)]["link_tokens"]
                out["scorers"][nm]["rates"][str(r)]["total_tokens"] = (
                    out["scorers"][name]["rates"][str(r)]["link_tokens"]
                    + out["scorers"][nm]["rates"][str(r)]["gen_tokens"])

    # ------------------------------------------------------------ 全 schema 参考系
    full_gen, full_state = [], []
    for ex in examples:
        st = make_stage_tokens(ex.schema, ex.question, ex.evidence,
                               kept_columns=None, with_samples=a.with_samples)
        full_gen.append(st.gen_prompt_tokens)
        full_state.append(n_tokens(ex.question_block()))
    out["full_schema_reference"] = {
        "gen_tokens": float(np.mean(full_gen)),
        "total_tokens": float(np.mean(full_gen)),
        "n_examples": len(examples),
    }

    # ------------------------------------------------------------ TEP（相对全 schema）
    ref_tok = float(np.mean(full_gen))
    for name, v in out["scorers"].items():
        v["tep_vs_full"] = {}
        for r in rates:
            d = v["rates"][str(r)]
            ref_perf = out["scorers"].get("bm25", {}).get("rates", {}).get(str(r), {})
            # TEP 用 SRR 作性能轴；参考点 = 全 schema（SRR 恒为 1，因为没删任何列）
            t = tep(d["srr"], d["total_tokens"], 1.0, ref_tok)
            v["tep_vs_full"][str(r)] = None if (t != t) else round(t, 4)

    # ------------------------------------------------------------ 显著性（SRR）
    if "risk_ft" in out["scorers"]:
        sig = {}
        for base in ("bm25", "embedding"):
            if base not in out["scorers"]:
                continue
            sig[base] = {}
            for r in rates:
                k = str(r)
                aa = out["scorers"][base]["rates"][k]["_srr"]
                bb = out["scorers"]["risk_ft"]["rates"][k]["_srr"]
                res = paired_bootstrap(aa, bb, n_boot=5000)
                sig[base][k] = {"delta": round(res["diff"], 4),
                                "ci": [round(res["ci_lo"], 4), round(res["ci_hi"], 4)],
                                "p": round(res["p_boot"], 5)}
        out["risk_ft_vs_baselines_srr"] = sig

    # 帕累托前沿（x=总 token，y=SRR @ 最高过滤率）
    pts, labels = [], []
    rmax = str(max(rates))
    for name, v in out["scorers"].items():
        pts.append((v["rates"][rmax]["total_tokens"], v["rates"][rmax]["srr"]))
        labels.append(name)
    front = pareto_front(pts)
    out["pareto"] = {"rate": float(rmax),
                     "front": [{"system": labels[i], "tokens": round(pts[i][0], 1),
                                "srr": round(pts[i][1], 4)} for i in front]}

    for v in out["scorers"].values():
        for d in v["rates"].values():
            for k in list(d):
                if k.startswith("_"):
                    d.pop(k)

    # ------------------------------------------------------------ 打印
    print(f"\n== BIRD-Dev 打分器对比（n={len(examples)}, mode={a.mode}, "
          f"expand={a.expand}, tokenizer={TOKENIZER}）==")
    print(f"{'scorer':<12}{'AUROC':>7} | " + " ".join(f"SRR@{r}" for r in rates))
    for nm, v in sorted(out["scorers"].items(), key=lambda kv: -kv[1]["rates"][rmax]["srr"]):
        row = " ".join(f"{v['rates'][str(r)]['srr']:>8.3f}" for r in rates)
        print(f"{nm:<12}{v['auroc']:>7.4f} | {row}")
    print(f"\n{'scorer':<12}  " + " ".join(f"C̄@{r}" for r in rates))
    for nm, v in sorted(out["scorers"].items(), key=lambda kv: -kv[1]["rates"][rmax]["srr"]):
        row = " ".join(f"{v['rates'][str(r)]['avg_kept_columns']:>8.1f}" for r in rates)
        print(f"{nm:<12}  {row}")
    print(f"\n{'scorer':<12}  " + " ".join(f"tok@{r}" for r in rates))
    for nm, v in sorted(out["scorers"].items(), key=lambda kv: -kv[1]["rates"][rmax]["srr"]):
        row = " ".join(f"{v['rates'][str(r)]['total_tokens']:>8.0f}" for r in rates)
        print(f"{nm:<12}  {row}")
    if any(v.get("gate") for v in out["scorers"].values()):
        print("\n门控表级召回（门控漏掉 gold 表的话，后面再准也救不回来）:")
        for nm, v in out["scorers"].items():
            if v.get("gate"):
                print(f"  {nm:<12} gate={v['gate']:<6} table_recall={v['gate_table_recall']}")
    print(f"\n全 schema 直喂（参考系）: {out['full_schema_reference']['gen_tokens']:.0f} tokens/query")
    if out.get("risk_ft_vs_baselines_srr"):
        print("\n== risk_ft - baseline（SRR，配对 bootstrap 5000）==")
        for base, d in out["risk_ft_vs_baselines_srr"].items():
            for r in rates:
                s = d[str(r)]
                print(f"  vs {base:<10} filter {r:.2f}: Δ={s['delta']:+.4f} "
                      f"CI=[{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}] p={s['p']:.4f}")
    print("\n== 帕累托前沿 (x=tokens, y=SRR@" + rmax + ") ==")
    for p in out["pareto"]["front"]:
        print(f"  {p['system']:<12} tokens={p['tokens']:>8.0f}  SRR={p['srr']:.4f}")

    path = a.out or os.path.join(RESULTS, f"bird_eval_n{len(examples)}_v{'-'.join(views)}"
                                          f"_L{a.max_len}_s{a.with_samples}_{a.mode}"
                                          f"_x{a.expand}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
