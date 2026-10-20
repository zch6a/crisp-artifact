"""门控宽度消融：System-One 只对"BM25 前 N 列"打分，成本降多少、SRR 掉多少？

为什么这个可以**零 GPU 成本**做：
    laya 打分时 state 与候选集无关，每条 typed question 独立编码。
    所以"只对候选集 C 打分"得到的分数 == "对全库打分后把 C 之外的列 mask 掉"。
    于是直接用已经缓存好的 1534 条全量 laya 分数做 mask，就精确等价于重跑一遍门控版。
    （唯一差别是 batch 组合，不影响单条分数。）

为什么必须做这个：
    System-One 的 linking 成本是 **O(门控后候选列数)**，实测全量 1788 tokens/查询。
    而 per_db_payoff.py 算出：省下的 token ≈ 4.31/列，付得起 1788 需要 **444 列**。
    把成本减半（→894）拐点就降到 **237 列**，正好覆盖 BIRD 最大的库（199 列）。
    所以门控不是"可选优化"，它决定 System-One 能不能进入实用区间。

要回答的问题：N 取多少时，"省下的成本"最多而"SRR 损失"可忽略？
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import mean_ci, paired_bootstrap, strict_recall  # noqa: E402
from jevschema.risk import VIEWS  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import SummaryConfig, build_state  # noqa: E402
from jevschema.tokens import make_stage_tokens, n_tokens  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")
LAYA_TAG = "bird_layapareto_laya_ft_bird_vask_L1024_t6"


def _load(tag, examples):
    p = os.path.join(CACHE, f"{tag}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    miss = sum(1 for e in examples if e.ex_id not in store)
    print(f"  {tag}: 命中 {len(examples) - miss}/{len(examples)}")
    return [store.get(e.ex_id, {}) for e in examples]


def _norm(d, how="max"):
    v = np.asarray(list(d.values()), dtype=float)
    if how == "max":
        hi = v.max()
        return {k: float(x) for k, x in
                zip(d.keys(), v / hi if hi > 0 else np.zeros_like(v))}
    lo, hi = v.min(), v.max()
    return {k: float(x) for k, x in zip(d.keys(), (v - lo) / max(1e-9, hi - lo))}


def bm25_of(ex):
    from jevschema.summary import bm25_scores
    pairs = ex.schema.columns()
    docs = [f"{t} {t.replace('_',' ')} {c} {c.replace('_',' ')}" for t, c in pairs]
    q = ex.question + (" " + ex.evidence if ex.evidence else "")
    return dict(zip(pairs, bm25_scores(q, docs)))


def pick_tables_cheap(ex, bm, top_t=6):
    best = {}
    for (t, c), v in bm.items():
        if v > best.get(t, -1e9):
            best[t] = v
    return [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_t]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--gates", default="0,20,30,40,60,100,150,250")
    ap.add_argument("--cfgs", default="f1p5s15k1,f1p8k1,f1p0s15k1")
    ap.add_argument("--ks", default="8,10,12,15,18,20,25,30,35,40,50,60")
    ap.add_argument("--views", default="ask")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    gates = [int(x) for x in a.gates.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    cfgs = [parse_closure(g) for g in a.cfgs.split(",")]
    views = tuple(a.views.split(","))

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    n0 = len(ex)
    ex = [e for e in ex if e.gold is not None and e.gold.flat_pairs()]
    if n0 != len(ex):
        print(f"  剔除空 gold {n0 - len(ex)} 条")
    print(f"BIRD-Dev n={len(ex)}  gates={gates}  cfgs={a.cfgs}", flush=True)

    bm = _load("bird_bm25", ex)
    eb = _load("bird_embedding", ex)
    laya = _load(LAYA_TAG, ex)
    have_emb = any(len(d) > 0 for d in eb)
    if not any(laya):
        print("⚠️ laya 缓存缺失，退出（不要拿 bm25 顶替）")
        return

    fused = []
    for e, b, d in zip(ex, bm, eb):
        pairs = e.schema.columns()
        bn = _norm({p: float(b.get(p, 0.0)) for p in pairs}, "max")
        dn = (_norm({p: float(d.get(p, 0.0)) for p in pairs}, "minmax")
              if have_emb else {p: 0.0 for p in pairs})
        fused.append({p: 0.5 * bn[p] + 0.5 * dn[p] for p in pairs})

    # 每个 gate 的候选集 + linking token
    print("计算各 gate 的候选集与 linking token ...", flush=True)
    cand = {g: [] for g in gates}
    link = {g: [] for g in gates}
    for i, (e, b) in enumerate(zip(ex, bm)):
        pairs = e.schema.columns()
        keep_t = set(pick_tables_cheap(e, b, 6))
        # laya 缓存里有的列 = 打分时真正进过模型的列（其余视为已被门控排除）
        scored = [p for p in pairs if p in laya[i]]
        gated = [p for p in scored if p[0] in keep_t]
        order_b = sorted(gated, key=lambda p: -float(b.get(p, 0.0)))
        state, _ = build_state(e.question_block(), e.schema, SummaryConfig())
        st_tok = n_tokens(state)
        for g in gates:
            sub = order_b if g <= 0 else order_b[:g]
            cand[g].append(sub)
            qs = [VIEWS[v].format(t=t, c=c) for v in views for t, c in sub]
            link[g].append(st_tok + sum(n_tokens(q) for q in qs))
    link_mean = {g: float(np.mean(v)) for g, v in link.items()}
    print("  " + "  ".join(f"N={g}:{link_mean[g]:.0f}tok" for g in gates), flush=True)

    full_tok = float(np.mean([make_stage_tokens(e.schema, e.question, e.evidence)
                              .total_tokens for e in ex]))
    print(f"  全 schema 直给: {full_tok:.0f} tokens\n", flush=True)

    def run(name, sc_list, gate):
        rows = []
        for cfg in cfgs:
            for k in ks:
                srrs, cb, tk = [], [], []
                for i, e in enumerate(ex):
                    pairs = e.schema.columns()
                    sc = sc_list[i]
                    v = np.asarray([float(sc.get(p, -1e9)) for p in pairs])
                    order = [pairs[j] for j in np.argsort(-v, kind="stable")]
                    kept = closure(e.schema, order, k, cfg)
                    srrs.append(strict_recall(e.gold, kept))
                    cb.append(len(kept))
                    tk.append(make_stage_tokens(e.schema, e.question, e.evidence,
                                                kept_columns=kept).gen_prompt_tokens)
                rows.append({"name": name, "gate": gate, "cfg": cfg.tag(), "k": k,
                             "srr": float(np.mean(srrs)),
                             "cbar": float(np.mean(cb)),
                             "gen_tokens": float(np.mean(tk)),
                             "link_tokens": round(link_mean.get(gate, 0.0), 1),
                             "total_tokens": float(np.mean(tk)) + link_mean.get(gate, 0.0),
                             "ci": [round(x, 4) for x in mean_ci(srrs)[1:3]],
                             "_srr": srrs})
        return rows

    # 词法基线（link=0）
    all_rows = run("fused", fused, 0)
    all_rows[0]["name"] = "fused"
    for r in all_rows:
        r["link_tokens"] = 0.0
        r["total_tokens"] = r["gen_tokens"]

    for g in gates:
        masked = []
        for i, e in enumerate(ex):
            s = {p: float(laya[i].get(p, -1e9)) for p in cand[g][i]}
            masked.append(s)
        all_rows += run(f"laya_g{g}" if g > 0 else "laya_full", masked, g)

    # 每个 gate 的帕累托前沿上的最好点
    print(f"{'系统':<12}{'gate':>6}{'link tok':>10}{'最佳 SRR':>11}{'@tok':>8}"
          f"{'C̄':>7}   配置")
    summary = {}
    for name in ["fused"] + [f"laya_g{g}" if g > 0 else "laya_full" for g in gates]:
        rs = [r for r in all_rows if r["name"] == name]
        if not rs:
            continue
        best = max(rs, key=lambda r: r["srr"])
        lk = rs[0]["link_tokens"]
        print(f"{name:<12}{rs[0]['gate']:>6}{lk:>10.0f}{best['srr']:>11.4f}"
              f"{best['total_tokens']:>8.0f}{best['cbar']:>7.1f}   "
              f"{best['cfg']} k={best['k']}")
        summary[name] = {"gate": rs[0]["gate"], "link_tokens": lk,
                         "best_srr": round(best["srr"], 4),
                         "best_total_tokens": round(best["total_tokens"], 1),
                         "cfg": best["cfg"], "k": best["k"]}

    # 关键对照：达到 SRR≥0.95 的最省点，随 gate 怎么变
    print(f"\n== 达到 SRR≥0.95 的最省点：成本随 gate 怎么降 ==")
    print(f"{'系统':<12}{'link tok':>10}{'总 tok':>9}{'SRR':>8}{'省 vs 全schema':>15}"
          f"{'vs laya_full 省':>16}")
    base = None
    for name in ["fused"] + [f"laya_g{g}" if g > 0 else "laya_full" for g in gates]:
        rs = [r for r in all_rows if r["name"] == name]
        ok = [r for r in rs if r["srr"] >= 0.95]
        if not ok:
            print(f"{name:<12}{rs[0]['link_tokens']:>10.0f}{'不可达':>9}")
            continue
        r = min(ok, key=lambda z: z["total_tokens"])
        if name == "laya_full":
            base = r["total_tokens"]
        d = ""
        if base is not None and name != "laya_full" and name != "fused":
            d = f"{base - r['total_tokens']:>16.0f}"
        print(f"{name:<12}{r['link_tokens']:>10.0f}{r['total_tokens']:>9.0f}"
              f"{r['srr']:>8.4f}{full_tok - r['total_tokens']:>15.0f}{d}")

    # 显著性：每个 gate 与 laya_full 在同一 (cfg,k) 上的 SRR 差
    print(f"\n== SRR 差（配对 bootstrap vs laya_full，同 cfg 同 k）==")
    full_rows = {(r["cfg"], r["k"]): r for r in all_rows if r["name"] == "laya_full"}
    sig = {}
    for g in gates:
        if g <= 0:
            continue
        rs = [r for r in all_rows if r["name"] == f"laya_g{g}"]
        diffs = []
        for r in rs:
            f = full_rows.get((r["cfg"], r["k"]))
            if f is None:
                continue
            pb = paired_bootstrap(f["_srr"], r["_srr"], n_boot=2000)
            diffs.append((r["cfg"], r["k"], pb["diff"], pb["p_boot"]))
        if not diffs:
            continue
        worst = min(diffs, key=lambda z: z[2])
        nsig = sum(1 for z in diffs if z[3] < 0.05 and z[2] < 0)
        sig[f"g{g}"] = {"n_compare": len(diffs),
                        "worst_diff": round(worst[2], 4),
                        "worst_p": round(worst[3], 4),
                        "n_significantly_worse": nsig,
                        "mean_diff": round(float(np.mean([z[2] for z in diffs])), 4)}
        print(f"  gate={g:<4} 比较 {len(diffs):>3} 个点  平均 ΔSRR={sig[f'g{g}']['mean_diff']:+.4f}  "
              f"最差点 Δ={worst[2]:+.4f} (p={worst[3]:.4f})  "
              f"显著变差的点 {nsig}/{len(diffs)}")

    path = a.out or os.path.join(RESULTS, "gate_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(ex), "full_schema_tokens": round(full_tok, 1),
                   "link_tokens_by_gate": {str(g): round(link_mean[g], 1) for g in gates},
                   "summary": summary, "significance": sig,
                   "rows": [{kk: vv for kk, vv in r.items() if kk != "_srr"}
                            for r in all_rows]}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
