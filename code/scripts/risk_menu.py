"""风险感知预算分配 v3：菜单式（闭包配置 × 深度）联合选择 + 计数型风险模型。

v2 的两个实证毛病，这里逐个修：

（1）**逐列独立性在大库上系统性高估**。
    R = 1 - ∏(1 - p_i) 里，一个 200 列的库即使每列只有 p=0.01，
    R 也会累到 1-(0.99)^200 = 0.87。而实测失败率只有 0.026 —— 差 3.4 倍。
    根因不是"独立性假设错了"那么简单，而是 **LR 的长尾概率本身过于分散**：
    负例占 90%，它把小概率摊给了太多列，Σp_i 远大于真实的 gold 列数期望。
    修法：保留"独立性"的函数形式，但给 Σp_i 乘一个**在训练集上拟合的缩放置信度 λ**：
        R_count(λ) = 1 - exp(-λ · Σ_{i∈补集} p_i)
    λ 直接由"预测失败率 = 实测失败率"拟合出来，一个标量，随 N 自动伸缩，
    比全局单调映射（PAVA on R）更省参数、也更容易在论文里说清楚。

（2）**预算只调深度 m，没用上闭包旋钮**。
    前面的帕累托实验已经证明：同样的 C̄ 下，换闭包配置能白拿好几个点 SRR
    （例如 C̄≈32 时 f1p0s15k1 k=12 是 0.8703，f1p5s15k1 k=12 是 0.8924）。
    所以"预算"不该只是一维的 m，而是 **(闭包配置, 深度) 的二维菜单**：
        逐题挑「满足 R ≤ ε 且 token 最省」的那一格。
    这才是题目里 budget-aware 的字面意思：预算是**在多个便宜手段之间分配**，
    不是单纯决定截断点。

两个风险模型都保留，做消融：
    indep   R = 1 - ∏(1 - p_i)                      （v2 原版）
    count   R = 1 - exp(-λ · Σ p_i)                 （本脚本新增，论文主推）

λ 与可选的分层校准都在 **BIRD 训练集**上拟合，dev 只做应用，绝不参与拟合。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from jevschema.bird_data import load_bird_dev, load_bird_schemas  # noqa: E402
from jevschema.metrics import mean_ci, paired_bootstrap  # noqa: E402
from jevschema.selection import closure, parse_closure  # noqa: E402
from jevschema.summary import bm25_scores  # noqa: E402
from jevschema.tokens import make_stage_tokens  # noqa: E402

from train_lex_classifier import features  # noqa: E402

RESULTS = os.path.join(_ROOT, "results")
CACHE = os.path.join(RESULTS, "bird_cache")
MODEL = os.path.join(RESULTS, "lex_classifier_bird.json")


# ---------------------------------------------------------------- 缓存
def _load(kind, examples, fn):
    p = os.path.join(CACHE, f"bird_{kind}.jsonl")
    store = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            store[r["id"]] = {tuple(k.split("|")): v for k, v in r["s"].items()}
    if fn is None:      # 只读模式：缓存缺的留空，绝不用别的信号顶替
        return [store.get(e.ex_id, {}) for e in examples]
    todo = [e for e in examples if e.ex_id not in store]
    if todo:
        os.makedirs(CACHE, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for e in todo:
                s = fn(e)
                store[e.ex_id] = s
                f.write(json.dumps({"id": e.ex_id,
                                    "s": {"|".join(k): v for k, v in s.items()}},
                                   ensure_ascii=False) + "\n")
    return [store[e.ex_id] for e in examples]


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


# ---------------------------------------------------------------- 风险模型
def risk_indep(p, kept_idx):
    """1 - ∏_{i ∉ kept}(1 - p_i)，log 空间防下溢。"""
    mask = np.ones(len(p), dtype=bool)
    mask[kept_idx] = False
    q = np.clip(1.0 - p[mask], 1e-12, 1.0)
    return float(-np.expm1(np.sum(np.log(q))))


def risk_count(p, kept_idx, lam):
    """1 - exp(-λ · Σ_{i ∉ kept} p_i)。λ 是缩放置信度，训练集上拟合。"""
    mask = np.ones(len(p), dtype=bool)
    mask[kept_idx] = False
    s = float(np.sum(p[mask]))
    return float(-np.expm1(-lam * s))


def fit_lambda(xs, ys):
    """拟合 λ：min Σ NLL，q = 1 - exp(-λ x)。xs=Σp_i（补集），ys=是否真漏。"""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    def nll(lam):
        q = np.clip(-np.expm1(-lam * xs), 1e-9, 1 - 1e-9)
        return float(-np.sum(ys * np.log(q) + (1 - ys) * np.log(1 - q)))

    lo, hi = 1e-3, 20.0
    best, bx = nll(1.0), 1.0
    # 粗扫 + 局部三分，避免依赖 scipy
    for lam in np.geomspace(lo, hi, 200):
        v = nll(lam)
        if v < best:
            best, bx = v, float(lam)
    lo2, hi2 = bx / 1.5, bx * 1.5
    for _ in range(60):
        m1 = lo2 + (hi2 - lo2) / 3
        m2 = hi2 - (hi2 - lo2) / 3
        if nll(m1) < nll(m2):
            hi2 = m2
        else:
            lo2 = m1
    lam = (lo2 + hi2) / 2
    if nll(lam) > best:
        lam = bx
    return float(lam)


# ---------------------------------------------------------------- 打分与排序
def prep_examples(ex, w, bias, mu, sd, eb=None):
    """产出每题的 (pairs, p, order)：p=校准概率，order=融合分降序下标。"""
    bm = _load("bm25", ex, bm25_of)
    if eb is None:
        eb = _load("embedding", ex, None)   # 只读：缺缓存就是缺，不许用 bm25 顶替
    have_emb = any(len(d) > 0 for d in eb)
    out = []
    for e, b, d in zip(ex, bm, eb):
        pairs, X, _gold = features(e, None)
        p = 1.0 / (1.0 + np.exp(-np.clip((X - mu) / sd @ w + bias, -30, 30)))
        bn = _norm({q: float(b.get(q, 0.0)) for q in pairs}, "max")
        dn = (_norm({q: float(d.get(q, 0.0)) for q in pairs}, "minmax")
              if have_emb else {q: 0.0 for q in pairs})
        s = np.asarray([0.5 * bn[q] + 0.5 * dn[q] for q in pairs])
        out.append({"e": e, "pairs": pairs, "p": p, "s": s,
                    "order": np.argsort(-s, kind="stable")})
    return out


def build_candidates(rows, cfgs, ms, lam):
    """逐题把菜单每一格都算出来，一次建表，后面所有 chooser 共用。

    不改这个写法的话，每换一个 ε / 一个预算 B 就要重跑一遍闭包和 token 计数，
    1534 × 28 格 × 十几个设置 = 几十万次，跑不完也浪费。

    每格记录：
        0 cfg_idx  1 m  2 R_count  3 R_indep  4 R_mass  5 tokens  6 C  7 是否全召回
    """
    table = []
    for r in rows:
        e, pairs, order, p, s = r["e"], r["pairs"], r["order"], r["p"], r["s"]
        gold = set(e.gold.flat_pairs())
        tot = float(np.sum(s)) or 1.0
        cand = []
        for ci, cfg in enumerate(cfgs):
            for m in ms:
                if m > len(pairs):
                    continue
                kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
                kset = set(kept)
                kidx = np.asarray([j for j, q in enumerate(pairs) if q in kset],
                                  dtype=int)
                mass = float(np.sum(s[kidx])) / tot      # 留下的相关度质量占比
                T = make_stage_tokens(e.schema, e.question, e.evidence,
                                      kept_columns=kept).gen_prompt_tokens
                cand.append((ci, m,
                             risk_count(p, kidx, lam),
                             risk_indep(p, kidx),
                             1.0 - mass,               # 漏掉的质量 = 代理风险
                             float(T), len(kept),
                             1.0 if all(g in kset for g in gold) else 0.0))
        table.append(cand)
    return table


_RISK_COL = {"count": 2, "indep": 3, "mass": 4}


def apply_chooser(table, cfgs, chooser, risk_col):
    srrs, cb, tk, pick = [], [], [], []
    for cand in table:
        j = chooser(cand, risk_col)
        srrs.append(cand[j][7]); cb.append(cand[j][6]); tk.append(cand[j][5])
        pick.append((cfgs[cand[j][0]].tag(), cand[j][1]))
    return {"srr": float(np.mean(srrs)), "cbar": float(np.mean(cb)),
            "tok": float(np.mean(tk)), "_srr": srrs, "_pick": pick}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--eps", default="0.20,0.10,0.05,0.03,0.02,0.01")
    ap.add_argument("--cfgs", default="f1s0,f1p5s15k1,f1p8s20k1,f1p12s25k1")
    ap.add_argument("--ms", default="8,10,12,15,18,22,25,30,35,40,50,60")
    ap.add_argument("--model", default="count", choices=["count", "indep"])
    ap.add_argument("--mode", default="both", choices=["risk", "budget", "both"])
    ap.add_argument("--budgets", default="380,420,460,500,540")
    ap.add_argument("--fixed-grid", default="f1p5s15k1")
    ap.add_argument("--fixed-ks", default="10,15,20,25,30,35,40,50,60")
    ap.add_argument("--calib-train", type=int, default=2000)
    ap.add_argument("--clf", default=MODEL)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    eps_list = [float(x) for x in a.eps.split(",")]
    cfgs = [parse_closure(g) for g in a.cfgs.split(",")]
    ms = [int(x) for x in a.ms.split(",")]

    with open(a.clf, "r", encoding="utf-8") as f:
        M = json.load(f)
    w = np.asarray(M["w"]); bias = float(M["bias"])
    mu = np.asarray(M["mu"]); sd = np.asarray(M["sd"])
    sd = np.where(sd < 1e-6, 1.0, sd)     # 保存时 round 会把常量特征的 sd 抹成 0
    print(f"打分器 {os.path.basename(a.clf)}  (dev AUROC {M['rec']['auroc']}, "
          f"ECE {M['rec']['ece']})   风险模型 = {a.model}")
    print(f"菜单 {len(cfgs)} 配置 × {len(ms)} 深度 = {len(cfgs)*len(ms)} 格")

    schemas = load_bird_schemas()
    ex = load_bird_dev(limit=a.n if a.n > 0 else None, schemas=schemas, shuffle=False)
    print(f"BIRD-Dev n={len(ex)}")
    rows = prep_examples(ex, w, bias, mu, sd)

    # ---------------------------------------------------------- λ 拟合（训练集）
    lam = 1.0
    if a.model == "count" and a.calib_train > 0:
        from jevschema.bird_train import load_bird_train
        tr, _ = load_bird_train(limit=a.calib_train)
        trp = prep_examples(tr, w, bias, mu, sd)
        xs, ys = [], []
        for r in trp:
            e, pairs, order, p = r["e"], r["pairs"], r["order"], r["p"]
            gold = set(e.gold.flat_pairs())
            for cfg in cfgs:
                for m in ms:
                    if m > len(pairs):
                        continue
                    kept = closure(e.schema, [pairs[j] for j in order], m, cfg)
                    kidx = np.asarray([j for j, q in enumerate(pairs)
                                       if q in set(kept)], dtype=int)
                    mask = np.ones(len(p), dtype=bool); mask[kidx] = False
                    xs.append(float(np.sum(p[mask])))
                    ys.append(1.0 if any(g not in set(kept) for g in gold) else 0.0)
        lam = fit_lambda(xs, ys)
        xs = np.asarray(xs); ys = np.asarray(ys)
        print(f"\nλ 拟合（BIRD 训练 {len(tr)} 例，{len(xs)} 个采样点，dev 不参与）:")
        print(f"  λ = {lam:.4f}   （λ<1 说明 LR 长尾概率确实过于分散）")
        for lo, hi in [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 1e9)]:
            sel = (xs >= lo) & (xs < hi)
            if sel.sum():
                pred = float(np.mean(-np.expm1(-lam * xs[sel])))
                print(f"  Σp∈[{lo:.1f},{hi:.1f})  n={int(sel.sum()):<7} "
                      f"预测={pred:.4f}  实测={ys[sel].mean():.4f}  "
                      f"偏差={pred-ys[sel].mean():+.4f}")

    # ---------------------------------------------------------- 固定 (cfg,k) 对照
    print(f"\n== 固定 (配置, k) 对照 ==")
    print(f"{'cfg':<12}{'k':>4}{'SRR':>8}{'C̄':>8}{'tokens':>9}{'失败率':>9}")
    fcfg = [parse_closure(g) for g in a.fixed_grid.split(",")]
    fks = [int(x) for x in a.fixed_ks.split(",")]
    fixed = {}
    for cfg in fcfg:
        for k in fks:
            srrs, cb, tk = [], [], []
            for r in rows:
                e, pairs, order = r["e"], r["pairs"], r["order"]
                kept = closure(e.schema, [pairs[j] for j in order], k, cfg)
                srrs.append(1.0 if all(g in set(kept) for g in e.gold.flat_pairs()) else 0.0)
                cb.append(len(kept))
                tk.append(make_stage_tokens(e.schema, e.question, e.evidence,
                                            kept_columns=kept).gen_prompt_tokens)
            fixed[(cfg.tag(), k)] = {"srr": float(np.mean(srrs)),
                                     "cbar": float(np.mean(cb)),
                                     "tok": float(np.mean(tk)), "_srr": srrs}
            ci = mean_ci(srrs)
            print(f"{cfg.tag():<12}{k:>4}{np.mean(srrs):>8.4f}{np.mean(cb):>8.1f}"
                  f"{np.mean(tk):>9.0f}{1-np.mean(srrs):>9.4f}"
                  f"   CI=[{ci[1]:.4f},{ci[2]:.4f}]")

    # ---------------------------------------------------------- 候选表（只算一次）
    print("\n建候选表（每格 = 配置 × 深度，含风险/花销/是否全召回）...")
    table = build_candidates(rows, cfgs, ms, lam)

    def chooser_eps(eps):
        """风险约束：花最少的钱把风险压到 ε 以下（难题会失控，故另设预算形式）。"""
        def f(cand, rc):
            feas = [j for j, c in enumerate(cand) if c[rc] <= eps]
            if not feas:
                return min(range(len(cand)), key=lambda j: (cand[j][rc], cand[j][5]))
            return min(feas, key=lambda j: cand[j][5])
        return f

    def chooser_budget(B):
        """预算约束：在预算内把风险压到最低。

        为什么还要这个形式：风险约束遇到"风险压不下去"的难题会把预算顶到上限
        （实测 ε=0.05 时 98% 的题都选最贵配置，花 555 token 仍失败 26.7%）。
        预算约束天然封顶，难题占用的预算不会失控。
        """
        def f(cand, rc):
            feas = [j for j, c in enumerate(cand) if c[5] <= B]
            if not feas:
                return min(range(len(cand)), key=lambda j: cand[j][5])
            return min(feas, key=lambda j: (cand[j][rc], cand[j][5]))
        return f

    def chooser_oracle(cand, rc):
        """上界：偷看 gold，挑「能全召回且最省」的那一格。

        这个数字的作用是**止损**：如果连 oracle 自适应都赢不了固定 (cfg,k)，
        说明逐题自适应在这套菜单上根本没有空间，再打磨预算机制就是浪费时间，
        应该把结论写成负面结果。
        """
        feas = [j for j, c in enumerate(cand) if c[7] >= 1.0]
        if not feas:
            return min(range(len(cand)), key=lambda j: -cand[j][6])   # 全都不行 -> 取最大覆盖
        return min(feas, key=lambda j: cand[j][5])

    out = {"n": len(ex), "model": a.model, "lam": lam,
           "cfgs": [c.tag() for c in cfgs], "ms": ms,
           "fixed": {f"{c}_k{k}": {kk: vv for kk, vv in d.items() if kk != "_srr"}
                     for (c, k), d in fixed.items()},
           "menu_risk": {}, "menu_budget": {}, "menu_oracle": {}}
    menu_srr: dict = {}

    def report(tag, key, chooser, d, store):
        ci = mean_ci(d["_srr"])
        ms_used = [p[1] for p in d["_pick"]]
        frac_top = float(np.mean([1.0 if p[0] == cfgs[-1].tag() else 0.0
                                  for p in d["_pick"]]))
        frac_bot = float(np.mean([1.0 if p[0] == cfgs[0].tag() else 0.0
                                  for p in d["_pick"]]))
        rec = {"srr": round(d["srr"], 4), "cbar": round(d["cbar"], 2),
               "tok": round(d["tok"], 1), "fail": round(1 - d["srr"], 4),
               "m_med": float(np.median(ms_used)),
               "frac_top_cfg": round(frac_top, 4),
               "frac_bot_cfg": round(frac_bot, 4),
               "ci": [round(ci[1], 4), round(ci[2], 4)]}
        store[str(key)] = rec
        menu_srr[(tag, str(key))] = d["_srr"]
        print(f"{key:<7}{d['srr']:>8.4f}{d['cbar']:>8.1f}{d['tok']:>9.0f}"
              f"{1-d['srr']:>10.4f}{np.median(ms_used):>7.0f}"
              f"{frac_top:>10.3f}{frac_bot:>10.3f}"
              f"   CI=[{ci[1]:.4f},{ci[2]:.4f}]")

    HDR = (f"{'':<7}{'SRR':>8}{'C̄':>8}{'tokens':>9}{'实测失败':>10}"
           f"{'m中位':>7}{'最贵占比':>10}{'最省占比':>10}")
    rc = _RISK_COL[a.model]

    if a.mode in ("risk", "both"):
        print("\n== 形式 A：风险约束（min tokens s.t. R ≤ ε）==")
        print(HDR)
        for eps in eps_list:
            report("risk", eps, chooser_eps(eps),
                   apply_chooser(table, cfgs, chooser_eps(eps), rc),
                   out["menu_risk"])

    budgets = [float(x) for x in a.budgets.split(",")]
    if a.mode in ("budget", "both"):
        print("\n== 形式 B：预算约束（min R s.t. tokens ≤ B）==")
        print(HDR)
        for B in budgets:
            report("budget", B, chooser_budget(B),
                   apply_chooser(table, cfgs, chooser_budget(B), rc),
                   out["menu_budget"])

    print("\n== 上界：oracle 自适应（偷看 gold，取能全召回的最省格）==")
    print(HDR)
    d_or = apply_chooser(table, cfgs, chooser_oracle, rc)
    report("oracle", "-", chooser_oracle, d_or, out["menu_oracle"])
    best_fixed = max(fixed.items(), key=lambda kv: kv[1]["srr"])
    print(f"  oracle: SRR={d_or['srr']:.4f} C̄={d_or['cbar']:.1f} tok={d_or['tok']:.0f}")
    print(f"  固定最优: {best_fixed[0][0]} k={best_fixed[0][1]} "
          f"SRR={best_fixed[1]['srr']:.4f} C̄={best_fixed[1]['cbar']:.1f} "
          f"tok={best_fixed[1]['tok']:.0f}")
    # 达到同样 SRR 时，固定策略要花多少？—— 这是自适应真正的可挖空间
    tgt = d_or["srr"]
    ok = [kv for kv in fixed.items() if kv[1]["srr"] >= tgt - 1e-9]
    if ok:
        kv = min(ok, key=lambda z: z[1]["tok"])
        print(f"  固定策略达到 SRR≥{tgt:.4f} 的最省花销: {kv[1]['tok']:.0f} tokens "
              f"(C̄={kv[1]['cbar']:.1f})  → oracle 自适应省 "
              f"{kv[1]['tok']-d_or['tok']:.0f} tokens "
              f"({100*(kv[1]['tok']-d_or['tok'])/kv[1]['tok']:.1f}%)")
    else:
        print(f"  固定策略全网格达不到 SRR={tgt:.4f}（最高 "
              f"{best_fixed[1]['srr']:.4f}）→ 自适应有额外收益")

    def compare(store, tag, keys):
        print(f"\n== 等 C̄ 对照（{tag} vs 最接近的固定 (cfg,k)）==")
        for key in keys:
            r = store[str(key)]
            kk = min(fixed, key=lambda z: abs(fixed[z]["cbar"] - r["cbar"]))
            d = fixed[kk]
            pb = paired_bootstrap(d["_srr"], menu_srr[(tag, str(key))])
            print(f"  {tag}={key:<7} C̄={r['cbar']:5.1f} SRR={r['srr']:.4f}   "
                  f"vs {kk[0]} k={kk[1]:<3} C̄={d['cbar']:5.1f} SRR={d['srr']:.4f}   "
                  f"Δ={pb['diff']:+.4f}  p={pb['p_boot']:.4f} "
                  f"CI=[{pb['ci_lo']:+.4f},{pb['ci_hi']:+.4f}]")
        print(f"\n== 等 token 对照（同样花钱谁分高）==")
        for key in keys:
            r = store[str(key)]
            kk = min(fixed, key=lambda z: abs(fixed[z]["tok"] - r["tok"]))
            d = fixed[kk]
            pb = paired_bootstrap(d["_srr"], menu_srr[(tag, str(key))])
            print(f"  {tag}={key:<7} tok={r['tok']:6.1f} SRR={r['srr']:.4f}   "
                  f"vs {kk[0]} k={kk[1]:<3} tok={d['tok']:6.1f} SRR={d['srr']:.4f}   "
                  f"Δ={pb['diff']:+.4f}  p={pb['p_boot']:.4f}")

    if a.mode in ("risk", "both"):
        compare(out["menu_risk"], "risk", eps_list)
    if a.mode in ("budget", "both"):
        compare(out["menu_budget"], "budget", budgets)

    path = a.out or os.path.join(RESULTS, "risk_menu_bird.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {path}")


if __name__ == "__main__":
    main()
