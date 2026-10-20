"""Matched-budget trade-off (reviewer request): same reader (Jev fused, closure f1k1), same selector, same generator,
candidate pools = fixed top-k + closure for k in KS.  The conformal certificate of the fixed family is one point on
this curve (eps=.05 -> k=10, eps=.02 -> k=16); the curve shows what the budget buys in coverage, selection recall,
EX and total tokens.
  dump  : single-stage generation prompts per k, and selector (linking) prompts per k
  parse : --outputs <selector {h,text}> -> two-stage generation prompts + EX task files per k
  stats : coverage / size / token statistics per k (and of the parsed selections)
Outputs (results/): topk_single_genprompts.jsonl, tasks_topk_single.jsonl, topk_link_prompts.jsonl,
                    topk_two_genprompts.jsonl, tasks_topk_two.jsonl, topk_pools.json
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, CACHE, DATA, View, fused_scores, lexical, load_examples, read_scores
from v2_cascade import tier1_scores
from v2_e2e import prompt_of
from jevschema.certify import covers, kept_at
from jevschema.selection import closure, parse_closure
from jevschema.tokens import n_tokens, schema_prompt_text

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_llm_link.py"), encoding="utf-8").read()
_ns = {"__file__": __file__}
exec(_src.split("\nap = argparse")[0], _ns)
TMPL, parse_cols = _ns["TMPL"], _ns["parse_cols"]

KS = [4, 6, 10, 16, 25, 40]
KS_SEL = [4, 10, 16, 40]          # cost-reduced selector curve (both certified points included)
if os.environ.get("KSEL"):         # e.g. KSEL=10,16: only the pools of Table 4
    KS_SEL = [int(k) for k in os.environ["KSEL"].split(",")]
# READER=dist: the distilled local reader (no API; block_t10 seed 7, fused exactly like Jev) instead of Jev
READER = os.environ.get("READER", "jev")
READER_FILE = {"jev": "jev_t10_dev.jsonl", "dist": "block_t10_dev_s7.jsonl"}[READER]
RSFX = "" if READER == "jev" else f"_{READER}"
if os.environ.get("FULLPOOL") == "1":  # reviewer baseline: the same selector reading the FULL schema (pool = all columns)
    KS = KS + [100000]; KS_SEL = [100000]
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dump", "dumplink", "parse", "stats", "probe", "tabsel", "cert2", "cost", "selcost", "latdump"]); ap.add_argument("--outputs", default="")
a = ap.parse_args()
dev = load_examples("dev")
bm, em = lexical("dev")
s0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
ld = read_scores(os.path.join(CACHE, READER_FILE))
s1 = tier1_scores(dev, s0, [ld[e.ex_id] for e in dev], 0.02)
V = View(dev, s1, "f1k1")
pools = {k: [kept_at(e.schema, V.orders[i], k, V.cfg) for i, e in enumerate(dev)] for k in KS}


def link_prompt(e, cols):
    txt = schema_prompt_text(e.schema, tables=sorted({t for t, _ in cols}), columns=cols)
    return TMPL.format(db=e.db_id, schema=txt, ev=e.evidence or "", q=e.question)


def h(p):
    return hashlib.md5(p.encode("utf-8")).hexdigest()


if a.mode == "dump":
    seen_g, seen_l = set(), set()
    with open(os.path.join(RES, "topk_single_genprompts.jsonl"), "w", encoding="utf-8") as fg, \
         open(os.path.join(RES, "tasks_topk_single.jsonl"), "w", encoding="utf-8") as ft, \
         open(os.path.join(RES, "topk_link_prompts.jsonl"), "w", encoding="utf-8") as fl:
        for k in KS:
            for i, (e, s) in enumerate(zip(dev, pools[k])):
                gp = prompt_of(e, s); hg = h(gp)
                ft.write(json.dumps({"sys": f"top{k}", "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
                if hg not in seen_g:
                    seen_g.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
                lp = link_prompt(e, s); hl = h(lp)
                if hl not in seen_l:
                    seen_l.add(hl); fl.write(json.dumps({"h": hl, "prompt": lp}, ensure_ascii=False) + "\n")
    print("single-stage prompts", len(seen_g), "link prompts", len(seen_l))
elif a.mode == "dumplink":
    seen = set()
    with open(os.path.join(RES, f"topk_link_prompts_sel{RSFX}.jsonl"), "w", encoding="utf-8") as fl:
        for k in KS_SEL:
            for e, s in zip(dev, pools[k]):
                lp = link_prompt(e, s); hl = h(lp)
                if hl not in seen:
                    seen.add(hl); fl.write(json.dumps({"h": hl, "prompt": lp}, ensure_ascii=False) + "\n")
    print("selector prompts for k in", KS_SEL, ":", len(seen))
elif a.mode == "parse":
    KS = KS_SEL
    got = {}
    for fn in a.outputs.split(","):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); got[r["h"]] = r["text"]
    f1 = parse_closure("f1")
    seen = set(); info = {}
    with open(os.path.join(RES, "topk_two_genprompts.jsonl"), "w", encoding="utf-8") as fg, \
         open(os.path.join(RES, "tasks_topk_two.jsonl"), "w", encoding="utf-8") as ft:
        for k in KS:
            srr, cols, miss, fb = [], [], 0, 0
            for i, (e, s) in enumerate(zip(dev, pools[k])):
                t = got.get(h(link_prompt(e, s)))
                if t is None:
                    miss += 1; t = ""
                sel = [c for c in parse_cols(t, e.schema) if c in set(s)]
                if not sel:
                    sel, fb = list(s), fb + 1
                fk = closure(e.schema, sel + [x for x in e.schema.columns() if x not in sel], len(sel), f1)
                srr.append(covers(e.gold, set(fk))); cols.append(len(fk))
                gp = prompt_of(e, fk); hg = h(gp)
                ft.write(json.dumps({"sys": f"two_top{k}", "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
                if hg not in seen:
                    seen.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
            info[k] = {"selection_srr": float(np.mean(srr)), "selection_cols": float(np.mean(cols)), "missing_outputs": miss, "fallback": fb}
            print(k, info[k])
    json.dump(info, open(os.path.join(RES, "topk_selection.json"), "w"), indent=1)
elif a.mode == "tabsel":
    # table-level second stage: the selector's output is used only to choose TABLES; every pool column of a chosen
    # table is kept (then FK closure).  Reuses the column-level selector outputs (no new selector calls).
    #   --outputs <selector {h,text}>  [--write 1: also write generation prompts/tasks as two_tab_top{k}]
    import re
    got = {}
    for fn in a.outputs.split(","):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); got[r["h"]] = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S)
    f1 = parse_closure("f1"); info = {}
    write = os.environ.get("WRITE") == "1"
    fg = open(os.path.join(RES, f"topk_tab{os.environ.get('TAG', '')}_genprompts.jsonl"), "w", encoding="utf-8") if write else None
    ft = open(os.path.join(RES, f"tasks_topk_tab{os.environ.get('TAG', '')}.jsonl"), "w", encoding="utf-8") if write else None
    seen = set()
    for k in KS_SEL:
        srr, cols, ntab = [], [], []
        for i, (e, s) in enumerate(zip(dev, pools[k])):
            t = got.get(h(link_prompt(e, s)), "")
            sel = [c for c in parse_cols(t, e.schema) if c in set(s)]
            tabs = {tt for tt, _ in sel} or {tt for tt, _ in s}
            keep = [c for c in s if c[0] in tabs]
            fk = closure(e.schema, keep + [x for x in e.schema.columns() if x not in keep], len(keep), f1)
            srr.append(covers(e.gold, set(fk))); cols.append(len(fk)); ntab.append(len(tabs))
            if write:
                gp = prompt_of(e, fk); hg = h(gp)
                ft.write(json.dumps({"sys": f"tab{os.environ.get('TAG', '')}_top{k}", "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
                if hg not in seen:
                    seen.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
        info[k] = {"srr": round(float(np.mean(srr)), 4), "cols": round(float(np.mean(cols)), 1), "tables": round(float(np.mean(ntab)), 2)}
        print(k, info[k])
    if write:
        fg.close(); ft.close(); print("gen prompts", len(seen))
elif a.mode == "cert2":
    # CERTIFIED two-stage set.  For a fixed selector output (table-level selection inside pool k), the family
    #   F_lam(x) = closure_FK( B(x) U first lam columns of pi(x) not in B(x) ),  lam = 0..n
    # is nested in lam and reaches the full schema, so Theorem 1 applies to r2(x) = min{lam : F_lam(x) covers G(x)}:
    # split-conformal lam_hat gives P(final set covers) >= 1-eps.  Cross-fit (2 folds, seed 0) for the reported sets,
    # plus R random halves for the coverage distribution.  The selector never sees calibration labels.
    #   --outputs <selector {h,text}>   env POOLK (default 16), TAG (output suffix), WRITE=1 -> generation prompts
    import re, math
    got = {}
    for fn in a.outputs.split(","):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); got[r["h"]] = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S)
    f1 = parse_closure("f1"); K = int(os.environ.get("POOLK", "16")); tag = os.environ.get("TAG", "")
    COMBO = os.environ.get("COMBO", "")
    if COMBO:
        from v2_rsl_eval_lib import rsl_union_cols
        RSLC = rsl_union_cols(dev, os.environ.get("RSLM", "qwen32b"))
    orders, r2 = [], []
    for i, (e, s) in enumerate(zip(dev, pools[K])):
        t = got.get(h(link_prompt(e, s)), "")
        sel = [c for c in parse_cols(t, e.schema) if c in set(s)]
        tabs = {tt for tt, _ in sel} or {tt for tt, _ in s}
        if COMBO:  # combine with a released/re-run RSL-SQL linking (env COMBO, RSLM = its model tag)
            rc = RSLC[i]; rt = {tt for tt, _ in rc}
            if COMBO == "rsltab":
                tabs = rt & {tt for tt, _ in s} or tabs
            elif COMBO in ("uniontab", "unioncols"):
                tabs = tabs | (rt & {tt for tt, _ in s})
        B = [c for c in s if c[0] in tabs]
        if COMBO == "unioncols":
            B += [c for c in RSLC[i] if c not in set(B)]
        if os.environ.get("FBORDER") == "sametable":   # fallback restores columns of the selected tables first
            rest = [c for c in V.orders[i] if c not in set(B)]
            o = B + [c for c in rest if c[0] in tabs] + [c for c in rest if c[0] not in tabs]
        else:
            o = B + [c for c in V.orders[i] if c not in set(B)]
        o += [c for c in e.schema.columns() if c not in set(o)]
        orders.append((o, len(B)))
        lo, hi = 0, len(o) - len(B)
        if not covers(e.gold, set(closure(e.schema, o, len(o), f1))):
            r2.append(10 ** 9); continue
        while lo < hi:                                   # nested family -> binary search for the smallest lam
            mid = (lo + hi) // 2
            if covers(e.gold, set(closure(e.schema, o, len(B) + mid, f1))):
                hi = mid
            else:
                lo = mid + 1
        r2.append(lo)
    r2 = np.asarray(r2); n = len(dev)
    if os.environ.get("DUMPR2"):  # per-question data for the three-way protocol (pool size / fallback / test splits)
        f1sz = lambda i, lam: len(closure(dev[i].schema, orders[i][0], min(len(orders[i][0]), orders[i][1] + lam), f1))
        dump = {"pool_k": K, "r2": [int(x) for x in r2],
                "size": {lam: [f1sz(i, lam) for i in range(n)] for lam in range(0, 31)},
                "pool_cover": {k: [bool(covers(e.gold, set(pools[k][i]))) for i, e in enumerate(dev)] for k in KS},
                "pool_size": {k: [len(pools[k][i]) for i in range(n)] for k in KS}}
        json.dump(dump, open(os.path.join(RES, f"cert2{tag}_r2_k{K}.json"), "w"))
        print("dumped r2", len(r2))
    def lam_hat(idx, eps):
        rr = np.sort(r2[idx]); j = math.ceil((len(rr) + 1) * (1 - eps))
        return int(rr[j - 1]) if j <= len(rr) else 10 ** 9
    size = lambda i, lam: len(closure(dev[i].schema, orders[i][0], min(len(orders[i][0]), orders[i][1] + lam), f1))
    out = {"pool_k": K, "selector_recall_lam0": float(np.mean(r2 == 0))}
    rng = np.random.default_rng(0)
    for eps in (0.02, 0.05, 0.10):
        cov, sz = [], []
        for _ in range(200):
            p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
            lh = lam_hat(cal, eps); cov.append(np.mean(r2[te] <= lh)); sz.append(np.mean([size(i, lh) for i in te]))
        perm = np.random.default_rng(0).permutation(n); A, Bf = perm[: n // 2], perm[n // 2:]
        lamA, lamB = lam_hat(Bf, eps), lam_hat(A, eps)              # cross-fit: each fold calibrated on the other
        lam_i = {int(i): lamA for i in A}; lam_i.update({int(i): lamB for i in Bf})
        cf_cov = float(np.mean([r2[i] <= lam_i[i] for i in range(n)])); cf_sz = float(np.mean([size(i, lam_i[i]) for i in range(n)]))
        out[f"eps{eps}"] = {"coverage_mean": float(np.mean(cov)), "coverage_p05": float(np.percentile(cov, 5)), "size_mean": float(np.mean(sz)),
                            "crossfit_lams": [lamA, lamB], "crossfit_coverage": cf_cov, "crossfit_size": cf_sz}
        print(f"eps={eps}: coverage {np.mean(cov):.4f} (5th pct {np.percentile(cov,5):.4f}) size {np.mean(sz):.1f} | cross-fit lam {lamA},{lamB} cov {cf_cov:.4f} size {cf_sz:.1f}")
        if os.environ.get("WRITE") == "1" and eps in (0.05, 0.10):
            fg = open(os.path.join(RES, f"cert2{tag}_e{eps}_genprompts.jsonl"), "w", encoding="utf-8")
            ft = open(os.path.join(RES, f"tasks_cert2{tag}_e{eps}.jsonl"), "w", encoding="utf-8"); seen = set()
            for i, e in enumerate(dev):
                o, nb = orders[i]; fk = closure(e.schema, o, min(len(o), nb + lam_i[i]), f1)
                gp = prompt_of(e, fk); hg = h(gp)
                ft.write(json.dumps({"sys": f"cert2{tag}_e{eps}", "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
                if hg not in seen:
                    seen.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
            fg.close(); ft.close(); print("  wrote", len(seen), "generation prompts")
    if os.environ.get("SEEDS"):  # cross-fit seed sensitivity: fold split by seed s, each fold calibrated on the other
        for eps in (0.05, 0.10):
            for s in range(int(os.environ["SEEDS"])):
                perm = np.random.default_rng(s).permutation(n); A, Bf = perm[: n // 2], perm[n // 2:]
                lamA, lamB = lam_hat(Bf, eps), lam_hat(A, eps)
                lam_i = {int(i): lamA for i in A}; lam_i.update({int(i): lamB for i in Bf})
                out.setdefault("seeds", {})[f"{eps}_{s}"] = {int(i): int(l) for i, l in lam_i.items()}
                print(f"SEED eps={eps} s={s} lam {lamA},{lamB} cov {np.mean([r2[i] <= lam_i[i] for i in range(n)]):.4f} "
                      f"size {np.mean([size(i, lam_i[i]) for i in range(n)]):.1f}")
    if os.environ.get("ALLLAM"):  # generation prompts for every question at each listed fallback depth
        for lam in map(int, os.environ["ALLLAM"].split(",")):
            fg = open(os.path.join(RES, f"cert2{tag}_lam{lam}_genprompts.jsonl"), "w", encoding="utf-8")
            ft = open(os.path.join(RES, f"tasks_cert2{tag}_lam{lam}.jsonl"), "w", encoding="utf-8"); seen = set()
            for i, e in enumerate(dev):
                o, nb = orders[i]; fk = closure(e.schema, o, min(len(o), nb + lam), f1)
                gp = prompt_of(e, fk); hg = h(gp)
                ft.write(json.dumps({"sys": f"cert2{tag}_lam{lam}", "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
                if hg not in seen:
                    seen.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
            fg.close(); ft.close(); print(f"  lam={lam}: wrote", len(seen), "prompts")
    print("selector recall at lam=0:", out["selector_recall_lam0"])
    json.dump(out, open(os.path.join(RES, f"cert2{tag}_k{K}.json"), "w"), indent=1)
elif a.mode == "latdump":
    # latency benchmark inputs for N random questions (seed 0): full-schema generator prompt, selector prompt for the
    # k=POOLK pool of the current READER, and the certified generator prompt of system tasks_<CERTSYS> (same question)
    K = int(os.environ.get("POOLK", "16")); N = int(os.environ.get("N", "200")); cs = os.environ["CERTSYS"]
    gp = {}
    for f in ("full", cs):
        for l in open(os.path.join(RES, f"{f}_genprompts.jsonl"), encoding="utf-8"):
            r = json.loads(l); gp[r["h"]] = r["prompt"]
    th = {}
    for f in ("full", cs):
        for l in open(os.path.join(RES, f"tasks_{f}.jsonl"), encoding="utf-8-sig"):
            r = json.loads(l); th[(f, r["i"])] = r["h"]
    idx = sorted(np.random.default_rng(0).choice(len(dev), N, replace=False).tolist())
    with open(os.path.join(RES, f"latency_inputs{RSFX}.jsonl"), "w", encoding="utf-8") as fo:
        for i in idx:
            e = dev[i]
            fo.write(json.dumps({"i": i, "db": e.db_id, "full": gp[th[("full", i)]], "sel": link_prompt(e, pools[K][i]),
                                 "cert": gp[th[(cs, i)]], "n_cols": len(e.schema.columns())}, ensure_ascii=False) + "\n")
    print("wrote", len(idx), "latency inputs")
elif a.mode == "selcost":
    # selector tokens (in / out) per question for the k=POOLK pool of the current READER; --outputs: selector outputs
    import re
    K = int(os.environ.get("POOLK", "16")); sel_out = {}
    for l in open(a.outputs, encoding="utf-8"):
        r = json.loads(l); sel_out[r["h"]] = r["text"]
    sin = [n_tokens(link_prompt(e, pools[K][i])) for i, e in enumerate(dev)]
    sout = [n_tokens(re.sub(r"<think>.*?</think>", "", sel_out.get(h(link_prompt(e, pools[K][i])), ""), flags=re.S)) for i, e in enumerate(dev)]
    print(f"READER={READER} k={K}: selector in {np.mean(sin):.1f} out {np.mean(sout):.1f}")
elif a.mode == "cost":
    # End-to-end token accounting per BIRD-Dev question (same tokenizer for every stage): reader, selector (in/out),
    # linker calls (RSL-SQL re-run, replayed from its cache), generator input.  --outputs: 32B selector outputs.
    import re
    sel_out = {}
    for l in open(a.outputs, encoding="utf-8"):
        r = json.loads(l); sel_out[r["h"]] = r["text"]
    gp = {}
    for f in ("cert232_e0.05", "cert2cunioncolsp16_e0.05", "rsl_q32", "rslcert_q32"):
        for l in open(os.path.join(RES, f + "_genprompts.jsonl"), encoding="utf-8"):
            r = json.loads(l); gp[r["h"]] = r["prompt"]
    def gen_tok(task):
        hs = {}
        for l in open(os.path.join(RES, f"tasks_{task}.jsonl"), encoding="utf-8"):
            r = json.loads(l); hs.setdefault(r["sys"], {})[r["i"]] = r["h"]
        return {s: float(np.mean([n_tokens(gp[x]) for x in d.values()])) for s, d in hs.items()}
    jev = {}
    for l in open(os.path.join(CACHE, "jev_t10_dev.jsonl"), encoding="utf-8"):
        r = json.loads(l); jev[r["id"]] = (r.get("in_tok") or 0) + (r.get("out_tok") or 0)
    full = [n_tokens(prompt_of(e, e.schema.columns())) for e in dev]
    cert = [n_tokens(prompt_of(e, pools[10][i])) for i, e in enumerate(dev)]
    sel_in = [n_tokens(link_prompt(e, pools[16][i])) for i, e in enumerate(dev)]
    sel_o = [n_tokens(re.sub(r"<think>.*?</think>", "", sel_out.get(h(link_prompt(e, pools[16][i])), ""), flags=re.S)) for i, e in enumerate(dev)]
    rin, rout = [], []
    _cost_log = os.environ.get("RSL_COST_LOG", os.path.join(DATA, "rslrun32", "src", "sql_log", "cost_calls.jsonl"))
    if os.path.exists(_cost_log):
        for l in open(_cost_log, encoding="utf-8"):
            r = json.loads(l); rin.append(n_tokens(r["system"] + "\n" + r["prompt"])); rout.append(n_tokens(r["answer"]))
    else:
        print(f"warning: RSL cost log not found ({_cost_log}); RSL cost columns set to zero")
    res = {"reader_jev_tokens": float(np.mean([jev.get(e.ex_id, 0) for e in dev])), "gen_full": float(np.mean(full)),
           "gen_certificate": float(np.mean(cert)), "selector_in": float(np.mean(sel_in)), "selector_out": float(np.mean(sel_o)),
           "rsl_calls_per_q": len(rin) / len(dev), "rsl_in_per_q": float(np.sum(rin)) / len(dev), "rsl_out_per_q": float(np.sum(rout)) / len(dev)}
    res.update({f"gen_{k}": v for k, v in gen_tok("cert232_e0.05").items()})
    res.update({f"gen_{k}": v for k, v in gen_tok("cert2cunioncolsp16_e0.05").items()})
    res.update({f"gen_{k}": v for k, v in gen_tok("rsl_q32").items()})
    res.update({f"gen_{k}": v for k, v in gen_tok("rslcert_q32").items()})
    for k, v in res.items():
        print(f"{k:40s} {v:9.1f}")
    json.dump(res, open(os.path.join(RES, "cost_table.json"), "w"), indent=1)
elif a.mode == "probe":
    # selection recall on the prompts present in each --outputs file only (e.g. a 300-prompt selector probe); no files written
    import re
    for fn in a.outputs.split(","):
        got = {}
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l); got[r["h"]] = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S)
        f1 = parse_closure("f1"); res = {}
        for k in KS_SEL:
            srr, cols = [], []
            for e, s in zip(dev, pools[k]):
                t = got.get(h(link_prompt(e, s)))
                if t is None:
                    continue
                sel = [c for c in parse_cols(t, e.schema) if c in set(s)] or list(s)
                fk = closure(e.schema, sel + [x for x in e.schema.columns() if x not in sel], len(sel), f1)
                srr.append(covers(e.gold, set(fk))); cols.append(len(fk))
            res[k] = (len(srr), round(float(np.mean(srr)), 4) if srr else None, round(float(np.mean(cols)), 1) if cols else None)
        print(os.path.basename(fn), res)
else:
    st = {}
    for k in KS:
        cov = float(np.mean([covers(e.gold, set(s)) for e, s in zip(dev, pools[k])]))
        cols = float(np.mean([len(s) for s in pools[k]]))
        gtok = float(np.mean([n_tokens(prompt_of(e, s)) for e, s in zip(dev, pools[k])]))
        ltok = float(np.mean([n_tokens(link_prompt(e, s)) for e, s in zip(dev, pools[k])]))
        st[k] = {"pool_coverage": cov, "pool_cols": cols, "single_stage_prompt_tok": gtok, "selector_prompt_tok": ltok}
        print(k, {x: round(y, 4) for x, y in st[k].items()})
    st["full"] = {"pool_coverage": 1.0, "pool_cols": float(np.mean([len(e.schema.columns()) for e in dev])),
                  "single_stage_prompt_tok": float(np.mean([n_tokens(prompt_of(e, None)) for e in dev]))}
    json.dump(st, open(os.path.join(RES, "topk_pools.json"), "w"), indent=1)
