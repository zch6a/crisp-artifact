"""Difficulty strata (BIRD simple/moderate/challenging) for pooled configurations vs full schema, question-clustered
bootstrap (10k) + sign test.  python strata_pooled.py <config>  where config in {twostage, zerojev, crisp}."""
import json, math, os, sys
import numpy as np
D = json.load(open("dev_difficulty.json"))
C = {"twostage": [("OmniSQL-7B", "flags_omni_q9.json", "tpl:full", "tpl:hybrid_ds"),
                  ("DeepSeek-V4.1", "flags_ds_hyb.json", "full", "hybrid_ds"),
                  ("Qwen2.5-Coder-7B", "flags_q7b_hyb.json", "full", "hybrid_ds"),
                  ("GPT-5.6", "flags_gpt_hyb.json", "full", "hybrid_gpt56"),
                  ("Llama-3.1-8B", "flags_l8b_hyb.json", "full", "hybrid_ds")],
     "zerojev": [("OmniSQL-7B", "flags_omni_q16.json", "tpl:full", "tpl:hybrid_distds"),
                 ("DeepSeek-V4.1", "flags_ds_distds.json", "full", "hybrid_distds"),
                 ("Qwen2.5-Coder-7B", "flags_q7b_distds.json", "full", "hybrid_distds"),
                 ("Llama-3.1-8B", "flags_l8b_distds.json", "full", "hybrid_distds")],
     "sc": [("OmniSQL-7B SC", "flags_q13_sc.json", "tpl:full", "tpl:hybrid_ds")]}
E = C[sys.argv[1]]
for stratum in ("simple", "moderate", "challenging", "all"):
    rows = {}
    for g, fn, fk, hk in E:
        if not os.path.exists(fn):
            continue
        F = json.load(open(fn))
        if fk not in F or hk not in F:
            print("  keys missing", g, fn, [k for k in F][:8]); continue
        for i in range(min(len(F[fk]), len(F[hk]))):
            if F[fk][i] is None or F[hk][i] is None or (stratum != "all" and D[i] != stratum):
                continue
            rows.setdefault(i, []).append(F[hk][i] - F[fk][i])
    qs = sorted(rows); S = np.array([sum(rows[q]) for q in qs]); N = np.array([len(rows[q]) for q in qs])
    rng = np.random.default_rng(0); idx = rng.integers(0, len(qs), (10000, len(qs))); bs = S[idx].sum(1) / N[idx].sum(1)
    qm = S / N; w, l = int((qm > 0).sum()), int((qm < 0).sum()); n = w + l
    ps = min(1.0, 2 * sum(math.comb(n, j) for j in range(max(w, l), n + 1)) / 2 ** n) if n else 1.0
    print(f"{sys.argv[1]:9s} {stratum:11s} q={len(qs):4d} diff={S.sum()/N.sum():+.4f} CI[{np.quantile(bs,.025):+.4f},"
          f"{np.quantile(bs,.975):+.4f}] p={2*min((bs<=0).mean(),(bs>=0).mean()):.4f} sign {w}/{l} p={ps:.3g}")
