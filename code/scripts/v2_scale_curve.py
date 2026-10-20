"""Gain vs schema size on BIRD-Dev: questions binned by the number of columns of their database; per bin a
question-clustered pooled bootstrap over generators (10k) of (system - full).  Flags from box/ (GPU-box ex_remote).
  -> results/scale_curve.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, BOX, load_examples

B = os.path.join(BOX, "")
dev = load_examples("dev")
ncol = np.array([len(e.schema.columns()) for e in dev])
CONF = {
 "CRISP eps=.05 (5 gens)": [(f"flags_all2_{g}.json", "full", "jev_t10:t1_e0.05") for g in
                            ("dsv4flash", "gpt56luna", "llama1530", "qwen2.5-32b-awq", "qwen2.5-coder_7b")],
 "two-stage, uniform DS linker (5 gens)": [("flags_r1.json", "tpl:full", "tpl:hybrid_ds"), ("flags_ds_hyb.json", "full", "hybrid_ds"),
                            ("flags_q7b_hyb.json", "full", "hybrid_ds"), ("flags_gpt_hyb2.json", "full", "hybrid_ds"),
                            ("flags_l8b_hyb.json", "full", "hybrid_ds")],
 "OmniSQL official input, CRISP alone": [("flags_offgreedy_all_set.json", "off:full", "off:jev_t10:t1_e0.05")],
 "OmniSQL official input, value-aware two-stage (2 linkers)": [("flags_offgreedy_all_set.json", "off:full", "off:hybridv_ds"),
                            ("flags_offgreedy_all_set.json", "off:full", "off:hybridv_g6l")],
}
BINS = [(0, 30), (31, 60), (61, 100), (101, 10 ** 6)]
out = {"bins": [f"{a}-{b if b < 10**6 else 'max'}" for a, b in BINS],
       "n_questions": [int(((ncol >= a) & (ncol <= b)).sum()) for a, b in BINS],
       "n_dbs": [len({e.db_id for e, c in zip(dev, ncol) if a <= c <= b}) for a, b in BINS]}
print("bins", out["bins"], "questions", out["n_questions"], "dbs", out["n_dbs"])
rng = np.random.default_rng(0)
for name, E in CONF.items():
    Fs = [(json.load(open(B + f)), fk, sk) for f, fk, sk in E]
    res = []
    for a, b in BINS:
        rows = {}
        for F, fk, sk in Fs:
            for i in range(len(dev)):
                if a <= ncol[i] <= b and F[fk][i] is not None and F[sk][i] is not None:
                    rows.setdefault(i, []).append(F[sk][i] - F[fk][i])
        qs = sorted(rows); S = np.array([sum(rows[q]) for q in qs]); C = np.array([len(rows[q]) for q in qs])
        idx = rng.integers(0, len(qs), (10000, len(qs))); bs = S[idx].sum(1) / C[idx].sum(1)
        d = S.sum() / C.sum(); p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        res.append({"diff": float(d), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "p": float(p), "q": len(qs)})
    out[name] = res
    print(f"{name:58s} " + " | ".join(f"{r['diff']*100:+.2f} (p={r['p']:.3f})" for r in res))
json.dump(out, open(os.path.join(RES, "scale_curve.json"), "w"), indent=1)
