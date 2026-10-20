"""Pooled significance of the ZERO-JEV two-stage configuration (distilled-reader certificate + DeepSeek selection) vs the full
schema, across generators, question-clustered bootstrap (10k) + question-level sign test.
Each entry: (generator, flags file, full-system key, hybrid key)."""
import json, math, os
import numpy as np

E = [("OmniSQL-7B", "flags_omni_q16.json", "tpl:full", "tpl:hybrid_distds"),
     ("DeepSeek-V4.1", "flags_ds_distds.json", "full", "hybrid_distds"),
     ("Qwen2.5-Coder-7B", "flags_q7b_distds.json", "full", "hybrid_distds"),
     ("Llama-3.1-8B", "flags_l8b_distds.json", "full", "hybrid_distds")]
rows = {}
for g, fn, fk, hk in E:
    if not os.path.exists(fn):
        print("missing", g); continue
    F = json.load(open(fn))
    d = [(i, F[hk][i] - F[fk][i]) for i in range(len(F[fk])) if F[fk][i] is not None and i < len(F[hk]) and F[hk][i] is not None]
    print(f"{g:22s} n={len(d)} diff={np.mean([x for _, x in d]):+.4f}")
    for i, x in d:
        rows.setdefault(i, []).append(x)
qs = sorted(rows); S = np.array([sum(rows[q]) for q in qs]); C = np.array([len(rows[q]) for q in qs])
rng = np.random.default_rng(0); idx = rng.integers(0, len(qs), (10000, len(qs))); bs = S[idx].sum(1) / C[idx].sum(1)
qm = S / C; w, l = int((qm > 0).sum()), int((qm < 0).sum()); n = w + l
p_sign = min(1.0, 2 * sum(math.comb(n, j) for j in range(max(w, l), n + 1)) / 2 ** n)
print(f"POOLED zero-Jev two-stage vs full: diff={S.sum()/C.sum():+.4f} 95%CI[{np.quantile(bs,.025):+.4f},{np.quantile(bs,.975):+.4f}] "
      f"p={2*min((bs<=0).mean(),(bs>=0).mean()):.4f} | question sign {w}/{l} p={p_sign:.2e}")
