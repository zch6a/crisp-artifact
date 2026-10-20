"""One omnibus test per family: pool several configurations of the same claim over questions (question-clustered
bootstrap, 10k; question-level sign test), e.g. 'two-stage beats the official full schema in OmniSQL's official
setting' over {greedy, SC} x {linkers}.  python pooled_configs.py <name> "<flags.json>|<full_key>|<sys_key>" ..."""
import json, math, sys
import numpy as np
rows = {}
for arg in sys.argv[2:]:
    fn, fk, sk = arg.split("|")
    F = json.load(open(fn))
    for i in range(len(F[fk])):
        if F[fk][i] is not None and i < len(F[sk]) and F[sk][i] is not None:
            rows.setdefault(i, []).append(F[sk][i] - F[fk][i])
qs = sorted(rows); S = np.array([sum(rows[q]) for q in qs]); C = np.array([len(rows[q]) for q in qs])
rng = np.random.default_rng(0); idx = rng.integers(0, len(qs), (10000, len(qs))); bs = S[idx].sum(1) / C[idx].sum(1)
qm = S / C; w, l = int((qm > 0).sum()), int((qm < 0).sum()); n = w + l
ps = min(1.0, 2 * sum(math.comb(n, j) for j in range(max(w, l), n + 1)) / 2 ** n)
print(f"{sys.argv[1]}: {len(sys.argv)-2} configs, diff={S.sum()/C.sum():+.4f} CI[{np.quantile(bs,.025):+.4f},{np.quantile(bs,.975):+.4f}] "
      f"p={2*min((bs<=0).mean(),(bs>=0).mean()):.4f} | question sign {w}/{l} p={ps:.2e}")
