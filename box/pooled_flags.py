"""Final EX statistics from the recomputed flags (ex_remote.py, fixed gold-cache semantics, 64 procs).
Per generator: EX of every system and paired diff vs full; pooled across generators: question-clustered bootstrap
(10000) + question-level sign test; separately for the confirmatory generators (GPT-5.6, DeepSeek-V4.1).
  python pooled_flags.py flags_all2_<g>.json ...
"""
import json, math, sys
import numpy as np

gens = {f.split("flags_all2_")[1][:-5]: json.load(open(f)) for f in sys.argv[1:]}
systems = ["jev_t10:t1_e0.02", "jev_t10:t1_e0.05", "jev_t10:t1_e0.1", "t0_e0.05", "laya_t10:t1_e0.05", "oracle"]
rng = np.random.default_rng(0)


def sign_p(w, l):
    n = w + l; k = max(w, l)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n) if n else 1.0


print("== per generator (EX, diff vs full, wins/losses)")
for g, F in gens.items():
    full = F["full"]; ok = [x for x in full if x is not None]
    line = f"{g:18s} full={np.mean(ok):.4f}"
    for s in systems:
        if s not in F:
            continue
        pr = [(a, b) for a, b in zip(full, F[s]) if a is not None and b is not None]
        d = np.mean([b - a for a, b in pr]); w = sum(b > a for a, b in pr); l = sum(b < a for a, b in pr)
        line += f" | {s.split(':')[-1]} {np.mean([b for _, b in pr]):.4f} ({d:+.4f}, {w}/{l})"
    print(line)


def pooled(names, s):
    rows = {}
    for g in names:
        F = gens[g]
        if s not in F:
            continue
        for i, (a, b) in enumerate(zip(F["full"], F[s])):
            if a is not None and b is not None:
                rows.setdefault(i, []).append(b - a)
    qs = sorted(rows); S = np.array([sum(rows[q]) for q in qs]); C = np.array([len(rows[q]) for q in qs])
    m = S.sum() / C.sum(); idx = rng.integers(0, len(qs), (10000, len(qs))); bs = S[idx].sum(1) / C[idx].sum(1)
    qm = S / C; w, l = int((qm > 0).sum()), int((qm < 0).sum())
    return m, np.quantile(bs, .025), np.quantile(bs, .975), 2 * min((bs <= 0).mean(), (bs >= 0).mean()), w, l, sign_p(w, l)


for title, names in (("ALL generators", list(gens)), ("CONFIRMATORY (GPT-5.6, DeepSeek-V4.1)", [g for g in gens if g in ("gpt56luna", "dsv4flash")])):
    print(f"\n== pooled, question-clustered: {title}: {names}")
    for s in systems[:-1]:
        m, lo, hi, p, w, l, ps = pooled(names, s)
        print(f"  {s:20s} diff={m:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} | question sign {w}/{l} p={ps:.4f}")
