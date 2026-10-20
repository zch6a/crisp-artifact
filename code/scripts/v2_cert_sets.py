"""Certify any per-question selected schema with Theorem 4 (calibrated fallback to CRISP's reader ranking).
  python v2_cert_sets.py <sets.json> <tag>      sets.json: {"ids": [...], "sets": [[[table, col], ...], ...]} (dev order)
B(x) = the given set; o(x) = B(x) followed by the remaining columns in reader order; F_lam = FK-closure(o[:|B|+lam]);
r2 = min lam covering G(x); split-conformal lam_hat with 2-fold cross-fitting (seed 0) for the reported sets, plus 200
random halves for the coverage distribution.  -> results/cert_<tag>_genprompts.jsonl, tasks_cert_<tag>.jsonl"""
import hashlib, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, fused_scores, lexical, load_examples, read_scores
from v2_cascade import tier1_scores
from v2_e2e import prompt_of
from jevschema.certify import covers
from jevschema.selection import closure, parse_closure

src, tag = sys.argv[1], sys.argv[2]
dev = load_examples("dev")
bm, em = lexical("dev")
s0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
ld = read_scores(os.path.join(CACHE, "jev_t10_dev.jsonl"))
V = View(dev, tier1_scores(dev, s0, [ld[e.ex_id] for e in dev], 0.02), "f1k1")
S = json.load(open(src))
assert S["ids"] == [e.ex_id for e in dev]
f1 = parse_closure("f1"); n = len(dev)
orders, r2 = [], []
for i, e in enumerate(dev):
    B0 = [tuple(c) for c in S["sets"][i]]
    o = B0 + [c for c in V.orders[i] if c not in set(B0)]
    o += [c for c in e.schema.columns() if c not in set(o)]
    orders.append((o, len(B0)))
    lo, hi = 0, len(o) - len(B0)
    while lo < hi:
        mid = (lo + hi) // 2
        if covers(e.gold, set(closure(e.schema, o, len(B0) + mid, f1))):
            hi = mid
        else:
            lo = mid + 1
    r2.append(lo)
r2 = np.asarray(r2)
def lam_hat(idx, eps):
    rr = np.sort(r2[idx]); j = math.ceil((len(rr) + 1) * (1 - eps)); return int(rr[j - 1]) if j <= len(rr) else 10 ** 9
size = lambda i, lam: len(closure(dev[i].schema, orders[i][0], min(len(orders[i][0]), orders[i][1] + lam), f1))
perm = np.random.default_rng(0).permutation(n); A_, B_ = perm[: n // 2], perm[n // 2:]
rng = np.random.default_rng(0); out = {"base_recall": float(np.mean(r2 == 0)), "base_size": float(np.mean([o[1] for o in orders]))}
print(f"{tag}: base recall {out['base_recall']:.4f}, base size {out['base_size']:.1f}")
with open(os.path.join(RES, f"cert_{tag}_genprompts.jsonl"), "w", encoding="utf-8") as fg, \
        open(os.path.join(RES, f"tasks_cert_{tag}.jsonl"), "w", encoding="utf-8") as ft:
    seen = set()
    for eps in (0.05, 0.10):
        cov, sz = [], []
        for _ in range(200):
            p = rng.permutation(n); c_, t_ = p[: n // 2], p[n // 2:]; lh = lam_hat(c_, eps)
            cov.append(np.mean(r2[t_] <= lh)); sz.append(np.mean([size(i, lh) for i in t_]))
        lamA, lamB = lam_hat(B_, eps), lam_hat(A_, eps); lam_i = {int(i): lamA for i in A_}; lam_i.update({int(i): lamB for i in B_})
        out[f"eps{eps}"] = {"coverage": float(np.mean(cov)), "p05": float(np.percentile(cov, 5)), "size": float(np.mean(sz)), "lams": [lamA, lamB]}
        print(f"  eps={eps}: coverage {np.mean(cov):.4f} (p05 {np.percentile(cov,5):.4f}) size {np.mean(sz):.1f} lam {lamA},{lamB}")
        for i, e in enumerate(dev):
            o, nb = orders[i]; cols = closure(e.schema, o, min(len(o), nb + lam_i[i]), f1)
            gp = prompt_of(e, cols); hg = hashlib.md5(gp.encode("utf-8")).hexdigest()
            ft.write(json.dumps({"sys": f"cert_{tag}_e{eps}", "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
            if hg not in seen:
                seen.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
    print("  generation prompts", len(seen))
json.dump(out, open(os.path.join(RES, f"cert_{tag}.json"), "w"), indent=1)
