"""Baseline: the released CodeS schema-item classifier (cross-encoder linker in the RESDSQL line), scored by
v2_codes_samples.py + codes_score.py (lab GPU) -> cache/codes_dev.jsonl {id, s: {"t|c": table_prob * column_prob}}.
Sets on BIRD-Dev:
  codes_filter   CodeS's own filtering (top-6 tables, top-10 columns per kept table), + our FK closure (as for RSL-SQL)
  codes_cert     CRISP certificate with the CodeS ranking as the reader (fixed budget family, f1k1 closure,
                 split-conformal k_hat, 2-fold cross-fit as everywhere): any linker's ranking can be certified
Prints coverage (SRR) and size; WRITE=1 writes generation prompts/tasks (results/codes_genprompts.jsonl, tasks_codes.jsonl)."""
import hashlib, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import CACHE, RES, View, load_examples, read_scores
from jevschema.selection import closure, parse_closure
from jevschema.certify import kept_at

dev = load_examples("dev")
S = read_scores(os.path.join(CACHE, "codes_dev.jsonl"))
assert all(e.ex_id in S for e in dev), sum(e.ex_id not in S for e in dev)
f1 = parse_closure("f1")
cov = lambda e, k: all(req & k for req in e.gold.requirements)
TP = {}
for l in open(os.path.join(CACHE, "codes_dev.jsonl"), encoding="utf-8"):
    r = json.loads(l); TP[r["id"]] = r["tp"]

sets = {}
filt = []
for e in dev:
    s, tp = S[e.ex_id], TP[e.ex_id]
    tabs = sorted(tp, key=lambda t: -tp[t])[:6]
    k = []
    for t in tabs:
        cols = sorted([p for p in e.schema.columns() if p[0] == t], key=lambda p: -s.get(p, 0.0))[:10]
        k += cols
    order = k + [p for p in e.schema.columns() if p not in set(k)]
    filt.append(set(closure(e.schema, order, len(k), f1)))
sets["codes_filter"] = filt

V = View(dev, [S[e.ex_id] for e in dev], "f1k1")
n = len(dev); perm = np.random.default_rng(0).permutation(n); A, B = perm[: n // 2], perm[n // 2:]


def khat(idx, eps):
    rr = np.sort(V.r[idx]); j = math.ceil((len(rr) + 1) * (1 - eps))
    return int(rr[j - 1]) if j <= len(rr) else 10 ** 9


rng = np.random.default_rng(0)
for eps in (0.05, 0.10):
    cv, sz = [], []
    for _ in range(200):
        p = rng.permutation(n); c, t = p[: n // 2], p[n // 2:]; kh = khat(c, eps)
        cv.append(np.mean(V.r[t] <= kh)); sz.append(np.mean([V.size(i, min(kh, V.n_cols[i])) for i in t[:300]]))
    kA, kB = khat(B, eps), khat(A, eps)
    ki = {int(i): kA for i in A}; ki.update({int(i): kB for i in B})
    sets[f"codes_cert_e{eps}"] = [set(kept_at(e.schema, V.orders[i], min(ki[i], V.n_cols[i]), V.cfg)) for i, e in enumerate(dev)]
    print(f"CRISP(CodeS reader) eps={eps}: coverage {np.mean(cv):.4f} (p05 {np.percentile(cv, 5):.4f}) size~{np.mean(sz):.1f} k_hat {kA},{kB}")
for name, K in sets.items():
    print(f"{name:20s} SRR={np.mean([cov(e, k) for e, k in zip(dev, K)]):.4f} Cbar={np.mean([len(k) for k in K]):.1f}")

if os.environ.get("WRITE") == "1":
    from v2_e2e import prompt_of
    seen = set()
    with open(os.path.join(RES, "codes_genprompts.jsonl"), "w", encoding="utf-8") as fg, \
            open(os.path.join(RES, "tasks_codes.jsonl"), "w", encoding="utf-8") as ft:
        for name, K in sets.items():
            for i, (e, k) in enumerate(zip(dev, K)):
                cols = [p for p in e.schema.columns() if p in k] or list(e.schema.columns())
                gp = prompt_of(e, cols); hg = hashlib.md5(gp.encode("utf-8")).hexdigest()
                ft.write(json.dumps({"sys": name, "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
                if hg not in seen:
                    seen.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
    print("generation prompts", len(seen))
