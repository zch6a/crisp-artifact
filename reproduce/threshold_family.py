"""Build the threshold-family (strongest single-stage) prompts and EX task files.

    python reproduce/threshold_family.py --eps 0.05 --out results/

Reads the released reader scores (cache/jev_t10_dev.jsonl) and BIRD-Dev metadata
(data/raw/bird/dev.json, dev_tables.json), and writes
    results/thresh_e<eps>_genprompts.jsonl   {h, prompt}
    results/tasks_thresh_e<eps>.jsonl        {sys, i, h, db, gold}
The threshold family is the budget family used in Table 6; at eps=.05 it keeps 16.8
columns and covers 0.9505 of BIRD-Dev.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "code", "scripts"))

import numpy as np
from jevschema.certify import calibrate, covers, kept_at, thresh_family
from jevschema.tokens import build_gen_prompt, schema_prompt_text
from v2data import View, fused_scores, lexical, load_examples, read_scores
from v2_cascade import tier1_scores
from jevschema.paths import CACHE, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--tier1", default="jev_t10")
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--out", default=RESULTS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = load_examples("dev")
    bm, em = lexical("dev")
    s0 = [fused_scores(e, bm.get(e.ex_id, {}), em.get(e.ex_id, {})) for e in dev]
    ld = read_scores(os.path.join(CACHE, f"{a.tier1}_dev.jsonl"))
    s1 = tier1_scores(dev, s0, [ld[e.ex_id] for e in dev], a.beta)
    v = View(dev, s1, "f1k1")
    fam = thresh_family(v.r, v.n_cols, [v.sorted_scores(i) for i in range(len(dev))])
    fold = np.random.default_rng(0).permutation(len(dev)) % 2
    lam = np.zeros(len(dev))
    for f in (0, 1):
        cal = [i for i in range(len(dev)) if fold[i] != f]
        te = [i for i in range(len(dev)) if fold[i] == f]
        L = calibrate(fam, cal, a.eps)
        for i in te:
            lam[i] = L
    kept = [kept_at(dev[i].schema, v.orders[i], fam.budget(i, lam[i]), v.cfg) for i in range(len(dev))]
    cov = float(np.mean([1.0 if covers(dev[i].gold, set(kept[i])) else 0.0 for i in range(len(dev))]))
    size = float(np.mean([len(k) for k in kept]))
    tag = f"thresh_e{a.eps:g}"
    pf = os.path.join(a.out, f"{tag}_genprompts.jsonl")
    tf = os.path.join(a.out, f"tasks_{tag}.jsonl")
    with open(pf, "w", encoding="utf-8") as fp, open(tf, "w", encoding="utf-8") as ft:
        for i, e in enumerate(dev):
            k = kept[i]
            txt = schema_prompt_text(e.schema, tables=sorted({t for t, _ in k}), columns=k, with_samples=0)
            p = build_gen_prompt(e.db_id, txt, e.question, e.evidence or "")
            h = hashlib.md5(p.encode("utf-8")).hexdigest()
            fp.write(json.dumps({"h": h, "prompt": p}, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"sys": tag, "i": i, "h": h, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
    print(json.dumps({"eps": a.eps, "coverage": cov, "columns": size, "prompts": pf, "tasks": tf}, indent=1))


if __name__ == "__main__":
    main()
