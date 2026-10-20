"""Same-condition comparison with RSL-SQL (Cao et al. 2024) using its *released* per-question BIRD-Dev linking outputs
(main_result/BIRD_DEV/{gpt-4o,deepseek}/step1/LLM.json = bidirectional schema linking; step2/augmentation.json =
information augmentation), scored with our gold requirements and our SRR / column-count metrics.
Sets: step1, step2, union(step1, step2); each also with our FK closure (f1) for reference.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v2data import RES, DATA, load_examples
from jevschema.selection import closure, parse_closure

dev = load_examples("dev")
col_re = re.compile(r"^\s*`?([^`.]+?)`?\s*\.\s*`?(.+?)`?\s*$")


def to_pairs(e, items):
    lut = {(t.name.lower(), c.name.lower()): (t.name, c.name) for t in e.schema.tables for c in t.columns}
    out = set()
    for s in items or []:
        m = col_re.match(str(s))
        if m and (m.group(1).lower(), m.group(2).lower()) in lut:
            out.add(lut[(m.group(1).lower(), m.group(2).lower())])
    return out


def covered(e, k):
    return all(req & k for req in e.gold.requirements)


f1 = parse_closure("f1")
res = {}
for m in ("gpt-4o", "deepseek"):
    A = json.load(open(f"{DATA}/rsl/{m}_LLM.json", encoding="utf-8"))
    B = json.load(open(f"{DATA}/rsl/{m}_augmentation.json", encoding="utf-8"))
    assert len(A) == len(dev) and len(B) == len(dev), (len(A), len(B), len(dev))
    sets = {"step1": [to_pairs(e, a.get("columns")) for e, a in zip(dev, A)],
            "step2": [to_pairs(e, b.get("columns")) for e, b in zip(dev, B)]}
    sets["union"] = [x | y for x, y in zip(sets["step1"], sets["step2"])]
    for name, S in list(sets.items()):
        fk = []
        for e, k in zip(dev, S):
            order = list(k) + [p for p in e.schema.columns() if p not in k]
            fk.append(set(closure(e.schema, order, len(k), f1)) if k else set())
        sets[name + "+FK"] = fk
    for name, S in sets.items():
        srr = float(np.mean([covered(e, k) for e, k in zip(dev, S)]))
        cb = float(np.mean([len(k) for k in S]))
        res[f"{m}:{name}"] = {"srr": srr, "cbar": cb}
        print(f"RSL-SQL {m:9s} {name:12s} SRR={srr:.4f} Cbar={cb:5.1f}")
json.dump(res, open(os.path.join(RES, "rsl_eval.json"), "w"), indent=1)

# Same-generator EX comparison (reviewer request): the RSL-SQL linked schema (union of its two steps + our FK closure,
# its best-recall setting) is given to exactly the generators and prompt template used for every CRISP system.
# Empty linking falls back to the full schema.  -> results/rsl_genprompts.jsonl, tasks_rsl.jsonl
if os.environ.get("WRITE") == "1":
    import hashlib
    from v2_e2e import prompt_of
    seen = set()
    with open(os.path.join(RES, "rsl_genprompts.jsonl"), "w", encoding="utf-8") as fg, \
            open(os.path.join(RES, "tasks_rsl.jsonl"), "w", encoding="utf-8") as ft:
        for m in ("gpt-4o", "deepseek"):
            A = json.load(open(f"{DATA}/rsl/{m}_LLM.json", encoding="utf-8"))
            B = json.load(open(f"{DATA}/rsl/{m}_augmentation.json", encoding="utf-8"))
            for i, (e, a, b) in enumerate(zip(dev, A, B)):
                k = to_pairs(e, a.get("columns")) | to_pairs(e, b.get("columns"))
                order = list(k) + [p for p in e.schema.columns() if p not in k]
                cols = closure(e.schema, order, len(k), f1) if k else list(e.schema.columns())
                gp = prompt_of(e, cols); hg = hashlib.md5(gp.encode("utf-8")).hexdigest()
                ft.write(json.dumps({"sys": f"rsl_{m}_union_fk", "i": i, "h": hg, "db": e.db_id, "gold": e.sql}, ensure_ascii=False) + "\n")
                if hg not in seen:
                    seen.add(hg); fg.write(json.dumps({"h": hg, "prompt": gp}, ensure_ascii=False) + "\n")
    print("RSL generation prompts", len(seen))
