"""RSL-SQL linking as column lists (union of its step-1 and step-2 linking, as in v2_rsl_eval.py), importable."""
import json, re, os
from jevschema.paths import DATA

col_re = re.compile(r"^\s*`?([^`.]+?)`?\s*\.\s*`?(.+?)`?\s*$")


def to_pairs(e, items):
    lut = {(t.name.lower(), c.name.lower()): (t.name, c.name) for t in e.schema.tables for c in t.columns}
    out = []
    for s in items or []:
        m = col_re.match(str(s))
        if m and (m.group(1).lower(), m.group(2).lower()) in lut:
            p = lut[(m.group(1).lower(), m.group(2).lower())]
            if p not in out:
                out.append(p)
    return out


def rsl_union_cols(dev, m):
    A = json.load(open(os.path.join(DATA, "rsl", f"{m}_LLM.json"), encoding="utf-8"))
    B = json.load(open(os.path.join(DATA, "rsl", f"{m}_augmentation.json"), encoding="utf-8"))
    assert len(A) == len(dev) == len(B)
    return [to_pairs(e, (a.get("columns") or []) + (b.get("columns") or [])) for e, a, b in zip(dev, A, B)]
