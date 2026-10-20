"""Table requirements (reviewer request): a base table that the reference SQL reads (FROM/JOIN, EXISTS, COUNT(*) ...)
must appear in the prompt even when no column of it is referenced.  For every such table T without a column
requirement inside T we add the requirement R_T = C(T): at least one column of T is kept (the prompt then contains T).
augment(exs) edits e.gold.requirements in place; v2data.load_examples applies it when the env var TABREQ=1."""
import os
import sqlglot
from sqlglot import exp


def base_tables(sql, schema):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return set()
    out = set()
    for t in (tree.find_all(exp.Table) if tree is not None else []):
        tb = schema.table((t.name or "").strip("`\"[]").lower())
        if tb:
            out.add(tb.name)
    return out


def augment(exs):
    added = 0
    for e in exs:
        g = e.gold
        if g is None:
            continue
        reqs = list(g.requirements)
        if os.environ.get("STRICT") == "1":  # sensitivity: every alternative of an ambiguous reference is required
            reqs = [{p} for r in reqs for p in sorted(r)]
        for t in sorted(base_tables(e.sql, e.schema)):
            cols = {(t, c.name) for c in e.schema.table(t).columns}
            if not any(r and r <= cols for r in reqs):
                reqs.append(cols)
                added += 1
        g.requirements = reqs
    return added


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v2data import load_examples
    os.environ.pop("TABREQ", None)
    dev = load_examples("dev")
    before = [len(e.gold.requirements) for e in dev]
    n = augment(dev)
    aff = [i for i, e in enumerate(dev) if len(e.gold.requirements) > before[i]]
    print("table-only references added:", n, "questions affected:", len(aff), f"({len(aff) / len(dev):.4f})")
    for i in aff[:8]:
        print(i, dev[i].sql[:220])
