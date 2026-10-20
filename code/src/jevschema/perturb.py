"""Schema perturbations for the reader-contamination test (Dr.Spider-style "schema abbreviation").

Every table / column name is rewritten deterministically; the question and evidence are left untouched, so a reader
that memorized BIRD-Dev schemas loses its cue while a reader that matches meaning keeps most of its recall.
  abbreviate(name): split into tokens (snake / space / camelCase); tokens longer than 3 characters keep their first
  letter plus following consonants, capped at 4 characters ("customer" -> "cstm", "ProductID" -> "PrdcID").
Collisions inside a table (or between tables) get a numeric suffix.  Gold requirements are mapped with the same table.
"""
from __future__ import annotations

import re

from .data_loader import Example
from .gold_labels import GoldSet
from .schema_graph import Column, ForeignKey, SchemaGraph, Table

_TOK = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[^A-Za-z\d]+")


def _abbr_token(t: str) -> str:
    if not t.isalpha() or len(t) <= 3:
        return t
    head, rest = t[0], "".join(ch for ch in t[1:] if ch.lower() not in "aeiou")
    return (head + rest)[:4]


def abbreviate(name: str) -> str:
    toks = _TOK.findall(name)
    out = "".join(_abbr_token(t) if t.strip() and not re.fullmatch(r"[^A-Za-z\d]+", t) else ("_" if t.strip() == "" else t)
                  for t in toks)
    return out or name


def _unique(name, used):
    base, k = name, 2
    while name.lower() in used:
        name = f"{base}{k}"; k += 1
    used.add(name.lower())
    return name


def perturb_example(e: Example, fn=abbreviate) -> Example:
    tmap, cmap, used_t = {}, {}, set()
    tables = []
    for t in e.schema.tables:
        nt = _unique(fn(t.name), used_t); tmap[t.name] = nt
        used_c, cols = set(), []
        for c in t.columns:
            nc = _unique(fn(c.name), used_c); cmap[(t.name, c.name)] = (nt, nc)
            cols.append(Column(nc, c.dtype, c.is_pk, list(c.sample_values)))
        tables.append(Table(nt, cols))
    fks = [ForeignKey(cmap[(f.src_table, f.src_col)][0], cmap[(f.src_table, f.src_col)][1],
                      cmap[(f.dst_table, f.dst_col)][0], cmap[(f.dst_table, f.dst_col)][1])
           for f in e.schema.fks if (f.src_table, f.src_col) in cmap and (f.dst_table, f.dst_col) in cmap]
    sch = SchemaGraph(db_id=e.schema.db_id, tables=tables, fks=fks)
    e2 = Example(ex_id=e.ex_id, db_id=e.db_id, question=e.question, sql=e.sql, schema=sch, evidence=e.evidence,
                 difficulty=e.difficulty)
    if e.gold is not None:
        g = GoldSet()
        g.requirements = [{cmap[p] for p in req if p in cmap} for req in e.gold.requirements]
        g.tables = {tmap.get(t, t) for t in e.gold.tables}
        for k, v in vars(e.gold).items():
            if k not in ("requirements", "tables"):
                setattr(g, k, v)
        e2.gold = g
    return e2
