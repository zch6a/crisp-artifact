"""Schema-scaling benchmark built from BIRD-Dev: each question's database is merged with
distractor tables from BIRD-train holdout schemas (renamed d{j}_<table>, own FKs kept) until it
has >= N columns.  Gold requirements are untouched (original tables keep their names), so SRR,
closure, and execution against the original sqlite remain valid.  Deterministic per (N, ex_id).
"""
from __future__ import annotations

import random
import zlib

from .data_loader import Example
from .schema_graph import Column, ForeignKey, SchemaGraph, Table


def _pool(train_examples):
    seen, pool = set(), []
    for e in train_examples:
        key = tuple(sorted(t.name.lower() for t in e.schema.tables))
        if key in seen:
            continue
        seen.add(key)
        pool.append(e.schema)
    return pool


class Scaler:
    def __init__(self, train_examples):
        self.pool = _pool(train_examples)

    def merge(self, ex, n_target):
        base = ex.schema
        rng = random.Random(zlib.crc32(f"{ex.ex_id}|{n_target}".encode()))
        tables = [Table(t.name, [Column(c.name, c.dtype, c.is_pk, list(c.sample_values)) for c in t.columns])
                  for t in base.tables]
        fks = list(base.fks)
        n = base.n_columns()
        order = list(range(len(self.pool)))
        rng.shuffle(order)
        j = 0
        for pi in order:
            if n >= n_target:
                break
            g = self.pool[pi]
            j += 1
            pfx = f"d{j}_"
            for t in g.tables:
                tables.append(Table(pfx + t.name, [Column(c.name, c.dtype, c.is_pk, []) for c in t.columns]))
                n += len(t.columns)
            for f in g.fks:
                fks.append(ForeignKey(pfx + f.src_table, f.src_col, pfx + f.dst_table, f.dst_col))
        sch = SchemaGraph(db_id=base.db_id, tables=tables, fks=fks)
        e2 = Example(ex_id=ex.ex_id, db_id=ex.db_id, question=ex.question, sql=ex.sql, schema=sch,
                     evidence=ex.evidence, difficulty=ex.difficulty)
        e2.gold = ex.gold
        return e2
