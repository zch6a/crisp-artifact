"""Hard schema-scaling benchmark: distractors are *copies of the target database's own tables*
(T_archive, T_backup, T_staging, ...), with identical column names and the original FK structure
replicated among the copies.  This mimics enterprise schemas full of archive / staging / history
tables: lexical matching collides on every column name and FK closure can walk into a copy.
Gold requirements refer to the original tables; SQL still executes on the original sqlite DB.
"""
from __future__ import annotations

from .data_loader import Example
from .schema_graph import Column, ForeignKey, SchemaGraph, Table

SUFFIXES = ["archive", "backup", "staging", "hist", "old", "tmp", "snapshot", "v2", "bak", "raw",
            "stage2", "legacy", "mirror", "export", "import", "copy", "audit", "log", "cache", "shadow"]


class HardScaler:
    def merge(self, ex, n_target):
        base = ex.schema
        tables = [Table(t.name, [Column(c.name, c.dtype, c.is_pk, list(c.sample_values)) for c in t.columns])
                  for t in base.tables]
        fks = list(base.fks)
        n = base.n_columns()
        j = 0
        while n < n_target:
            sfx = SUFFIXES[j % len(SUFFIXES)] + ("" if j < len(SUFFIXES) else str(j // len(SUFFIXES)))
            j += 1
            for t in base.tables:
                tables.append(Table(f"{t.name}_{sfx}", [Column(c.name, c.dtype, c.is_pk, []) for c in t.columns]))
                n += len(t.columns)
            for f in base.fks:
                fks.append(ForeignKey(f"{f.src_table}_{sfx}", f.src_col, f"{f.dst_table}_{sfx}", f.dst_col))
        sch = SchemaGraph(db_id=base.db_id, tables=tables, fks=fks)
        e2 = Example(ex_id=ex.ex_id, db_id=ex.db_id, question=ex.question, sql=ex.sql, schema=sch,
                     evidence=ex.evidence, difficulty=ex.difficulty)
        e2.gold = ex.gold
        return e2
