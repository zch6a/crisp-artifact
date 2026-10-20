"""Spider 2.0 SQLite subset (135 questions, OmniSQL release data/spider2_sqlite): examples for the reader.

Schemas are read from the released SQLite files (PRAGMA table_info / foreign_key_list), so table and column names
are exactly those of the official DDL text that is pruned later.  No public gold SQL exists for most instances, so
``gold`` is None: EX is measured against gold_exec_result, coverage cannot be.  external_knowledge (markdown
documents for some instances) is passed to the reader as evidence, truncated to EVID_CHARS characters.
"""
from __future__ import annotations

import json
import os
import sqlite3

from .data_loader import Example
from .schema_graph import Column, ForeignKey, SchemaGraph, Table
from .paths import EXT

ROOT = os.environ.get("SPIDER2_SQLITE", os.path.join(EXT, "omnidata", "data", "spider2_sqlite"))
EVID_CHARS = 3000


def _schema(db_id: str) -> SchemaGraph:
    con = sqlite3.connect(f"file:{ROOT}/databases_original/{db_id}.sqlite?mode=ro", uri=True)
    tabs, fks = [], []
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
    for t in names:
        cols = [Column(r[1], (r[2] or "text").lower(), bool(r[5]), []) for r in con.execute(f'PRAGMA table_info("{t}")')]
        tabs.append(Table(t, cols))
        for r in con.execute(f'PRAGMA foreign_key_list("{t}")'):
            if r[4] is not None:
                fks.append(ForeignKey(t, r[3], r[2], r[4]))
    con.close()
    tn = {t.name.lower() for t in tabs}
    fks = [f for f in fks if f.dst_table.lower() in tn]
    return SchemaGraph(db_id=db_id, tables=tabs, fks=fks)


def load_spider2sqlite():
    D = json.load(open(os.path.join(ROOT, "test.json"), encoding="utf-8"))
    cache, out = {}, []
    for it in D:
        db = it["db_id"]
        if db not in cache:
            cache[db] = _schema(db)
        out.append(Example(ex_id=it["instance_id"], db_id=db, question=it["question"], sql="", schema=cache[db],
                           evidence=(it.get("external_knowledge") or "")[:EVID_CHARS], difficulty="sqlite"))
    return out
