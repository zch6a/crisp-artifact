"""数据准备：raw 落盘 + gold 抽取 + 统计体检。

为什么要单独一步：gold 抽取（sqlglot 解析）有失败率，如果每次实验都临时抽一次，
不同脚本之间可能拿到不同的 gold（解析失败就退化到正则），结果不可比。
这里抽一次、存盘、后续所有脚本只读缓存。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from jevschema.data_loader import (  # noqa: E402
    DATA_DIR, RAW_DIR, PROCESSED_DIR, load_spider_split,
)
from jevschema.gold_labels import gold_stats  # noqa: E402
from jevschema.schema_graph import load_spider_schemas  # noqa: E402

# raw 文件来源：优先 <CRISP_DATA>/raw 下已下载的，其次仓库根目录下的 .tmp
CANDIDATE_SOURCES = [
    os.path.join(DATA_DIR, "raw"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), ".tmp"),
]

FILES = [
    ("spider_schema.json", "spider_schema.json"),
    ("spider_val.parquet", "spider_val.parquet"),
    ("spider_train.parquet", "spider_train.parquet"),
    ("bird_mini.json", "bird_mini_dev_sqlite.json"),
]


def ensure_raw() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for src_name, dst_name in FILES:
        dst = os.path.join(RAW_DIR, dst_name)
        if os.path.exists(dst):
            continue
        for cand in CANDIDATE_SOURCES:
            p = os.path.join(cand, src_name)
            if os.path.exists(p):
                shutil.copy2(p, dst)
                print(f"copied {dst_name} <- {p}")
                break
        else:
            print(f"[warn] 缺少 {dst_name}（不影响 Spider 实验，BIRD 需要它）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ensure_raw()
    schemas = load_spider_schemas(os.path.join(RAW_DIR, "spider_schema.json"))
    print(f"schema: {len(schemas)} dbs, "
          f"avg columns/db = {sum(s.n_columns() for s in schemas.values())/len(schemas):.1f}, "
          f"avg tables/db = {sum(s.n_tables() for s in schemas.values())/len(schemas):.1f}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for split, fname in (("dev", "spider_val"), ("train", "spider_train")):
        path = os.path.join(RAW_DIR, f"{fname}.parquet")
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在")
            continue
        limit = a.limit or None
        exs = load_spider_split(path, schemas, limit=limit)
        stats = gold_stats([e.gold for e in exs if e.gold])
        out = os.path.join(PROCESSED_DIR, f"spider_{split}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps({
                    "ex_id": e.ex_id, "db_id": e.db_id, "question": e.question, "sql": e.sql,
                    "n_columns": e.schema.n_columns(), "n_tables": e.schema.n_tables(),
                    "gold_reqs": [sorted(list(r)) for r in e.gold.requirements],
                    "gold_tables": sorted(e.gold.tables),
                    "parse_ok": e.gold.parse_ok,
                }, ensure_ascii=False) + "\n")
        print(f"{split}: {len(exs)} examples -> {out}")
        print("   gold stats:", json.dumps({k: round(v, 4) for k, v in stats.items()}))


if __name__ == "__main__":
    main()
