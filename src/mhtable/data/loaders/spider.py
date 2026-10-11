"""
Spider (Yu et al., 2018): human-written questions with gold SQL over 206 SQLite databases.

All four question files are loaded — train_spider, train_others, dev and test — because the
split is by database across Spider and MMQA together (PLAN.md §0), not by Spider's own splits.
Keys come from Spider's annotated `tables.json` / `test_tables.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from mhtable.data.loaders import RawExample, SourceBundle
from mhtable.data.raw import find_one
from mhtable.data.schema import SchemaGraph

ARCHIVE = "spider/spider.zip"
QUESTION_FILES = ("train_spider.json", "train_others.json", "dev.json", "test.json")


def load(root: Path, limit: int | None = None) -> SourceBundle:
    """`root` is the extracted spider.zip."""
    base = find_one(root, "tables.json").parent
    b = SourceBundle()

    entries = json.loads((base / "tables.json").read_text(encoding="utf-8"))
    if (base / "test_tables.json").exists():
        seen = {e["db_id"] for e in entries}
        entries += [e for e in json.loads((base / "test_tables.json").read_text(encoding="utf-8"))
                    if e["db_id"] not in seen]
    for e in entries:
        db = e["db_id"]
        for folder in ("database", "test_database"):
            f = base / folder / db / f"{db}.sqlite"
            if f.exists():
                b.db_files[f"spider/{db}"] = f
                break
        else:
            continue  # schema without a database file: no questions can be executed
        b.schemas[f"spider/{db}"] = SchemaGraph.from_tables_json(e)

    missing = 0
    for fname in QUESTION_FILES:
        path = base / fname
        if not path.exists():
            continue
        for i, x in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            if limit is not None and i >= limit:
                break
            ref = f"spider/{x['db_id']}"
            if ref not in b.db_files:
                missing += 1
                continue
            b.examples.append(RawExample(source="spider", ext_id=f"{Path(fname).stem}{i}", db_id=x["db_id"],
                                         db_ref=ref, question=x["question"], sql=x["query"],
                                         extra={"spider_split": Path(fname).stem}))
    b.stats = {"spider_dbs": len(b.db_files), "questions_without_db": missing}
    return b
