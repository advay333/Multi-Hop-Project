"""
BIRD (Li et al., NeurIPS 2023): 69 train + 11 dev large real databases, questions with gold SQL
and an "evidence" hint mapping question wording to values (e.g. "female refers to gender = 'F'").

Each archive holds the question file, a tables.json-style schema file and a nested zip of the
databases. Built on molab only (the databases are ~33 GB extracted).
"""

from __future__ import annotations

import json
from pathlib import Path

from mhtable.data.loaders import RawExample, SourceBundle
from mhtable.data.raw import extract_nested, find_one
from mhtable.data.schema import SchemaGraph

ARCHIVES = {"train": "bird/train.zip", "dev": "bird/dev.zip"}


def load(roots: dict[str, Path], limit: int | None = None) -> SourceBundle:
    """`roots` maps "train"/"dev" to the extracted archives."""
    b = SourceBundle()
    for split, root in sorted(roots.items()):
        questions = json.loads(find_one(root, f"{split}.json").read_text(encoding="utf-8"))
        tables = json.loads(find_one(root, f"{split}_tables.json").read_text(encoding="utf-8"))
        db_root = extract_nested(root, f"{split}_databases.zip")
        for e in tables:
            db = e["db_id"]
            hits = [p for p in db_root.rglob(f"{db}.sqlite") if "__MACOSX" not in p.parts]
            if hits:
                b.db_files[f"bird/{db}"] = hits[0]
                b.schemas[f"bird/{db}"] = SchemaGraph.from_tables_json(e)
        for i, x in enumerate(questions):
            if limit is not None and i >= limit:
                break
            ref = f"bird/{x['db_id']}"
            if ref not in b.db_files:
                continue
            b.examples.append(RawExample(source="bird", ext_id=f"{split}{i}", db_id=x["db_id"], db_ref=ref,
                                         question=x["question"], sql=x["SQL"],
                                         evidence=(x.get("evidence") or None),
                                         extra={"bird_split": split, "difficulty": x.get("difficulty")}))
    b.stats = {"bird_dbs": len(b.db_files)}
    return b
