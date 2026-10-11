"""
MMQA ships each question's tables inline (column names + row lists) rather than as database
files. This module turns them into SQLite files and maps them back to their Spider database.

Findings this relies on (verified 2026-10-11 on the mirror):
- Table contents are a pure function of the schema: 484 distinct two-table and 175 distinct
  three-table schemas, each with exactly one content version. So one SQLite file per schema
  signature, shared by every question that uses it (659 files, not 3,313).
- Cell values are loosely typed (ints, numeric strings, floats in one row). Column types are
  inferred per column so numeric comparisons in the gold SQL behave numerically.
- 3,276 of 3,313 instances match exactly one Spider DB on table names + column sets.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from mhtable.data.norm import norm_number
from mhtable.data.schema import SchemaGraph, col_key, make_edge


def _norm_name(x: str) -> str:
    """Names compared across MMQA files and Spider: lowercase, spaces == underscores."""
    return re.sub(r"[\s_]+", " ", x.lower()).strip()


def schema_signature(record: dict) -> str:
    key = sorted((n.lower(), [c.lower() for c in t["table_columns"]])
                 for n, t in zip(record["table_names"], record["tables"]))
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]


def content_digest(record: dict) -> str:
    key = sorted(((n, t["table_columns"], t["table_content"]) for n, t in zip(record["table_names"], record["tables"])),
                 key=lambda x: x[0].lower())
    return hashlib.sha1(json.dumps(key, default=str).encode()).hexdigest()


def _infer_type(values: list) -> str:
    """INTEGER if every non-null value is integral, REAL if all numeric, else TEXT."""
    seen = False
    integral = True
    for v in values:
        if v is None or v == "":
            continue
        seen = True
        if isinstance(v, bool):
            return "TEXT"
        if isinstance(v, int):
            continue
        if isinstance(v, float):
            integral &= v.is_integer()
            continue
        n = norm_number(str(v))
        if n is None:
            return "TEXT"
        integral &= "." not in n
    if not seen:
        return "TEXT"
    return "INTEGER" if integral else "REAL"


def fix_mojibake(s: str) -> str:
    """Undo UTF-8 text that was decoded as Latin-1/cp1252 ("MisÃ©rables" -> "Misérables").

    MMQA's embedded tables contain such strings where the source Spider databases are correct.
    Correct non-ASCII text fails the round trip (its Latin-1 bytes are not valid UTF-8), so it is
    returned unchanged.
    """
    if s.isascii():
        return s
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        return fixed
    return s


def _coerce(v, ty: str):
    if isinstance(v, str):
        v = fix_mojibake(v)
    if v is None or v == "" or ty == "TEXT":
        return v
    x = v if isinstance(v, (int, float)) else float(norm_number(str(v)))
    return int(x) if ty == "INTEGER" else float(x)


def materialise(record: dict, path: Path) -> None:
    """Write the record's tables to a new SQLite file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        for name, t in zip(record["table_names"], record["tables"]):
            cols = t["table_columns"]
            rows = t["table_content"]
            types = [_infer_type([r[i] if i < len(r) else None for r in rows]) for i in range(len(cols))]
            col_defs = ", ".join(f'"{c}" {ty}' for c, ty in zip(cols, types))
            con.execute(f'CREATE TABLE "{name}" ({col_defs})')
            ph = ", ".join("?" * len(cols))
            con.executemany(f'INSERT INTO "{name}" VALUES ({ph})',
                            ([_coerce(r[i] if i < len(r) else None, ty) for i, ty in enumerate(types)] for r in rows))
        con.commit()
    finally:
        con.close()
    tmp.replace(path)


def spider_matches(record: dict, spider: dict[str, SchemaGraph]) -> list[str]:
    """Spider db_ids containing every table of the record (same name, columns a subset)."""
    inst = [(_norm_name(n), {_norm_name(c) for c in t["table_columns"]})
            for n, t in zip(record["table_names"], record["tables"])]
    hits = []
    for db, g in spider.items():
        tabs = {_norm_name(t): {_norm_name(c) for c in g.column_set(t)} for t in g.tables}
        if all(n in tabs and cols <= tabs[n] for n, cols in inst):
            hits.append(db)
    return sorted(hits)


def schema_for(record: dict, db_path: Path, sig: str, spider_db: SchemaGraph | None) -> tuple[SchemaGraph, str]:
    """SchemaGraph of a materialised MMQA DB, with keys borrowed from Spider when it maps to one.

    Returns (graph, key_source) where key_source is "spider" or "mmqa_lists".
    """
    g = SchemaGraph.from_sqlite(f"mmqa_{sig}", db_path)
    present = {_norm_name(t): t for t in g.tables}

    def local(spider_key: str) -> str | None:
        t, c = spider_key.split(".", 1)
        lt = present.get(_norm_name(t))
        if lt is None:
            return None
        match = [x for x in g.columns[lt] if _norm_name(x) == _norm_name(c)]
        return col_key(lt, match[0]) if match else None

    if spider_db is not None:
        g.pks = {k for k in (local(p) for p in spider_db.pks) if k}
        for e in spider_db.fks:
            pair = [local(c) for c in e[0]]
            if all(pair):
                g.fks.add(make_edge(*pair))
        return g, "spider"

    # Fallback: MMQA's flat key-name lists. A name present in two tables is a join between them.
    for name in {_norm_name(x) for x in record.get("foreign_keys", [])}:
        owners = [col_key(t, c) for t in g.tables for c in g.columns[t] if _norm_name(c) == name]
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                if owners[i].split(".")[0] != owners[j].split(".")[0]:
                    g.fks.add(make_edge(owners[i], owners[j]))
    for name in {_norm_name(x) for x in record.get("primary_keys", [])}:
        g.pks |= {col_key(t, c) for t in g.tables for c in g.columns[t] if _norm_name(c) == name}
    return g, "mmqa_lists"
