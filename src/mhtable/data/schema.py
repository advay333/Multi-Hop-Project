"""
SchemaGraph: tables, columns, primary keys and declared foreign keys of one database.

Built from a Spider/BIRD `tables.json` entry (human-annotated keys) or read from a SQLite file
(MMQA's materialised tables, which declare no keys). Lookups are case-insensitive: every key is
lowercased; original spellings are kept for writing SQL.

Edges are single column pairs. `tables.json` lists FK column pairs individually and cannot
distinguish a composite key from two independent FKs between the same tables, so composite
keys appear as several parallel edges (DESIGN.md §5.3 allows multi-pair edges once the oracle
needs them).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from mhtable.types import Edge


def col_key(table: str, column: str) -> str:
    return f"{table.lower()}.{column.lower()}"


def make_edge(a: str, b: str) -> Edge:
    """Undirected single-pair edge, canonically ordered so equal joins compare equal."""
    return (tuple(sorted((a.lower(), b.lower()))),)


@dataclass
class SchemaGraph:
    db_id: str
    table_names: dict[str, str] = field(default_factory=dict)  # lower -> original
    columns: dict[str, list[str]] = field(default_factory=dict)  # lower table -> original column names, in order
    column_types: dict[str, str] = field(default_factory=dict)  # "t.c" -> declared type (lowercase)
    pks: set[str] = field(default_factory=set)  # {"t.c"}
    fks: set[Edge] = field(default_factory=set)  # declared FK edges

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_tables_json(cls, entry: dict) -> "SchemaGraph":
        g = cls(entry["db_id"])
        tabs = entry["table_names_original"]
        for t in tabs:
            g.table_names[t.lower()] = t
            g.columns[t.lower()] = []
        keys: list[str | None] = []
        types = entry.get("column_types") or [""] * len(entry["column_names_original"])
        for (ti, c), ty in zip(entry["column_names_original"], types):
            if ti < 0:  # the "*" pseudo-column
                keys.append(None)
                continue
            t = tabs[ti]
            g.columns[t.lower()].append(c)
            g.column_types[col_key(t, c)] = ty.lower()
            keys.append(col_key(t, c))
        for pk in entry.get("primary_keys", []):
            for i in (pk if isinstance(pk, list) else [pk]):  # BIRD lists composite PKs as lists
                if keys[i]:
                    g.pks.add(keys[i])
        for a, b in entry.get("foreign_keys", []):
            if keys[a] and keys[b] and keys[a] != keys[b]:
                g.fks.add(make_edge(keys[a], keys[b]))
        return g

    @classmethod
    def from_sqlite(cls, db_id: str, path: str | Path) -> "SchemaGraph":
        """Tables/columns/types and any declared keys from the SQLite catalogue."""
        g = cls(db_id)
        con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
            for t in names:
                g.table_names[t.lower()] = t
                g.columns[t.lower()] = []
                for _, c, ty, _, _, pk in con.execute(f'PRAGMA table_info("{t}")'):
                    g.columns[t.lower()].append(c)
                    g.column_types[col_key(t, c)] = (ty or "").lower()
                    if pk:
                        g.pks.add(col_key(t, c))
            for t in names:
                for row in con.execute(f'PRAGMA foreign_key_list("{t}")'):
                    ref_table, frm, to = row[2], row[3], row[4]
                    if ref_table.lower() in g.table_names and to:
                        g.fks.add(make_edge(col_key(t, frm), col_key(ref_table, to)))
        finally:
            con.close()
        return g

    # ------------------------------------------------------------------ queries
    @property
    def tables(self) -> list[str]:
        return list(self.table_names)

    def has_column(self, table: str, column: str) -> bool:
        return col_key(table, column) in self.column_types

    def column_set(self, table: str) -> set[str]:
        return {c.lower() for c in self.columns.get(table.lower(), [])}

    def is_pk(self, table: str, column: str) -> bool:
        return col_key(table, column) in self.pks

    def edges_from(self, table: str) -> list[Edge]:
        t = table.lower() + "."
        return sorted(e for e in self.fks if any(c.startswith(t) for c in e[0]))

    def is_fk_edge(self, edge: Edge) -> bool:
        return all(make_edge(*pair) in self.fks for pair in edge)

    def to_dict(self) -> dict:
        return {"db_id": self.db_id, "table_names": self.table_names, "columns": self.columns,
                "column_types": self.column_types, "pks": sorted(self.pks),
                "fks": sorted([list(p) for p in e] for e in self.fks)}

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaGraph":
        return cls(d["db_id"], dict(d["table_names"]), {k: list(v) for k, v in d["columns"].items()},
                   dict(d["column_types"]), set(d["pks"]),
                   {tuple(tuple(p) for p in e) for e in d["fks"]})
