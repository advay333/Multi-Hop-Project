"""
Per-database index the environment needs before any episode runs (report C.2, DESIGN.md §6.4):

  column stats   rows, non-null, distinct, uniqueness u, numeric?, average text length
  edge candidates  JAR-style joinability for column pairs (one side must be unique):
                   e = name similarity, j = value containment, u = uniqueness,
                   omega = (e + j) * max(u_k, u_l)
  cell index     distinct values of non-ID columns (TableRAG-style), capped per column,
                 searchable by question tokens

Built offline once per database and stored as `index/<db_ref>.json` in the data repo, where
the environment loads it from:

    python -m mhtable.env.index --family spider_mmqa --revision data-v2 --push   # molab / laptop
    python -m mhtable.env.index --family spider_mmqa --data-dir <local repo copy> --out data/cache/index Uses only the database
and its schema — never gold information.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mhtable.data.execute import QueryTimeout, connect_ro, run
from mhtable.data.norm import norm_value
from mhtable.data.schema import SchemaGraph, make_edge
from mhtable.env.text import name_similarity, plain, tokens
from mhtable.sqlrender import column_sql, table_sql
from mhtable.types import Edge

NUMERIC_TYPES = ("int", "real", "num", "float", "double", "dec")
VALUES_PER_COLUMN = 10000  # most frequent distinct values indexed per column (TableRAG: 10k per table)
MAX_VALUE_CHARS = 100  # longer texts are not anchor candidates
CONTAINMENT_SAMPLE = 200  # distinct values sampled from the non-unique side
MIN_NAME_SIM = 0.25  # pairs below this name similarity are only kept if declared FKs
STAT_TIMEOUT_S = 20.0


@dataclass
class ColumnStats:
    key: str  # "table.column"
    n_rows: int
    n_nonnull: int
    n_distinct: int
    numeric: bool
    avg_chars: float
    min: float | None = None  # numeric columns only: value range, used to rank numeric anchors
    max: float | None = None

    @property
    def u(self) -> float:
        return self.n_distinct / self.n_nonnull if self.n_nonnull else 0.0

    @property
    def unique(self) -> bool:
        return self.n_nonnull > 0 and self.n_distinct == self.n_nonnull

    @property
    def id_like(self) -> bool:
        """Unique integers (surrogate keys): reached by joins, never good anchor values."""
        return self.u >= 0.99 and self.numeric and self.n_nonnull > 1


@dataclass
class EdgeCandidate:
    edge: Edge
    e: float  # name similarity
    j: float  # containment of the non-unique side's values in the unique side
    u: float  # max uniqueness of the two columns
    omega: float
    declared: bool  # a declared FK in the schema (used only by the `declared` edge mode)


@dataclass
class DbIndex:
    db_ref: str
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    edges: list[EdgeCandidate] = field(default_factory=list)
    cells: dict[str, list] = field(default_factory=dict)  # column key -> [raw values], most frequent first

    def to_dict(self) -> dict:
        return {"db_ref": self.db_ref, "columns": {k: asdict(v) for k, v in self.columns.items()},
                "edges": [asdict(e) for e in self.edges], "cells": self.cells}

    @classmethod
    def from_dict(cls, d: dict) -> "DbIndex":
        return cls(d["db_ref"], {k: ColumnStats(**v) for k, v in d["columns"].items()},
                   [EdgeCandidate(tuple(tuple(p) for p in e["edge"]), e["e"], e["j"], e["u"], e["omega"], e["declared"])
                    for e in d["edges"]],
                   {k: list(v) for k, v in d["cells"].items()})


# ---------------------------------------------------------------- building
def _is_numeric(schema: SchemaGraph, key: str, sample: list) -> bool:
    declared = schema.column_types.get(key, "")
    if any(t in declared for t in NUMERIC_TYPES):
        return True
    vals = [v for v in sample if v is not None]
    return bool(vals) and all(isinstance(v, (int, float)) for v in vals)


def column_stats(con, schema: SchemaGraph) -> dict[str, ColumnStats]:
    out = {}
    for t in schema.tables:
        tsql = table_sql(schema, t)
        try:
            n_rows = run(con, f"SELECT COUNT(*) FROM {tsql}", STAT_TIMEOUT_S)[0][0]
        except QueryTimeout:
            continue
        for c in schema.columns[t]:
            key = f"{t}.{c.lower()}"
            col = column_sql(schema, key)
            try:
                nn, nd, avg, lo, hi = run(con, f"SELECT COUNT({col}), COUNT(DISTINCT {col}), AVG(LENGTH({col})), "
                                               f"MIN({col}), MAX({col}) FROM {tsql}", STAT_TIMEOUT_S)[0]
                sample = [r[0] for r in run(con, f"SELECT {col} FROM {tsql} WHERE {col} IS NOT NULL LIMIT 50",
                                            STAT_TIMEOUT_S)]
            except QueryTimeout:
                continue
            numeric = _is_numeric(schema, key, sample)
            num = (lambda v: float(v) if isinstance(v, (int, float)) else None)
            out[key] = ColumnStats(key, n_rows, nn, nd, numeric, float(avg or 0),
                                   num(lo) if numeric else None, num(hi) if numeric else None)
    return out


def _containment(con, schema, child: str, parent: str) -> float:
    """Fraction of sampled distinct child values that occur in the parent column."""
    c_t, p_t = child.split(".")[0], parent.split(".")[0]
    c, p = column_sql(schema, child), column_sql(schema, parent)
    sql = (f"SELECT COUNT(*), SUM(v IN (SELECT {p} FROM {table_sql(schema, p_t)})) FROM "
           f"(SELECT DISTINCT {c} AS v FROM {table_sql(schema, c_t)} WHERE {c} IS NOT NULL LIMIT {CONTAINMENT_SAMPLE})")
    n, hit = run(con, sql, STAT_TIMEOUT_S)[0]
    return (hit or 0) / n if n else 0.0


GENERIC_TOKENS = {"id", "code", "key", "no", "num", "name", "type"}


def edge_name_similarity(kt: str, kc: str, ct: str, cc: str) -> float:
    """How much two columns' names suggest a join. A generic column name ("id") only counts
    together with its table name, so customers.id ~ invoices.customer_id but not ~ artists.id."""
    sims = [name_similarity(f"{kt} {kc}", cc), name_similarity(kc, f"{ct} {cc}")]
    if set(tokens(kc)) - GENERIC_TOKENS and set(tokens(cc)) - GENERIC_TOKENS:
        sims.append(name_similarity(kc, cc))
    return max(sims)


def infer_edges(con, schema: SchemaGraph, stats: dict[str, ColumnStats]) -> list[EdgeCandidate]:
    """Candidate joins: (unique column k, column c in another table) with a similar name or a
    declared FK, scored with JAR's omega = (e + j) * max(u). Declared types are not required to
    match (Spider declares some id columns as text on one side and number on the other, and
    SQLite joins them fine); value containment j decides instead. Pairs of two unique surrogate
    keys (1:1 id-to-id) are down-weighted: their value overlap is usually coincidental."""
    out = []
    keys = [s for s in stats.values() if s.unique]
    for k in keys:
        kt, kc = k.key.split(".", 1)
        for c in stats.values():
            ct, cc = c.key.split(".", 1)
            if ct == kt or c.n_nonnull == 0:
                continue
            edge = make_edge(k.key, c.key)
            declared = edge in schema.fks
            e = edge_name_similarity(kt, kc, ct, cc)
            if e < MIN_NAME_SIM and not declared:
                continue
            if c.unique and c.key < k.key and any(x.edge == edge for x in out):
                continue  # both sides unique: the pair was already scored from the other side
            try:
                j = _containment(con, schema, c.key, k.key)
            except QueryTimeout:
                continue
            u = max(k.u, c.u)
            omega = (e + j) * u * (0.5 if k.id_like and c.id_like else 1.0)
            out.append(EdgeCandidate(edge, round(e, 4), round(j, 4), round(u, 4), round(omega, 4), declared))
    best = {}  # keep one candidate per edge (the higher-scoring direction)
    for x in out:
        if x.edge not in best or x.omega > best[x.edge].omega:
            best[x.edge] = x
    return sorted(best.values(), key=lambda x: -x.omega)


def cell_index(con, schema: SchemaGraph, stats: dict[str, ColumnStats]) -> dict[str, list]:
    out = {}
    for key, s in stats.items():
        if s.id_like or s.n_nonnull == 0 or s.avg_chars > MAX_VALUE_CHARS:
            continue
        t = key.split(".", 1)[0]
        col = column_sql(schema, key)
        try:
            rows = run(con, f"SELECT {col}, COUNT(*) AS n FROM {table_sql(schema, t)} WHERE {col} IS NOT NULL "
                            f"GROUP BY {col} ORDER BY n DESC LIMIT {VALUES_PER_COLUMN}", STAT_TIMEOUT_S)
        except QueryTimeout:
            continue
        out[key] = [v for v, _ in rows if not (isinstance(v, str) and len(v) > MAX_VALUE_CHARS)]
    return out


def build_db_index(db_ref: str, db_file: Path, schema: SchemaGraph) -> DbIndex:
    con = connect_ro(db_file)
    try:
        stats = column_stats(con, schema)
        return DbIndex(db_ref, stats, infer_edges(con, schema, stats), cell_index(con, schema, stats))
    finally:
        con.close()


# ---------------------------------------------------------------- cell search
class CellSearch:
    """Token inverted index over a DbIndex's cell values, for question -> (column, value) lookup."""

    def __init__(self, index: DbIndex):
        self.values: list[tuple[str, object, str]] = []  # (column key, raw value, normalised)
        self.by_token: dict[str, list[int]] = defaultdict(list)
        self.by_prefix: dict[str, list[int]] = defaultdict(list)  # first 5 chars: "australia(n)"
        for key, vals in index.cells.items():
            for v in vals:
                if isinstance(v, (int, float)):
                    continue  # numbers are offered as numeric anchors from the question instead
                nv = norm_value(v)
                i = len(self.values)
                self.values.append((key, v, nv))
                for tok in set(tokens(nv)):
                    self.by_token[tok].append(i)
                    if len(tok) >= 5:
                        self.by_prefix[tok[:5]].append(i)

    def lookup(self, text: str, tables: set[str] | None = None, k: int = 8) -> list[tuple[str, object, float]]:
        """Top-k (column, raw value, score). A value spelled out in full in the text scores 1+."""
        q = norm_value(text)
        q_plain = f" {plain(text)} "
        q_tokens = set(tokens(q))
        scored = {}
        candidates = {i for tok in q_tokens for i in self.by_token.get(tok, ())}
        candidates |= {i for tok in q_tokens if len(tok) >= 5 for i in self.by_prefix.get(tok[:5], ())}
        q_prefixes = {t[:5] for t in q_tokens if len(t) >= 5}
        for i in candidates:
            key, raw, nv = self.values[i]
            if tables is not None and key.split(".", 1)[0] not in tables:
                continue
            vt = set(tokens(nv))
            fuzzy = {t for t in vt - q_tokens if len(t) >= 5 and t[:5] in q_prefixes}  # "australian" ~ "australia"
            overlap = (len(vt & q_tokens) + 0.6 * len(fuzzy)) / len(vt)
            exact = f" {plain(nv)} " in q_plain or (len(nv) > 3 and nv in q)
            scored[i] = overlap + (1.0 if exact else 0.0) + 0.01 * min(len(nv), 30) / 30
        top = sorted(scored, key=lambda i: -scored[i])[:k]
        return [(self.values[i][0], self.values[i][1], round(scored[i], 4)) for i in top]


# ---------------------------------------------------------------- CLI
def build_family(data_dir: Path, family: str, out: Path, workers: int = 1) -> dict:
    """Index every database referenced by a family's instances (local data dir layout)."""
    from concurrent.futures import ProcessPoolExecutor

    sources = {"spider_mmqa": ("spider", "mmqa"), "bird": ("bird",)}[family]
    jobs = []
    for src in sources:
        schemas = json.loads((data_dir / "schemas" / f"{src}.json").read_text(encoding="utf-8"))
        for ref, sd in schemas.items():
            if not (out / "index" / f"{ref}.json").exists():
                jobs.append((ref, data_dir / "db" / f"{ref}.sqlite", SchemaGraph.from_dict(sd)))
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max(1, workers)) as pool:
        for idx in pool.map(build_db_index, *zip(*jobs)) if jobs else []:
            path = out / "index" / f"{idx.db_ref}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(idx.to_dict(), ensure_ascii=False, default=str), encoding="utf-8")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} databases indexed ({time.time() - t0:.0f}s)", flush=True)
    return {"family": family, "indexed": done}


def main(argv: list[str] | None = None) -> int:
    from mhtable import storage

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=["spider_mmqa", "bird"], required=True)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="local copy of the data repo (schemas/, db/); default: fetch from HF at --revision")
    p.add_argument("--revision", default=None, help="data repo revision to fetch, e.g. data-v2")
    p.add_argument("--out", type=Path, default=Path("data/index_build"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--push", action="store_true")
    args = p.parse_args(argv)
    data_dir = args.data_dir
    if data_dir is None:
        sources = {"spider_mmqa": ("spider", "mmqa"), "bird": ("bird",)}[args.family]
        for pat in ["schemas/*"] + [f"db/{s}/*" for s in sources]:
            data_dir = storage.fetch("data", pat, revision=args.revision)
    print(build_family(data_dir, args.family, args.out, args.workers))
    if args.push:
        storage.push_large("data", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
