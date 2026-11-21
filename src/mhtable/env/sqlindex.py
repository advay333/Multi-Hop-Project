"""
SQLite indexes for the large tables of a database family (PLAN §4a, Track B).

    python -m mhtable.env.sqlindex --data-dir <data copy> --family bird --workers 8 [--dry-run]

Why: the environment's look-ahead runs a capped COUNT(*) for every candidate action. A JOIN on
a column without an index makes SQLite build a temporary ("automatic") index on the joined
table inside every single query, and an equality filter on an unindexed column scans the whole
table. On BIRD's largest databases (millions of rows) each look-ahead then takes seconds.
A permanent index is built once and turns both into lookups.

An index is a sorted copy of one or more columns with pointers back to the rows, like the
index at the back of a book. It changes no data and no query result, only speed (and the
order SQLite returns rows when a query has no ORDER BY).

What gets indexed, only in tables with at least `--min-rows` rows (small tables are fast anyway):
  - join columns: both sides of every candidate join edge in the env index (`DbIndex.edges`)
    with omega >= `--edge-min` or declared as a foreign key; composite edges get one index
    over all their columns on that table
  - filter columns: columns that can carry an anchor — those in the cell index (categorical
    values) and numeric columns — unless their values are long text (avg > `--max-chars`)
Columns already led by an existing index, or the INTEGER PRIMARY KEY (rowid), are skipped.

Files are changed in place, so `--data-dir` must hold real files (e.g. from
`storage download`), never HuggingFace cache links. Databases with nothing to index are not
opened for writing, so they stay byte-identical. Re-running is safe (CREATE INDEX IF NOT EXISTS).
Writes a manifest `db_index_manifest.json` into --data-dir: per DB what was created in this run,
sizes, seconds, and `mhx_indexes` = our indexes present in the file. `changed` lists every DB
holding at least one of our indexes, so a re-run after an interruption still lists the DBs indexed
before it (they are what must be uploaded). `--dry-run --plan-out` estimates the added bytes.
`--analyze` also runs ANALYZE on the indexed tables (planner statistics; off by default).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mhtable.data.schema import SchemaGraph
from mhtable.env.index import DbIndex

FAMILIES = {"spider_mmqa": ("spider", "mmqa"), "bird": ("bird",)}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def plan_indexes(idx: DbIndex, schema: SchemaGraph, min_rows: int = 50_000, edge_min: float = 0.5,
                 max_chars: float = 64.0, anchors: bool = True) -> list[tuple[str, tuple[str, ...]]]:
    """(table key, column keys) to index, in a stable order; gold-free (env index + schema only)."""
    big = {k.split(".", 1)[0] for k, c in idx.columns.items() if c.n_rows >= min_rows}
    want: dict[tuple[str, tuple[str, ...]], None] = {}
    for e in idx.edges:
        if e.omega < edge_min and not e.declared:
            continue
        per_table: dict[str, list[str]] = {}
        for pair in e.edge:
            for c in pair:
                per_table.setdefault(c.split(".", 1)[0], []).append(c)
        for t, cols in per_table.items():
            if t in big:
                want[(t, tuple(dict.fromkeys(cols)))] = None
    if anchors:
        for k, c in idx.columns.items():
            t = k.split(".", 1)[0]
            if t in big and (k in idx.cells or c.numeric) and c.avg_chars <= max_chars:
                want[(t, (k,))] = None
    return [x for x in want if x[0] in schema.table_names]


def _existing_leads(con: sqlite3.Connection, table: str) -> set[tuple[str, ...]]:
    """Column tuples (lowercase) already usable as an index prefix, incl. the rowid alias."""
    leads = set()
    info = con.execute(f"PRAGMA table_info({_q(table)})").fetchall()
    pk = [r for r in info if r[5]]
    if len(pk) == 1 and (pk[0][2] or "").upper() == "INTEGER":
        leads.add((pk[0][1].lower(),))
    for row in con.execute(f"PRAGMA index_list({_q(table)})").fetchall():
        cols = [r[2] for r in con.execute(f"PRAGMA index_info({_q(row[1])})").fetchall()]
        if None in cols:  # expression index
            continue
        cols = tuple(c.lower() for c in cols)
        for i in range(1, len(cols) + 1):
            leads.add(cols[:i])
    return leads


def index_name(table: str, cols: tuple[str, ...]) -> str:
    h = hashlib.sha1(("|".join((table,) + cols)).encode()).hexdigest()[:12]
    return f"mhx_{h}"


def our_indexes(db_file: Path) -> int:
    con = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
    try:
        return con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name LIKE 'mhx!_%' ESCAPE '!'").fetchone()[0]
    finally:
        con.close()


def estimate_bytes(idx: DbIndex, table: str, cols: tuple[str, ...]) -> int:
    """Rough on-disk size of an index: rows x (key bytes + rowid/record overhead) x b-tree slack."""
    rows = max((c.n_rows for k, c in idx.columns.items() if k.split(".", 1)[0] == table), default=0)
    key = sum((idx.columns[c].avg_chars if c in idx.columns else 8.0) + 2 for c in cols)
    return int(rows * (key + 10) * 1.3)


def index_db(db_file: Path, idx: DbIndex, schema: SchemaGraph, dry_run: bool = False, analyze: bool = False,
             **plan_kw) -> dict:
    """Create the planned indexes in `db_file`. Returns a report entry."""
    plan = plan_indexes(idx, schema, **plan_kw)
    rep = {"db_ref": idx.db_ref, "bytes_before": db_file.stat().st_size, "planned": len(plan),
           "created": [], "skipped_existing": 0, "est_added_bytes": 0, "seconds": 0.0}
    if not plan:
        rep["bytes_after"] = rep["bytes_before"]
        rep["mhx_indexes"] = our_indexes(db_file)
        return rep
    if db_file.is_symlink() or "blobs" in db_file.resolve().parts:
        raise RuntimeError(f"{db_file} is a HuggingFace cache link; download real files first")
    t0 = time.time()
    ro = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
    todo = []
    leads_by_table: dict[str, set] = {}
    for t, cols in plan:
        tname = schema.table_names[t]
        if t not in leads_by_table:
            leads_by_table[t] = _existing_leads(ro, tname)
        names = tuple(c.split(".", 1)[1].lower() for c in cols)
        if names in leads_by_table[t]:
            rep["skipped_existing"] += 1
            continue
        originals = []
        for n in names:
            originals.append(next((c for c in schema.columns[t] if c.lower() == n), n))
        todo.append((t, tname, cols, originals))
        rep["est_added_bytes"] += estimate_bytes(idx, t, cols)
        leads_by_table[t] |= {names[:i] for i in range(1, len(names) + 1)}
    ro.close()
    if (todo or (analyze and our_indexes(db_file))) and not dry_run:
        con = sqlite3.connect(str(db_file))
        con.execute("PRAGMA cache_size=-1000000")  # ~1 GB page cache for the index sorts
        for t, tname, cols, originals in todo:
            t1 = time.time()
            con.execute(f"CREATE INDEX IF NOT EXISTS {_q(index_name(t, cols))} ON {_q(tname)} "
                        f"({', '.join(_q(c) for c in originals)})")
            con.commit()
            rep["created"].append({"table": t, "columns": list(cols), "seconds": round(time.time() - t1, 1)})
        if analyze:  # every table carrying one of our indexes, also those indexed by an earlier run
            t1 = time.time()
            tables = [r[0] for r in con.execute(
                "SELECT DISTINCT tbl_name FROM sqlite_master WHERE type = 'index' AND name LIKE 'mhx!_%' ESCAPE '!'")]
            for tname in tables:
                con.execute(f"ANALYZE {_q(tname)}")
            con.commit()
            rep["analyzed"] = {"tables": len(tables), "seconds": round(time.time() - t1, 1)}
        con.close()
    elif todo:
        rep["created"] = [{"table": t, "columns": list(cols), "seconds": 0.0} for t, _, cols, _ in todo]
    rep["seconds"] = round(time.time() - t0, 1)
    rep["bytes_after"] = db_file.stat().st_size
    rep["mhx_indexes"] = our_indexes(db_file)
    return rep


def _job(args) -> dict:
    db_file, idx_d, schema_d, dry_run, analyze, plan_kw = args
    return index_db(Path(db_file), DbIndex.from_dict(idx_d), SchemaGraph.from_dict(schema_d), dry_run, analyze,
                    **plan_kw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="holds db/, schemas/ (real files)")
    p.add_argument("--index-dir", type=Path, default=None, help="holds index/ (default: --data-dir)")
    p.add_argument("--family", choices=sorted(FAMILIES), default="bird")
    p.add_argument("--dbs", nargs="*", default=None, help="only these db_refs (e.g. bird/talkingdata)")
    p.add_argument("--min-rows", type=int, default=50_000)
    p.add_argument("--edge-min", type=float, default=0.5)
    p.add_argument("--max-chars", type=float, default=64.0)
    p.add_argument("--no-anchors", action="store_true", help="index join columns only")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--dry-run", action="store_true", help="report the plan, change nothing")
    p.add_argument("--plan-out", type=Path, default=None, help="write the report JSON here (also for --dry-run)")
    p.add_argument("--analyze", action="store_true", help="run ANALYZE on indexed tables (planner statistics)")
    args = p.parse_args(argv)
    index_dir = args.index_dir or args.data_dir
    plan_kw = dict(min_rows=args.min_rows, edge_min=args.edge_min, max_chars=args.max_chars,
                   anchors=not args.no_anchors)

    jobs = []
    for src in FAMILIES[args.family]:
        sf = args.data_dir / "schemas" / f"{src}.json"
        if not sf.exists():
            print(f"[skip] {src}: no {sf}", file=sys.stderr)
            continue
        schemas = json.loads(sf.read_text(encoding="utf-8"))
        for ref, sd in sorted(schemas.items()):
            if args.dbs and ref not in args.dbs:
                continue
            db_file = args.data_dir / "db" / f"{ref}.sqlite"
            idx_file = index_dir / "index" / f"{ref}.json"
            if not db_file.exists() or not idx_file.exists():
                print(f"[skip] {ref}: no local db or index", file=sys.stderr)
                continue
            jobs.append((str(db_file), json.loads(idx_file.read_text(encoding="utf-8")), sd, args.dry_run,
                         args.analyze, plan_kw))
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)  # biggest first: they take longest

    t0, reports = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, j): j[0] for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            reports.append(r)
            if r["created"]:
                print(f"{r['db_ref']}: {len(r['created'])} indexes, {r['bytes_before'] / 2**30:.2f} -> "
                      f"{r['bytes_after'] / 2**30:.2f} GiB, {r['seconds']:.0f}s ({time.time() - t0:.0f}s total)",
                      file=sys.stderr, flush=True)
    reports.sort(key=lambda r: r["db_ref"])
    changed = [r["db_ref"] for r in reports if r["mhx_indexes"] > 0]
    out = {"family": args.family, "params": plan_kw, "dry_run": args.dry_run, "analyze": args.analyze,
           "databases": len(reports), "changed": changed,
           "est_added_bytes": sum(r["est_added_bytes"] for r in reports),
           "would_change": [r["db_ref"] for r in reports if r["created"]], "bytes_before": sum(r["bytes_before"] for r in reports),
           "bytes_after": sum(r["bytes_after"] for r in reports), "seconds": round(time.time() - t0, 1),
           "per_db": reports}
    if not args.dry_run:
        (args.data_dir / "db_index_manifest.json").write_text(json.dumps(out, indent=2))
    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(out, indent=2))
    summary = {k: v for k, v in out.items() if k != "per_db"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
