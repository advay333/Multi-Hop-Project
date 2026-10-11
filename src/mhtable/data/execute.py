"""
Read-only SQLite execution with a wall-clock timeout, shared by the data build, the oracle and
the environment.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from sqlglot import exp


class QueryTimeout(Exception):
    pass


def connect_ro(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, check_same_thread=False)
    con.text_factory = lambda b: b.decode("utf-8", "replace")  # some Spider/BIRD cells are not valid UTF-8
    return con


def run(con: sqlite3.Connection, sql: str, timeout_s: float = 10.0, max_rows: int | None = None) -> list[tuple]:
    """Execute `sql`; raise QueryTimeout after `timeout_s` seconds. `max_rows` caps the fetch."""
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = con.execute(sql)
        return cur.fetchmany(max_rows) if max_rows else cur.fetchall()
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            raise QueryTimeout(f"query exceeded {timeout_s}s") from None
        raise
    finally:
        con.set_progress_handler(None, 0)


def subtable_sql(ast: exp.Select) -> str:
    """Gold sub-table: the query's FROM/JOIN/WHERE, projected onto every qualified column it uses."""
    cols = {}
    for c in ast.find_all(exp.Column):
        if c.table:
            cols[c.sql(dialect="sqlite")] = None
    proj = ", ".join(cols) or "*"
    frm = ast.args.get("from") or ast.args.get("from_")  # key renamed in sqlglot >= 26
    sql = f"SELECT {proj} {frm.sql(dialect='sqlite')}"
    for j in ast.args.get("joins") or []:
        sql += " " + j.sql(dialect="sqlite")
    if ast.args.get("where"):
        sql += " " + ast.args["where"].sql(dialect="sqlite")
    return sql
