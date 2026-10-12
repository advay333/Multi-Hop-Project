"""
Render MDP state as SQL. Shared by the oracle (replaying gold plans) and the environment
(executing the policy's actions), so both define the intermediate relation R_t identically:

    R_t = σ_{anchors}( visited[0] ⋈_{joins[0]} visited[1] ⋈ ... )

Tables and columns are given as lowercase keys ("head", "head.born_state") and written with
their original spellings from the SchemaGraph, double-quoted. Self-joins are out of scope, so
table names double as aliases.
"""

from __future__ import annotations

from mhtable.data.schema import SchemaGraph
from mhtable.types import Anchor, Edge

COMPARISONS = {"=", "<", ">", "<=", ">=", "!="}


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def literal(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def table_sql(schema: SchemaGraph, table: str) -> str:
    return quote(schema.table_names[table.lower()])


def column_sql(schema: SchemaGraph, key: str) -> str:
    """'head.born_state' -> '"head"."born_state"' with original spellings."""
    t, c = key.split(".", 1)
    t = t.lower()
    original = next((x for x in schema.columns[t] if x.lower() == c.lower()), c)
    return f"{table_sql(schema, t)}.{quote(original)}"


def anchor_sql(schema: SchemaGraph, a: Anchor) -> str:
    col = column_sql(schema, a.column)
    op, vals = a.op, a.values
    if op in COMPARISONS:
        return f"{col} {op} {literal(vals[0])}"
    if op in ("like", "ilike", "not_like", "not_ilike"):
        return f"{col} {'NOT ' if op.startswith('not_') else ''}LIKE {literal(vals[0])}"
    if op in ("in", "not_in"):
        return f"{col} {'NOT ' if op == 'not_in' else ''}IN ({', '.join(literal(v) for v in vals)})"
    if op == "between":
        return f"{col} BETWEEN {literal(vals[0])} AND {literal(vals[1])}"
    if op in ("is", "not_is"):
        target = literal(vals[0]) if vals else "NULL"
        return f"{col} IS {'NOT ' if op == 'not_is' else ''}{target}"
    raise ValueError(f"unsupported anchor op {op!r}")


def edge_on(schema: SchemaGraph, edge: Edge) -> str:
    return " AND ".join(f"{column_sql(schema, a)} = {column_sql(schema, b)}" for a, b in edge)


def from_where(schema: SchemaGraph, tables: list[str], joins: list[Edge], anchors: list[Anchor]) -> str:
    """`FROM ... JOIN ... ON ... WHERE ...` for a join tree.

    `joins[i]` must connect `tables[i + 1]` to one of `tables[: i + 1]` (visit order).
    """
    if len(joins) != len(tables) - 1:
        raise ValueError(f"{len(tables)} tables need {len(tables) - 1} joins, got {len(joins)}")
    sql = f"FROM {table_sql(schema, tables[0])}"
    for t, e in zip(tables[1:], joins):
        sql += f" JOIN {table_sql(schema, t)} ON {edge_on(schema, e)}"
    if anchors:
        sql += " WHERE " + " AND ".join(anchor_sql(schema, a) for a in anchors)
    return sql


def rowid_select(schema: SchemaGraph, tables: list[str]) -> str:
    """Select one rowid per table, in sorted table order (identifies each row of R_t)."""
    return ", ".join(f"{table_sql(schema, t)}.rowid" for t in sorted(tables))
