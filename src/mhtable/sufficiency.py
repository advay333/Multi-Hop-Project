"""
Sub-table sufficiency (report A.5) — gold-aware, outside the environment.

    path_match(state, gold)          the policy took exactly the gold joins and filters
    sufficient(env, state, mat, gold) the materialised sub-table contains what the answer needs

Sufficiency is decided on rows, using the oracle's stored entity set of the final gold relation
(the row ids, per table, of the rows the gold FROM/JOIN/WHERE keeps):
  - the gold path itself is always sufficient;
  - monotone questions (plain selections): every gold row id of every gold table must still be
    in the policy's relation — extra rows are harmless;
  - non-monotone questions (counts, sums, superlatives, negation): the relation must be over
    the same tables, with the same number of rows and exactly the gold row ids;
  - rows cut by the token budget (`overflow`) or an empty sub-table are insufficient.
Conservative cases, counted as insufficient: the gold entity set was truncated at 2,000 row ids
and the path differs; the policy's relation is too large to compare; a query times out.

Limits (this is a debugging proxy, not a metric of record): row ids are compared per table, not
as joined tuples, and the columns shown to the reader are not checked.
"""

from __future__ import annotations

import sqlite3

from mhtable.data.execute import QueryTimeout, run
from mhtable.oracle import _no_rowid_tables, items
from mhtable.sqlrender import from_where, rowid_select
from mhtable.types import GoldRecord, State

COMPARE_CAP = 20_000
TIMEOUT_S = 10.0


def path_match(state: State, gold: GoldRecord) -> bool:
    """Join-path exact match: same join edges and same filters as the gold query (any order)."""
    if items(state.joins, state.anchors) != items(gold.gold_edges, gold.gold_anchors):
        return False
    return not gold.gold_tables or set(state.visited) == {t.lower() for t in gold.gold_tables}


def sufficient(env, state: State, mat, gold: GoldRecord, _cache: dict | None = None) -> bool:
    if not state.visited or mat.rows_shown == 0 or mat.overflow:
        return False
    if path_match(state, gold):
        return True
    es = gold.gold_entity_sets[-1] if gold.gold_entity_sets else None
    if es is None or es.get("truncated"):
        return False
    gold_tables = sorted(es["rowids"])
    if not set(gold_tables) <= set(state.visited):
        return False
    monotone = gold.monotone is True
    if not monotone and (set(state.visited) != set(gold_tables) or state.n_rows != es["rows"]):
        return False
    try:
        con = env._conn(state.instance.db_ref)
        schema = env.data.schema(state.instance.db_ref)
        no_rowid = _no_rowid_tables(con, _cache if _cache is not None else {})
        sql = (f"SELECT DISTINCT {rowid_select(schema, gold_tables, no_rowid)} "
               f"{from_where(schema, state.visited, state.joins, state.anchors)}")
        rows = run(con, sql, TIMEOUT_S, COMPARE_CAP + 1)
    except (QueryTimeout, sqlite3.Error):
        return False
    if len(rows) > COMPARE_CAP:
        return False
    for i, t in enumerate(gold_tables):
        mine, want = {r[i] for r in rows}, set(es["rowids"][t])
        if not (want <= mine if monotone else want == mine):
            return False
    return True
