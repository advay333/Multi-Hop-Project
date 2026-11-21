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
    if state.n_rows < 0:  # look-ahead timed out: the row-id comparison would time out too
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


def gold_hops(gold: GoldRecord) -> dict[str, int]:
    """Gold table -> hop index along the canonical trajectory: the start table (SEED or first
    ANCHOR) is hop 0, each JOIN adds the next hop."""
    out: dict[str, int] = {}
    for a in gold.gold_trajectory:
        if a.kind in ("SEED", "ANCHOR", "JOIN") and a.table and a.table.lower() not in out:
            out[a.table.lower()] = len(out)
    return out


def entity_overlap(env, state: State, gold: GoldRecord, _cache: dict | None = None) -> list[dict]:
    """Per-hop entity recall / precision vs the gold relation E* (report Part D, diagnostic).

    For each gold table: recall = share of its gold row ids still in the policy's final relation,
    precision = share of the policy's row ids of that table that are gold. A gold table the policy
    never visited has recall 0 and no precision. Empty list when the gold entity set was truncated
    or the comparison is too large / times out (unknown, not zero)."""
    es = gold.gold_entity_sets[-1] if gold.gold_entity_sets else None
    if es is None or es.get("truncated"):
        return []
    hops = gold_hops(gold)
    tables = sorted(es["rowids"])
    if state.n_rows < 0:  # size unknown (look-ahead timed out): comparison treated as unknown
        return []
    seen = [t for t in tables if t in state.visited]
    mine: dict[str, set] = {}
    if seen:
        try:
            con = env._conn(state.instance.db_ref)
            schema = env.data.schema(state.instance.db_ref)
            no_rowid = _no_rowid_tables(con, _cache if _cache is not None else {})
            sql = (f"SELECT DISTINCT {rowid_select(schema, seen, no_rowid)} "
                   f"{from_where(schema, state.visited, state.joins, state.anchors)}")
            rows = run(con, sql, TIMEOUT_S, COMPARE_CAP + 1)
        except (QueryTimeout, sqlite3.Error):
            return []
        if len(rows) > COMPARE_CAP:
            return []
        for i, t in enumerate(seen):
            mine[t] = {r[i] for r in rows}
    out = []
    for t in tables:
        want = set(es["rowids"][t])
        got = mine.get(t)
        hit = len(want & got) if got is not None else 0
        out.append({"table": t, "hop": hops.get(t, len(hops)),
                    "recall": round(hit / len(want), 4) if want else None,
                    "precision": round(hit / len(got), 4) if got else None})
    return out
