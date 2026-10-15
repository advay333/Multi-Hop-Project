"""
Gold-SQL oracle (report C.4–C.5, DESIGN.md §6.3) — the only code that reads gold SQL.

For one question it produces:
  - the gold structure G*: tables, join edges, anchor filters (from `sqlshape`);
  - the canonical action sequence used for behaviour cloning;
  - the intermediate relation after every gold action (entity sets, for per-hop diagnostics);
  - the potential Φ used for reward shaping (Jaccard of structural items).

Canonical order. Start at the anchored table whose own filters leave the fewest rows (a
selectivity-aware plan; ties -> query order); apply its anchors, then join the remaining
tables breadth-first, applying each newly joined table's anchors right after its JOIN (ANCHOR is
only legal on visited tables). Questions without anchors start with SEED on the first FROM
table. Then STOP. BC imitates this one order; RL may find others — Φ is order-invariant.

Checks (failures dropped and counted, never repaired):
  oracle_reconstruct_mismatch  replaying our parsed plan does not give exactly the rows of the
                               gold query's FROM/JOIN/WHERE (i.e. parsing lost a condition)
  join_not_key                 a gold join pairs two non-unique columns (JAR's rule:
                               max(u_k, u_l) = 1), which creates spurious row pairings
  gold_relation_empty          the gold FROM/JOIN/WHERE selects no rows (e.g. Spider's empty
                               `scholar`/`yelp` tables): nothing to select, and a COUNT answer
                               of 0 needs no evidence
  oracle_timeout / oracle_error / oracle_no_rowid
Edge-candidate and anchor-candidate recall (report checks 3–4) need the environment's
candidate sets and are measured in P3 (gate G-6).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field

from sqlglot import exp

from mhtable.data.execute import QueryTimeout, run
from mhtable.data.schema import SchemaGraph, make_edge
from mhtable.data.sqlshape import analyse_sql
from mhtable.sqlrender import from_where, rowid_select
from mhtable.types import Action, Anchor, Edge

ROWID_CAP = 2000  # rowids stored per entity set; beyond this only the count is exact
RECONSTRUCT_CAP = 20000  # row identities compared exactly up to this size, counts beyond


@dataclass
class ParsedGold:
    tables: list[str]  # distinct tables, FROM/JOIN order
    edges: list[Edge]  # canonical (make_edge-ordered pairs), one per join
    anchors: list[Anchor]
    monotone: bool
    ast: exp.Expression = field(repr=False)


@dataclass
class OracleResult:
    trajectory: list[Action] = field(default_factory=list)
    entity_sets: list[dict] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- parsing
def edge_from_pairs(pairs) -> Edge:
    """[((t1, c1), (t2, c2)), ...] -> canonical Edge (composite keys keep several pairs)."""
    return tuple(sorted(make_edge(f"{a[0]}.{a[1]}", f"{b[0]}.{b[1]}")[0] for a, b in pairs))


def parse(gold_sql: str, schema: SchemaGraph, horizon: int = 5) -> tuple[ParsedGold | None, dict]:
    """Gold structure, or None when the query is outside the relaxed tier."""
    feats, ast = analyse_sql(gold_sql, schema, horizon)
    if feats["relaxed_reasons"]:
        return None, feats
    tables = list(dict.fromkeys(feats["tables"]))
    edges = [edge_from_pairs(e) for e in feats["edges"]]
    anchors = [Anchor(f"{t}.{c}", op, vals) for (t, c), op, vals in feats["anchors"]]
    monotone = not (feats["agg"] or feats["order_limit"] or feats.get("has_having")
                    or any(a.op.startswith("not_") or a.op == "!=" for a in anchors))
    return ParsedGold(tables, edges, anchors, monotone, ast), feats


# ---------------------------------------------------------------- structural potential
def edge_key(edge: Edge) -> tuple:
    return tuple(sorted(tuple(sorted(p)) for p in edge))


def items(edges: list[Edge], anchors: list[Anchor]) -> frozenset:
    """Structural items taken: join edges (unordered) and anchors (normalised)."""
    return frozenset([("edge", edge_key(e)) for e in edges] + [("anchor", a.key()) for a in anchors])


def potential(taken: frozenset, gold: frozenset) -> float:
    """Φ = Jaccard(taken, gold) (report C.4). The caller sets Φ(terminal) = 0."""
    union = taken | gold
    return len(taken & gold) / len(union) if union else 0.0


# ---------------------------------------------------------------- canonical plan
def _tables_of(edge: Edge) -> set[str]:
    return {c.split(".", 1)[0] for p in edge for c in p}


def canonical_actions(parsed: ParsedGold, start: str) -> list[Action]:
    by_table = defaultdict(list)
    for a in parsed.anchors:
        by_table[a.table].append(a)
    adj = defaultdict(list)  # table -> [(edge, other)] in query order
    for e in parsed.edges:
        t1, t2 = sorted(_tables_of(e))
        adj[t1].append((e, t2))
        adj[t2].append((e, t1))

    actions = [Action("ANCHOR", start, anchor=a) for a in by_table[start]] or [Action("SEED", start)]
    seen, queue = {start}, deque([start])
    while queue:
        t = queue.popleft()
        for e, other in adj[t]:
            if other in seen:
                continue
            seen.add(other)
            queue.append(other)
            actions.append(Action("JOIN", other, edge=e))
            actions += [Action("ANCHOR", other, anchor=a) for a in by_table[other]]
    return actions + [Action.stop()]


def replay(actions: list[Action]) -> list[tuple[list[str], list[Edge], list[Anchor]]]:
    """State (visited tables, joins, anchors) after each non-STOP action."""
    visited, joins, anchors, states = [], [], [], []
    for a in actions:
        if a.kind == "STOP":
            break
        if a.kind in ("SEED", "ANCHOR") and not visited:
            visited.append(a.table)
        if a.kind == "JOIN":
            visited.append(a.table)
            joins.append(a.edge)
        if a.kind == "ANCHOR":
            anchors.append(a.anchor)
        states.append((list(visited), list(joins), list(anchors)))
    return states


def _count(con, schema, tables, joins, anchors, timeout_s) -> int:
    return run(con, f"SELECT COUNT(*) {from_where(schema, tables, joins, anchors)}", timeout_s)[0][0]


def _entity_set(con, schema, tables, joins, anchors, timeout_s) -> dict:
    sql = f"SELECT {rowid_select(schema, tables)} {from_where(schema, tables, joins, anchors)}"
    rows = run(con, sql, timeout_s, ROWID_CAP + 1)
    truncated = len(rows) > ROWID_CAP
    rows = rows[:ROWID_CAP]
    rowids = {t: sorted({r[i] for r in rows}) for i, t in enumerate(sorted(tables))}
    n = _count(con, schema, tables, joins, anchors, timeout_s) if truncated else len(rows)
    return {"rows": n, "truncated": truncated, "rowids": rowids}


# ---------------------------------------------------------------- checks
def _original_rowids(parsed: ParsedGold, timeout_s: float, con) -> tuple[int, set | None]:
    """Row identities of the gold query's own FROM/JOIN/WHERE (aliases resolved)."""
    alias = {}
    for t in parsed.ast.find_all(exp.Table):
        if not t.find_ancestor(exp.Subquery):
            alias[t.name.lower()] = t.alias or t.name
    sel = ", ".join(f'"{alias[t]}".rowid' for t in sorted(parsed.tables))
    frm = parsed.ast.args.get("from") or parsed.ast.args.get("from_")
    body = frm.sql(dialect="sqlite") + "".join(" " + j.sql(dialect="sqlite") for j in parsed.ast.args.get("joins") or [])
    if parsed.ast.args.get("where"):
        body += " " + parsed.ast.args["where"].sql(dialect="sqlite")
    rows = run(con, f"SELECT {sel} {body}", timeout_s, RECONSTRUCT_CAP + 1)
    if len(rows) > RECONSTRUCT_CAP:
        return run(con, f"SELECT COUNT(*) {body}", timeout_s)[0][0], None
    return len(rows), set(rows)


def _unique(con, schema, key: str, cache: dict, timeout_s: float) -> bool:
    if key not in cache:
        from mhtable.sqlrender import column_sql, table_sql

        t = key.split(".", 1)[0]
        col = column_sql(schema, key)
        n_distinct, n = run(con, f"SELECT COUNT(DISTINCT {col}), COUNT({col}) FROM {table_sql(schema, t)}", timeout_s)[0]
        cache[key] = n_distinct == n
    return cache[key]


# ---------------------------------------------------------------- entry point
def run_oracle(parsed: ParsedGold, schema: SchemaGraph, con: sqlite3.Connection, timeout_s: float = 30.0,
               unique_cache: dict | None = None) -> OracleResult:
    res = OracleResult()
    unique_cache = {} if unique_cache is None else unique_cache
    try:
        # Selectivity-aware start: the anchored table whose own anchors leave the fewest rows.
        anchored = list(dict.fromkeys(a.table for a in parsed.anchors))
        if anchored:
            counts = {t: _count(con, schema, [t], [], [a for a in parsed.anchors if a.table == t], timeout_s)
                      for t in anchored}
            start = min(anchored, key=lambda t: (counts[t], anchored.index(t)))
        else:
            start = parsed.tables[0]
        actions = canonical_actions(parsed, start)

        states = replay(actions)
        res.entity_sets = [_entity_set(con, schema, *st, timeout_s) for st in states]
        res.trajectory = [Action(a.kind, a.table, a.anchor, a.edge, lookahead_rows=es["rows"])
                          for a, es in zip(actions, res.entity_sets)] + [actions[-1]]

        # Check: our replayed final relation == the gold query's own FROM/JOIN/WHERE.
        tables, joins, anchors = states[-1]
        n_orig, orig = _original_rowids(parsed, timeout_s, con)
        if orig is None:
            same = n_orig == res.entity_sets[-1]["rows"]
        else:
            mine = set(run(con, f"SELECT {rowid_select(schema, tables)} {from_where(schema, tables, joins, anchors)}",
                           timeout_s))
            same = mine == orig
        if not same:
            res.fail_reasons.append("oracle_reconstruct_mismatch")
        if res.entity_sets[-1]["rows"] == 0:
            res.fail_reasons.append("gold_relation_empty")

        # Check: every gold join has a unique column on at least one side.
        for e in parsed.edges:
            if not all(_unique(con, schema, a, unique_cache, timeout_s) or
                       _unique(con, schema, b, unique_cache, timeout_s) for a, b in e):
                res.fail_reasons.append("join_not_key")
                break
    except QueryTimeout:
        res.fail_reasons.append("oracle_timeout")
    except sqlite3.OperationalError as e:
        res.fail_reasons.append("oracle_no_rowid" if "rowid" in str(e).lower() else "oracle_error")
    except Exception:
        res.fail_reasons.append("oracle_error")
    return res
