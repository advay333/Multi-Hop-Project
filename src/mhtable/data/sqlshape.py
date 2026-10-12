"""
Structural analysis of a gold SQL query against the MDP's action set (report Part C.3).

A query fits when it is >= 1 equi-join over distinct tables, has no subquery or set operation,
its WHERE is an AND of `column op literal` filters, and max(#filters, 1) + #joins <= H - 1.
Everything above the joins (SELECT / aggregates / GROUP BY / ORDER BY / LIMIT) is the reader's
job and does not affect fit.

  strict  - exactly the report's action set: ops {=, <, >}, <= 2 filters, declared-FK joins
  relaxed - also LIKE / IN / BETWEEN / <= / >= / != / IS NULL, 3 filters, undeclared equi-joins

The P2 oracle builds on this to extract gold edges/anchors; until then the data build uses it
for tier flags and the question category.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from mhtable.data.schema import SchemaGraph, make_edge

STRICT_OPS = {exp.EQ: "=", exp.LT: "<", exp.GT: ">"}
RELAXED_CMP = {exp.LTE: "<=", exp.GTE: ">=", exp.NEQ: "!="}


# ---------------------------------------------------------------- SQL analysis
def _unwrap(e):
    while isinstance(e, exp.Paren):
        e = e.this
    return e


def _conjuncts(e):
    e = _unwrap(e)
    if isinstance(e, exp.And):
        return _conjuncts(e.left) + _conjuncts(e.right)
    return [e]


def analyse_sql(sql: str, schema: SchemaGraph | None, horizon: int = 5):
    """Structural features of a gold query plus why it falls outside each action tier.

    Returns (features, ast). `features["reasons"]` lists everything that breaks the strict tier,
    `features["relaxed_reasons"]` only what breaks the relaxed tier; empty means it fits.
    """
    out = {"reasons": [], "relaxed_reasons": []}
    try:
        ast = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        out["reasons"] = out["relaxed_reasons"] = ["parse_error"]
        return out, None

    hard = []  # breaks both tiers

    if isinstance(ast, (exp.Union, exp.Intersect, exp.Except)):
        hard.append("set_op")
        out["reasons"] = out["relaxed_reasons"] = hard
        return out, ast
    if not isinstance(ast, exp.Select):
        out["reasons"] = out["relaxed_reasons"] = ["not_select"]
        return out, ast
    if any(s is not ast for s in ast.find_all(exp.Select)):
        hard.append("subquery")

    # --- tables and aliases
    alias2tab, tab_list = {}, []
    for t in ast.find_all(exp.Table):
        if t.find_ancestor(exp.Subquery):
            continue
        name = t.name.lower()
        tab_list.append(name)
        alias2tab[(t.alias or t.name).lower()] = name
        alias2tab[name] = name
    out["n_tables"] = len(tab_list)
    if len(set(tab_list)) < len(tab_list):
        hard.append("self_join")

    def resolve(col: exp.Column):
        """(table, col) or None if it doesn't resolve (e.g. a double-quoted string literal)."""
        c = col.name.lower()
        if col.table:
            t = alias2tab.get(col.table.lower())
            return (t, c) if t else None
        if schema is None:
            return (None, c)
        cands = [t for t in set(tab_list) if c in schema.column_set(t)]
        return (cands[0], c) if len(cands) == 1 else None

    def is_literal(e):
        e = _unwrap(e)
        if isinstance(e, exp.Neg):
            e = e.this
        if isinstance(e, (exp.Literal, exp.Null, exp.Boolean)):
            return True
        # Spider writes strings as "Alabama", which sqlite parses as an identifier
        return (isinstance(e, exp.Column) and not e.table and e.this.args.get("quoted")
                and resolve(e) is None)

    def is_col(e):
        e = _unwrap(e)
        return isinstance(e, exp.Column) and not is_literal(e)

    # --- join edges
    edges = []
    joins = ast.args.get("joins") or []
    for j in joins:
        on = j.args.get("on")
        if on is None:
            if j.args.get("using"):
                hard.append("join_using")
            else:
                out.setdefault("_where_joins_needed", True)
            continue
        pairs = []
        for c in _conjuncts(on):
            if isinstance(c, exp.EQ) and is_col(c.left) and is_col(c.right):
                pairs.append((resolve(_unwrap(c.left)), resolve(_unwrap(c.right))))
            else:
                hard.append("non_equi_join")
        if pairs:
            edges.append(pairs)

    def literal_value(e):
        """Python value of a literal node: str / int / float / None."""
        e = _unwrap(e)
        if isinstance(e, exp.Neg):
            v = literal_value(e.this)
            return -v if isinstance(v, (int, float)) else v
        if isinstance(e, exp.Null):
            return None
        if isinstance(e, exp.Boolean):
            return 1 if e.this else 0
        if isinstance(e, exp.Column):  # Spider's double-quoted string literal
            return e.name
        if e.is_string:
            return e.this
        try:
            return int(e.this)
        except ValueError:
            return float(e.this)

    def add_anchor(col, op, value_nodes, relaxed_reason=None):
        """Record `col op literal(s)`; `relaxed_reason` marks ops outside the strict tier."""
        if not all(is_literal(v) for v in value_nodes):
            hard.append("func_or_expr_predicate")
            return
        anchors.append((resolve(_unwrap(col)), op, tuple(literal_value(v) for v in value_nodes)))
        if relaxed_reason:
            relaxed_only.append(relaxed_reason)

    def add_special(node, negated):
        """LIKE / IN / BETWEEN / IS anchors (relaxed tier only)."""
        name = type(node).__name__.lower()
        op = ("not_" if negated else "") + name
        if isinstance(node, exp.In):
            if node.args.get("query") is not None:
                hard.append("subquery")
                return
            values = node.expressions
        elif isinstance(node, exp.Between):
            values = [node.args["low"], node.args["high"]]
        elif isinstance(node, exp.Is):
            values = [] if isinstance(_unwrap(node.expression), exp.Null) else [node.expression]
        else:  # Like / ILike
            values = [node.expression]
        add_anchor(node.this, op, values, relaxed_reason=op)

    # --- WHERE predicates (column-column equalities across tables count as joins)
    anchors, relaxed_only = [], []
    where = ast.args.get("where")
    if where is not None:
        for c in _conjuncts(where.this):
            if isinstance(c, exp.Or):
                hard.append("or_predicate")
                continue
            if type(c) in STRICT_OPS or type(c) in RELAXED_CMP:
                l, r = _unwrap(c.left), _unwrap(c.right)
                if is_col(l) and is_col(r):
                    if type(c) is exp.EQ:
                        edges.append([(resolve(l), resolve(r))])
                    else:
                        hard.append("col_col_compare")
                    continue
                op = STRICT_OPS.get(type(c)) or RELAXED_CMP[type(c)]
                if is_col(r) and is_literal(l):  # 5 < col  ->  col > 5
                    l, r = r, l
                    op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(op, op)
                if is_col(l) and is_literal(r):
                    add_anchor(l, op, [r], relaxed_reason=("op_" + op) if type(c) in RELAXED_CMP else None)
                else:
                    hard.append("func_or_expr_predicate")
            elif isinstance(c, (exp.Like, exp.ILike, exp.In, exp.Between, exp.Is)) and is_col(c.this):
                add_special(c, negated=False)
            elif (isinstance(c, exp.Not) and isinstance(_unwrap(c.this), (exp.Like, exp.In, exp.Is))
                  and is_col(_unwrap(c.this).this)):
                add_special(_unwrap(c.this), negated=True)
            else:
                hard.append("func_or_expr_predicate")
    if ast.args.get("having"):
        out["has_having"] = True

    out["n_joins"] = len(edges)
    out["n_anchors"] = len(anchors)
    out["anchors"] = anchors  # [((table, col), op, (literal values...)), ...]
    out["edges"] = edges  # [[((t1, c1), (t2, c2)), ...], ...] one list of column pairs per join
    out["tables"] = tab_list  # in FROM/JOIN order

    if len(edges) == 0:
        hard.append("no_join")
    if len(edges) != len(set(tab_list)) - 1 and "self_join" not in hard and edges:
        hard.append("join_graph_not_tree")
    # One ON clause linking the new table to two earlier ones closes a cycle: not one join step.
    if any(len({c[0] for p in e for c in p if c}) != 2 for e in edges):
        hard.append("join_graph_not_tree")
    if any(None in p for e in edges for p in e) or any(a[0] is None for a in anchors):
        hard.append("unresolved_column")

    # --- FK check (strict only)
    strict_extra = []
    if schema is not None and edges and "unresolved_column" not in hard:
        for e in edges:
            if not all(make_edge(f"{a[0]}.{a[1]}", f"{b[0]}.{b[1]}") in schema.fks for a, b in e):
                strict_extra.append("join_not_declared_fk")
                break

    steps = max(len(anchors), 1) + len(edges)
    if len(anchors) > 3:
        hard.append("too_many_anchors")
    elif len(anchors) == 3:
        strict_extra.append("three_anchors")
    if steps > horizon - 1:
        hard.append("horizon_exceeded")

    # --- reader-side features (do not affect fit, but matter for the reader)
    sel = ast.expressions
    out["agg"] = sorted({type(a).__name__ for s in sel for a in s.find_all(exp.AggFunc)})
    out["arith_projection"] = any(s.find(exp.Div, exp.Mul, exp.Sub, exp.Add, exp.Case, exp.If)
                                  for s in sel)
    out["order_limit"] = bool(ast.args.get("order") and ast.args.get("limit"))
    out["category"] = ("Count" if "Count" in out["agg"] else
                       "Numerical" if out["agg"] or out["arith_projection"] else
                       "List" if len(sel) > 1 or out["order_limit"] else "Select")

    out["relaxed_reasons"] = sorted(set(hard))
    out["reasons"] = sorted(set(hard + strict_extra + relaxed_only))
    return out, ast
