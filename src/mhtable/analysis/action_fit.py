"""
Action-set fit analysis: how many Spider / BIRD / MMQA questions can our MDP represent?

"Fits" is defined against the report's final action set (Part C.3):
  SEED(T) | ANCHOR(T.c op v) with op in {=,<,>}, <=2 anchors | JOIN(FK edge) | STOP, H=5.
The gold SQL fits if it is: >=1 equi-join over distinct tables, no subquery / set-op,
WHERE = AND of <=2 `column op literal` predicates, and max(#anchors,1) + #joins <= H-1.
Everything above the join (SELECT/aggregate/GROUP BY/ORDER BY/LIMIT) is the reader's job.

Tiers:
  strict  - exactly the report's action set, joins must be declared FKs
  relaxed - also LIKE / IN / BETWEEN / <= / >= / != / IS NULL, up to 3 anchors,
            and undeclared equi-joins (the report infers edges via JAR anyway)
Plus, when a DB is available, execution checks: gold SQL runs and is non-empty, and the
gold sub-table (FROM/JOIN/WHERE, projected onto referenced columns) fits B tokens.

Usage:
    python -m mhtable.analysis.action_fit spider
    python -m mhtable.analysis.action_fit bird
    python -m mhtable.analysis.action_fit mmqa
"""

import collections
import re
import json
import sqlite3
import sys
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[3]
H = 5
BUDGET_TOKENS = 4096
CHARS_PER_TOKEN = 3.5
EXEC_TIMEOUT_S = 10
MAX_SUBTABLE_ROWS = 5000  # beyond this the sub-table is far over budget anyway

STRICT_OPS = {exp.EQ: "=", exp.LT: "<", exp.GT: ">"}
RELAXED_CMP = {exp.LTE: "<=", exp.GTE: ">=", exp.NEQ: "!="}


# ---------------------------------------------------------------- schema
class Schema:
    """Table -> columns and FK edges from a Spider/BIRD tables.json entry (lowercased)."""

    def __init__(self, entry: dict):
        tabs = [t.lower() for t in entry["table_names_original"]]
        self.cols = collections.defaultdict(set)
        names = []
        for ti, c in entry["column_names_original"]:
            names.append((tabs[ti] if ti >= 0 else None, c.lower()))
            if ti >= 0:
                self.cols[tabs[ti]].add(c.lower())
        self.fks = set()
        for a, b in entry["foreign_keys"]:
            self.fks.add(frozenset([names[a], names[b]]))
        self.tables = set(tabs)

    def tables_with_col(self, col):
        return [t for t, cs in self.cols.items() if col in cs]


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


def analyse_sql(sql: str, schema: Schema | None):
    """Return dict with structural features and the list of reasons it fails each tier."""
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
        cands = [t for t in set(tab_list) if c in schema.cols.get(t, ())]
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
                if is_col(r) and is_literal(l):
                    l, r = r, l
                if is_col(l) and is_literal(r):
                    anchors.append((resolve(l), STRICT_OPS.get(type(c)) or RELAXED_CMP[type(c)]))
                    if type(c) in RELAXED_CMP:
                        relaxed_only.append("op_" + RELAXED_CMP[type(c)])
                else:
                    hard.append("func_or_expr_predicate")
            elif isinstance(c, (exp.Like, exp.ILike, exp.In, exp.Between, exp.Is)):
                if is_col(c.this):
                    anchors.append((resolve(_unwrap(c.this)), type(c).__name__.lower()))
                    relaxed_only.append(type(c).__name__.lower())
                else:
                    hard.append("func_or_expr_predicate")
            elif isinstance(c, exp.Not):
                inner = _unwrap(c.this)
                if isinstance(inner, (exp.Like, exp.In, exp.Is)) and is_col(inner.this):
                    anchors.append((resolve(_unwrap(inner.this)), "not_" + type(inner).__name__.lower()))
                    relaxed_only.append("not_" + type(inner).__name__.lower())
                else:
                    hard.append("func_or_expr_predicate")
            else:
                hard.append("func_or_expr_predicate")
    if ast.args.get("having"):
        out["has_having"] = True

    out["n_joins"] = len(edges)
    out["n_anchors"] = len(anchors)
    out["anchors"] = anchors
    out["edges"] = edges

    if len(edges) == 0:
        hard.append("no_join")
    if len(edges) != len(set(tab_list)) - 1 and "self_join" not in hard and edges:
        hard.append("join_graph_not_tree")
    if any(None in p for e in edges for p in e) or any(a[0] is None for a in anchors):
        hard.append("unresolved_column")

    # --- FK check (strict only)
    strict_extra = []
    if schema is not None and edges and "unresolved_column" not in hard:
        for e in edges:
            if not all(frozenset(p) in schema.fks for p in e):
                strict_extra.append("join_not_declared_fk")
                break

    steps = max(len(anchors), 1) + len(edges)
    if len(anchors) > 3:
        hard.append("too_many_anchors")
    elif len(anchors) == 3:
        strict_extra.append("three_anchors")
    if steps > H - 1:
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


# ---------------------------------------------------------------- execution checks
def _connect(db_path: Path):
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    return con


def _run(con, sql, max_rows=None):
    deadline = time.time() + EXEC_TIMEOUT_S
    con.set_progress_handler(lambda: int(time.time() > deadline), 10000)
    cur = con.execute(sql)
    return cur.fetchmany(max_rows) if max_rows else cur.fetchall()


def subtable_sql(ast: exp.Select) -> str:
    """Gold sub-table: FROM/JOIN/WHERE of the gold query projected onto every column it uses."""
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


def exec_checks(db_path: Path, gold_sql: str, ast):
    res = {}
    try:
        con = _connect(db_path)
    except Exception:
        return {"exec": "no_db"}
    try:
        rows = _run(con, gold_sql)
        res["exec"] = "ok" if rows and any(v is not None for r in rows for v in r) else "empty"
    except Exception as e:
        res["exec"] = "timeout" if "interrupt" in str(e).lower() else "error"
        con.close()
        return res
    try:
        sub = _run(con, subtable_sql(ast), MAX_SUBTABLE_ROWS + 1)
        n = len(sub)
        chars = sum(len(str(v)) + 3 for r in sub for v in r)
        res["subtable_rows"] = n
        res["subtable_tokens"] = chars / CHARS_PER_TOKEN if n <= MAX_SUBTABLE_ROWS else float("inf")
    except Exception as e:
        res["subtable_rows"] = None
        res["subtable_tokens"] = None
        res["subtable_err"] = str(e)[:80]
    con.close()
    return res


# ---------------------------------------------------------------- dataset loaders
def load_spider(exclude_mmqa=True):
    base = ROOT / "data/raw/spider/spider_data"
    schemas = {e["db_id"]: Schema(e) for e in json.load(open(base / "tables.json"))}
    items = [dict(db_id=x["db_id"], question=x["question"], sql=x["query"], split="train")
             for x in json.load(open(base / "train_spider.json"))]
    items += [dict(db_id=x["db_id"], question=x["question"], sql=x["query"], split="train_others")
              for x in json.load(open(base / "train_others.json"))]
    mmqa_dbs = mmqa_source_dbs(schemas) if exclude_mmqa else set()
    for it in items:
        it["db_path"] = base / "database" / it["db_id"] / f"{it['db_id']}.sqlite"
        it["mmqa_source_db"] = it["db_id"] in mmqa_dbs
    return items, schemas, mmqa_dbs


def load_bird():
    base = ROOT / "data/raw/bird"
    tr = base / "train"
    tables = tr / "train_tables.json"  # ships inside train.zip; absent until it is extracted
    schemas = {e["db_id"]: Schema(e) for e in json.load(open(tables))} if tables.exists() else {}
    items = [dict(db_id=x["db_id"], question=x["question"], sql=x["SQL"], split="train",
                  evidence=x.get("evidence", ""))
             for x in json.load(open(tr / "train.json"))]
    for it in items:
        it["db_path"] = tr / "train_databases" / it["db_id"] / f"{it['db_id']}.sqlite"
        it["mmqa_source_db"] = False
    return items, schemas, set()


def _norm_name(x: str) -> str:
    return re.sub(r"[\s_]+", " ", x.lower()).strip()


def mmqa_table_sets():
    """Each MMQA instance as a frozenset of (table, frozenset(columns)), names normalised."""
    import ijson
    sets = collections.Counter()
    for fn in ["Synthesized_two_table.json", "Synthesized_three_table.json"]:
        with open(ROOT / "data/raw/mmqa_mirror" / fn, "rb") as f:
            for r in ijson.items(f, "item"):
                sets[frozenset((_norm_name(n), frozenset(_norm_name(c) for c in t["table_columns"]))
                               for n, t in zip(r["table_names"], r["tables"]))] += 1
    return sets


def mmqa_source_dbs(schemas):
    """Spider DBs containing every table of some MMQA instance, matched on name AND columns."""
    cache = ROOT / "data/cache/mmqa_source_dbs.json"
    if cache.exists():
        return set(json.load(open(cache)))
    db_tables = {db: {(_norm_name(t), frozenset(_norm_name(c) for c in cs)) for t, cs in s.cols.items()}
                 for db, s in schemas.items()}
    hits = set()
    for inst in mmqa_table_sets():
        for db, tabs in db_tables.items():
            if all(any(n == tn and cols <= tc for tn, tc in tabs) for n, cols in inst):
                hits.add(db)
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sorted(hits), open(cache, "w"))
    return hits


def load_mmqa():
    import ijson
    items = []
    for fn, k in [("Synthesized_two_table.json", 2), ("Synthesized_three_table.json", 3)]:
        with open(ROOT / "data/raw/mmqa_mirror" / fn, "rb") as f:
            for r in ijson.items(f, "item"):
                items.append(dict(db_id="|".join(sorted(r["table_names"])), sql=r["SQL"],
                                  question=r["Question"], split=f"{k}-table",
                                  db_path=None, mmqa_source_db=False))
    return items, {}, set()


# ---------------------------------------------------------------- report
def main(name):
    loader = {"spider": load_spider, "bird": load_bird, "mmqa": load_mmqa}[name]
    items, schemas, excluded = loader()
    rows = []
    for i, it in enumerate(items):
        a, ast = analyse_sql(it["sql"], schemas.get(it["db_id"]))
        r = dict(db_id=it["db_id"], split=it["split"], mmqa_source_db=it["mmqa_source_db"],
                 strict=not a["reasons"], relaxed=not a["relaxed_reasons"], **a)
        if r["relaxed"] and it["db_path"] is not None and Path(it["db_path"]).exists():
            r.update(exec_checks(it["db_path"], it["sql"], ast))
        rows.append(r)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(items)}", file=sys.stderr)

    out = ROOT / f"data/cache/action_fit_{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    def summarise(rs, label):
        n = len(rs)
        multi = [r for r in rs if "no_join" not in r["reasons"] and "parse_error" not in r["reasons"]]
        st = [r for r in rs if r["strict"]]
        rx = [r for r in rs if r["relaxed"]]

        def runs(xs):
            return [r for r in xs if r.get("exec") == "ok"] if xs and "exec" in xs[0] else xs

        def fits(xs):
            return [r for r in runs(xs) if (r.get("subtable_tokens") or float("inf")) <= BUDGET_TOKENS]

        print(f"\n=== {label}: {n} questions, {len({r['db_id'] for r in rs})} DBs ===")
        print(f"  multi-table (>=1 join, parses)      : {len(multi):6d}")
        for tier, xs in [("strict", st), ("relaxed", rx)]:
            line = f"  fits {tier:7s} action set           : {len(xs):6d}  ({len({r['db_id'] for r in xs})} DBs)"
            if xs and "exec" in xs[0]:
                line += (f" | executes non-empty: {len(runs(xs))}"
                         f" | gold sub-table <= {BUDGET_TOKENS} tok: {len(fits(xs))}"
                         f" ({len({r['db_id'] for r in fits(xs)})} DBs)")
            print(line)
        hops = collections.Counter(r["n_joins"] for r in rx)
        print(f"  relaxed by #joins                   : {dict(sorted(hops.items()))}")
        print(f"  relaxed by category                 : {dict(collections.Counter(r['category'] for r in rx))}")
        reasons = collections.Counter(x for r in multi for x in r["reasons"])
        print(f"  why multi-table questions fail strict: {reasons.most_common(12)}")

    summarise(rows, f"{name} (all)")
    if excluded:
        print(f"\nMMQA source DBs found in Spider: {len(excluded)}")
        summarise([r for r in rows if not r["mmqa_source_db"]], f"{name} (excluding MMQA source DBs)")
    print(f"\nper-question results: {out}")


if __name__ == "__main__":
    main(sys.argv[1])
