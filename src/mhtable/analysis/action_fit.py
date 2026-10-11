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
import sys
import time
from pathlib import Path

from mhtable.data.execute import QueryTimeout, connect_ro, run, subtable_sql
from mhtable.data.schema import SchemaGraph
from mhtable.data.sqlshape import analyse_sql

ROOT = Path(__file__).resolve().parents[3]
H = 5
BUDGET_TOKENS = 4096
CHARS_PER_TOKEN = 3.5
EXEC_TIMEOUT_S = 10
MAX_SUBTABLE_ROWS = 5000  # beyond this the sub-table is far over budget anyway



def exec_checks(db_path: Path, gold_sql: str, ast):
    res = {}
    try:
        con = connect_ro(db_path)
    except Exception:
        return {"exec": "no_db"}
    try:
        rows = run(con, gold_sql, EXEC_TIMEOUT_S)
        res["exec"] = "ok" if rows and any(v is not None for r in rows for v in r) else "empty"
    except Exception as e:
        res["exec"] = "timeout" if isinstance(e, QueryTimeout) else "error"
        con.close()
        return res
    try:
        sub = run(con, subtable_sql(ast), EXEC_TIMEOUT_S, MAX_SUBTABLE_ROWS + 1)
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
    schemas = {e["db_id"]: SchemaGraph.from_tables_json(e) for e in json.load(open(base / "tables.json"))}
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
    schemas = {e["db_id"]: SchemaGraph.from_tables_json(e) for e in json.load(open(tables))} if tables.exists() else {}
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
    db_tables = {db: {(_norm_name(t), frozenset(_norm_name(c) for c in s.column_set(t))) for t in s.tables}
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
        a, ast = analyse_sql(it["sql"], schemas.get(it["db_id"]), H)
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
