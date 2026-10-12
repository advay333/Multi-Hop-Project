import sqlite3

import pytest

from mhtable.data.execute import connect_ro
from mhtable.data.schema import SchemaGraph, make_edge
from mhtable.oracle import items, parse, potential, run_oracle
from mhtable.types import Action, Anchor

from conftest import JOIN3, SCHEMA_ENTRY


@pytest.fixture
def schema():
    return SchemaGraph.from_tables_json(SCHEMA_ENTRY)


def test_parse_gold_structure(schema):
    p, _ = parse(JOIN3 + "WHERE T3.born_state = 'Alabama'", schema)
    assert p.tables == ["department", "management", "head"]
    assert p.edges == [make_edge("department.department_id", "management.department_id"),
                       make_edge("management.head_id", "head.head_id")]
    assert p.anchors == [Anchor("head.born_state", "=", ("Alabama",))]
    assert p.monotone
    p, _ = parse("SELECT count(*) FROM department AS T1 JOIN management AS T2 ON T1.Department_ID = T2.department_ID", schema)
    assert not p.monotone and p.anchors == []
    p, _ = parse("SELECT count(*) FROM department", schema)
    assert p is None  # no join: outside both tiers


def test_canonical_trajectory_and_entity_sets(schema, dept_db):
    p, _ = parse(JOIN3 + "WHERE T3.born_state = 'Alabama'", schema)
    res = run_oracle(p, schema, connect_ro(dept_db))
    assert res.fail_reasons == []
    kinds = [(a.kind, a.table) for a in res.trajectory]
    assert kinds == [("ANCHOR", "head"), ("JOIN", "management"), ("JOIN", "department"), ("STOP", None)]
    assert [es["rows"] for es in res.entity_sets] == [1, 1, 1]
    assert res.entity_sets[-1]["rowids"] == {"department": [7], "head": [1], "management": [1]}  # INTEGER PK is the rowid
    assert [a.lookahead_rows for a in res.trajectory[:-1]] == [1, 1, 1]


def test_start_is_most_selective_anchored_table(schema, dept_db):
    sql = JOIN3 + "WHERE T1.Creation = '1789' AND T3.born_state = 'California'"
    p, _ = parse(sql, schema)
    res = run_oracle(p, schema, connect_ro(dept_db))
    # department.Creation='1789' keeps 2 rows, head.born_state='California' keeps 1 -> start at head
    assert res.trajectory[0] == Action("ANCHOR", "head", anchor=Anchor("head.born_state", "=", ("California",)))
    assert [a.kind for a in res.trajectory] == ["ANCHOR", "JOIN", "JOIN", "ANCHOR", "STOP"]
    assert res.fail_reasons == []


@pytest.mark.parametrize("where", [
    "T3.born_state IN ('Alabama', 'Texas')",
    "T1.Department_ID BETWEEN 2 AND 7",
    "T3.name LIKE 'Tiger%'",
    "T3.born_state IS NOT NULL",
    "T3.born_state NOT IN ('Texas')",
    "T1.Department_ID >= 2",
    "5 > T1.Department_ID",
])
def test_relaxed_filters_replay_exactly(schema, dept_db, where):
    p, _ = parse(JOIN3 + "WHERE " + where, schema)
    assert p is not None
    res = run_oracle(p, schema, connect_ro(dept_db))
    assert res.fail_reasons == [], where


def test_dropped_condition_is_caught(schema, dept_db):
    p, _ = parse(JOIN3 + "WHERE T3.born_state = 'Alabama'", schema)
    p.anchors = []  # simulate a parser that lost the filter
    assert "oracle_reconstruct_mismatch" in run_oracle(p, schema, connect_ro(dept_db)).fail_reasons


def test_join_on_non_unique_columns_is_rejected(tmp_path):
    db = tmp_path / "m2m.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE a (x INTEGER, v TEXT)")
    con.execute("CREATE TABLE b (y INTEGER, w TEXT)")
    con.executemany("INSERT INTO a VALUES (?, ?)", [(1, "p"), (1, "q")])
    con.executemany("INSERT INTO b VALUES (?, ?)", [(1, "r"), (1, "s")])
    con.commit()
    con.close()
    g = SchemaGraph.from_sqlite("m2m", db)
    p, _ = parse("SELECT a.v FROM a JOIN b ON a.x = b.y WHERE b.w = 'r'", g)
    assert "join_not_key" in run_oracle(p, g, connect_ro(db)).fail_reasons


def test_potential_is_jaccard_over_structural_items(schema):
    p, _ = parse(JOIN3 + "WHERE T3.born_state = 'Alabama'", schema)
    gold = items(p.edges, p.anchors)
    assert len(gold) == 3
    assert potential(frozenset(), gold) == 0.0
    assert potential(items([], p.anchors), gold) == pytest.approx(1 / 3)
    assert potential(gold, gold) == 1.0
    wrong = items(p.edges[:1], [Anchor("head.born_state", "=", ("Texas",))])
    assert potential(wrong, gold) == pytest.approx(1 / 4)  # 1 shared of 4 in the union: a wrong item lowers Φ
    same_values_other_spelling = items(p.edges, [Anchor("HEAD.born_state", "=", ("alabama",))])
    assert potential(same_values_other_spelling, gold) == 1.0
