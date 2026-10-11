import pytest

from mhtable.data.execute import QueryTimeout, connect_ro, run
from mhtable.data.schema import SchemaGraph, make_edge
from mhtable.data.sqlshape import analyse_sql

from conftest import QUESTIONS, SCHEMA_ENTRY


def test_schema_from_tables_json_and_roundtrip():
    g = SchemaGraph.from_tables_json(SCHEMA_ENTRY)
    assert g.tables == ["department", "head", "management"]
    assert g.is_pk("HEAD", "head_id")
    assert make_edge("management.head_id", "head.head_id") in g.fks
    assert g.edges_from("department") == [make_edge("department.department_id", "management.department_id")]
    assert SchemaGraph.from_dict(g.to_dict()) == g


def test_schema_from_sqlite(dept_db):
    g = SchemaGraph.from_sqlite("x", dept_db)
    assert g.column_set("head") == {"head_id", "name", "born_state"}
    assert g.is_pk("department", "Department_ID") and not g.fks


def test_analyse_sql_tiers():
    g = SchemaGraph.from_tables_json(SCHEMA_ENTRY)
    f, _ = analyse_sql(QUESTIONS[0]["query"], g)
    assert f["reasons"] == [] and f["n_joins"] == 2 and f["n_anchors"] == 1 and f["category"] == "Select"
    f, _ = analyse_sql(QUESTIONS[1]["query"], g)
    assert "no_join" in f["relaxed_reasons"] and f["category"] == "Count"
    f, _ = analyse_sql(QUESTIONS[0]["query"].replace("= 'Alabama'", "LIKE 'Ala%'"), g)
    assert f["relaxed_reasons"] == [] and "like" in f["reasons"]  # fits relaxed only


def test_run_and_timeout(dept_db):
    con = connect_ro(dept_db)
    assert run(con, "SELECT count(*) FROM head") == [(2,)]
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"
    with pytest.raises(QueryTimeout):
        run(con, slow, timeout_s=0.2)
