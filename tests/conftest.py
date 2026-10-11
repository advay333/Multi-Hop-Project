"""Shared fixtures: a tiny copy of the head / management / department example (DESIGN.md §10)."""

import json
import sqlite3

import pytest

SCHEMA_ENTRY = {
    "db_id": "department_management",
    "table_names_original": ["department", "head", "management"],
    "column_names_original": [[-1, "*"], [0, "Department_ID"], [0, "Name"], [0, "Creation"],
                              [1, "head_ID"], [1, "name"], [1, "born_state"],
                              [2, "department_ID"], [2, "head_ID"], [2, "temporary_acting"]],
    "column_types": ["text", "number", "text", "text", "number", "text", "text", "number", "number", "text"],
    "primary_keys": [1, 4, 7],
    "foreign_keys": [[8, 4], [7, 1]],
}

ROWS = {
    "department": [(1, "State", "1789"), (2, "Treasury", "1789"), (7, "Commerce", "1903")],
    "head": [(1, "Tiger Woods", "Alabama"), (2, "Sergio García", "California")],
    "management": [(7, 1, "Yes"), (2, 2, "No")],
}

JOIN3 = ("SELECT DISTINCT T1.Creation FROM department AS T1 JOIN management AS T2 "
         "ON T1.Department_ID = T2.department_ID JOIN head AS T3 ON T2.head_ID = T3.head_ID ")

QUESTIONS = [
    {"db_id": "department_management", "question": "Creation years of departments managed by someone born in Alabama?",
     "query": JOIN3 + "WHERE T3.born_state = 'Alabama'"},
    {"db_id": "department_management", "question": "How many departments are there?",
     "query": "SELECT count(*) FROM department"},
    {"db_id": "department_management", "question": "Creation years of departments managed by someone born in Texas?",
     "query": JOIN3 + "WHERE T3.born_state = 'Texas'"},
]


def write_db(path):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE department ("Department_ID" INTEGER PRIMARY KEY, "Name" TEXT, "Creation" TEXT)')
    con.execute('CREATE TABLE head ("head_ID" INTEGER PRIMARY KEY, "name" TEXT, "born_state" TEXT)')
    con.execute('CREATE TABLE management ("department_ID" INTEGER, "head_ID" INTEGER, "temporary_acting" TEXT)')
    for t, rows in ROWS.items():
        con.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
    con.commit()
    con.close()


@pytest.fixture
def dept_db(tmp_path):
    p = tmp_path / "department_management.sqlite"
    write_db(p)
    return p


@pytest.fixture
def spider_root(tmp_path):
    """A miniature extracted spider.zip with one database and three questions."""
    base = tmp_path / "spider_extract" / "spider_data"
    db_dir = base / "database" / "department_management"
    db_dir.mkdir(parents=True)
    write_db(db_dir / "department_management.sqlite")
    (base / "tables.json").write_text(json.dumps([SCHEMA_ENTRY]))
    (base / "train_spider.json").write_text(json.dumps(QUESTIONS))
    return tmp_path / "spider_extract"
