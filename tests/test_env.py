import json
import random
import shutil
from pathlib import Path

import pytest

from mhtable.data.schema import SchemaGraph, make_edge
from mhtable.env.actions import ActionConfig
from mhtable.env.env import EnvConfig, EnvData, TableEnv
from mhtable.env.index import CellSearch, build_db_index
from mhtable.types import Action, Anchor, Instance

from conftest import QUESTIONS, SCHEMA_ENTRY

REF = "spider/department_management"


@pytest.fixture
def data_dir(tmp_path, dept_db):
    """A miniature local copy of the data repo: schema, database and index for one DB."""
    root = tmp_path / "data"
    schema = SchemaGraph.from_tables_json(SCHEMA_ENTRY)
    (root / "schemas").mkdir(parents=True)
    (root / "schemas" / "spider.json").write_text(json.dumps({REF: schema.to_dict()}))
    (root / "db" / "spider").mkdir(parents=True)
    shutil.copy(dept_db, root / "db" / f"{REF}.sqlite")
    idx = build_db_index(REF, root / "db" / f"{REF}.sqlite", schema)
    (root / "index" / "spider").mkdir(parents=True)
    (root / "index" / f"{REF}.json").write_text(json.dumps(idx.to_dict(), default=str))
    return root


def _inst(q=QUESTIONS[0]["question"]):
    return Instance("spider:department_management:0", "spider", REF, "department_management", q,
                    tables=["department", "head", "management"])


def _env(data_dir, **action_kw):
    return TableEnv(EnvConfig(actions=ActionConfig(**action_kw)), EnvData(data_dir))


GOLD = [Action("ANCHOR", "head", anchor=Anchor("head.born_state", "=", ("Alabama",))),
        Action("JOIN", "management", edge=make_edge("head.head_id", "management.head_id")),
        Action("JOIN", "department", edge=make_edge("department.department_id", "management.department_id")),
        Action.stop()]


def test_index_stats_edges_and_cells(data_dir):
    idx = EnvData(data_dir).index(REF)[0]
    assert idx.columns["head.head_id"].unique and idx.columns["head.head_id"].id_like
    assert not idx.columns["head.born_state"].id_like
    edges = {c.edge: c for c in idx.edges}
    for gold in (GOLD[1].edge, GOLD[2].edge):
        assert gold in edges and edges[gold].j == 1.0 and edges[gold].omega >= 1.0
    assert "head.head_id" not in idx.cells  # surrogate keys are not anchor values
    hits = CellSearch(idx).lookup("someone born in alabama")
    assert hits[0][:2] == ("head.born_state", "Alabama")


def test_gold_trajectory_is_inside_the_action_space(data_dir):
    env = _env(data_dir)
    s = env.reset(_inst())
    for gold in GOLD:
        cands = env.candidate_actions(s)
        match = [a for a in cands if a.key() == gold.key()]
        assert match, f"gold {gold.kind} not offered at t={s.t}: {[a.key() for a in cands]}"
        s, done = env.step(s, match[0])
    assert done and s.n_rows == 1
    m = env.materialise(s)
    assert "1903" in m.text and "Alabama" in m.text and not m.overflow
    assert m.rows_shown == 1 and m.tokens <= env.cfg.budget_tokens


def test_masking_rules(data_dir):
    env = _env(data_dir, tier="strict")
    s = env.reset(_inst())
    first = env.candidate_actions(s)
    assert {a.kind for a in first} <= {"SEED", "ANCHOR"}  # no STOP / JOIN at t = 0
    s, _ = env.step(s, next(a for a in first if a.kind == "SEED" and a.table == "head"))
    later = env.candidate_actions(s)
    assert any(a.kind == "STOP" for a in later)
    assert all(a.table in s.visited for a in later if a.kind == "ANCHOR")  # anchors on visited tables only
    assert all(a.table not in s.visited for a in later if a.kind == "JOIN")
    s.t = env.cfg.actions.horizon - 1
    assert [a.kind for a in env.candidate_actions(s)] == ["STOP"]


def test_lookahead_counts_match_execution(data_dir):
    env = _env(data_dir)
    s = env.reset(_inst())
    for a in env.candidate_actions(s):
        nxt, _ = env.step(s, Action(a.kind, a.table, a.anchor, a.edge))  # recount without the hint
        assert nxt.n_rows == a.lookahead_rows


def test_budget_truncates(data_dir):
    env = _env(data_dir)
    s = env.reset(_inst())
    s, _ = env.step(s, Action("SEED", "department"))
    m = env.materialise(s, budget=25)
    assert m.overflow and m.rows_shown < 3 and m.tokens <= 25


def test_random_policy_never_crashes(data_dir):
    rng = random.Random(0)
    for tier in ("strict", "relaxed"):
        env = _env(data_dir, tier=tier)
        for q in [x["question"] for x in QUESTIONS] + ["How many heads older than 50 are there in 1789?"]:
            for _ in range(20):
                s, done = env.reset(_inst(q)), False
                while not done:
                    s, done = env.step(s, rng.choice(env.candidate_actions(s)))
                assert s.t <= env.cfg.actions.horizon
                env.materialise(s)


def test_env_never_touches_gold():
    """Firewall: no env module imports the oracle or GoldRecord, or opens a gold/ file."""
    import ast

    for f in Path(__file__).parents[1].joinpath("src", "mhtable", "env").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported |= {node.module or ""} | {f"{node.module}.{a.name}" for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
        assert not any("oracle" in m or m.endswith("GoldRecord") for m in imported), f.name
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not any(s.startswith("gold/") or "/gold/" in s for s in strings), f.name
