"""Track B: SQLite indexes, look-ahead memo, bounded materialise, speed profile."""

import hashlib
import json
import random

from mhtable.analysis.speed import profile_db
from mhtable.env import env as env_mod
from mhtable.env.actions import ActionConfig
from mhtable.env.env import EnvConfig, EnvData, TableEnv
from mhtable.env.sqlindex import index_db, main as sqlindex_main, plan_indexes

from test_env import GOLD, REF, _inst, data_dir  # noqa: F401 - fixture


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _walk(env, seed=0, episodes=20):
    """Counts and sub-table texts of seeded random episodes plus the gold path."""
    out, rng = [], random.Random(seed)
    for _ in range(episodes):
        s, done = env.reset(_inst()), False
        while not done:
            cands = env.candidate_actions(s)
            out.append(tuple((a.key(), a.lookahead_rows) for a in cands))
            s, done = env.step(s, rng.choice(cands))
        out.append(env.materialise(s).text)
    s = env.reset(_inst())
    for g in GOLD:
        s, done = env.step(s, next(a for a in env.candidate_actions(s) if a.key() == g.key()))
    out.append(env.materialise(s).text)
    return out


def _env(data_dir, **cfg):
    return TableEnv(EnvConfig(actions=ActionConfig(edge_threshold=0.5), **cfg), EnvData(data_dir))


def test_indexes_change_no_result_and_are_idempotent(data_dir):
    db = data_dir / "db" / f"{REF}.sqlite"
    before = _walk(_env(data_dir, lookahead_memo=0))
    data = EnvData(data_dir)
    idx, schema = data.index(REF)[0], data.schema(REF)
    plan = plan_indexes(idx, schema, min_rows=0)
    assert any(t == "management" and cols == ("management.head_id",) for t, cols in plan)  # a join column
    assert any(cols == ("head.born_state",) for _, cols in plan)  # an anchor column

    sha0 = _sha(db)
    dry = index_db(db, idx, schema, dry_run=True, min_rows=0)
    assert dry["created"] and _sha(db) == sha0  # dry run changes nothing

    rep = index_db(db, idx, schema, min_rows=0)
    assert rep["created"]
    assert _walk(_env(data_dir, lookahead_memo=0)) == before  # same counts, same sub-tables
    again = index_db(db, idx, schema, min_rows=0)
    assert again["created"] == [] and again["skipped_existing"] == again["planned"]


def test_small_databases_stay_byte_identical(data_dir):
    db = data_dir / "db" / f"{REF}.sqlite"
    sha0 = _sha(db)
    assert sqlindex_main(["--data-dir", str(data_dir), "--family", "spider_mmqa", "--workers", "1"]) == 0
    assert _sha(db) == sha0  # default --min-rows 50k: nothing to index in a tiny DB
    manifest = json.loads((data_dir / "db_index_manifest.json").read_text())
    assert manifest["changed"] == [] and manifest["databases"] == 1


def test_lookahead_memo_reuses_counts(data_dir):
    env = _env(data_dir)
    s = env.reset(_inst())
    first = env.candidate_actions(s)
    q = env.stats["lookahead_queries"]
    second = env.candidate_actions(env.reset(_inst()))
    assert [a.lookahead_rows for a in first] == [a.lookahead_rows for a in second]
    assert env.stats["lookahead_queries"] == q and env.stats["lookahead_memo_hits"] >= q > 0
    assert env.slow_queries and len(env.slow_queries) <= 10


def test_huge_relations_are_read_unordered_and_timeouts_do_not_crash(data_dir, monkeypatch):
    env = _env(data_dir, lookahead_cap=2)  # every relation of >= 2 rows counts as "huge"
    s = env.reset(_inst())
    seed = next(a for a in env.candidate_actions(s) if a.kind == "SEED")
    s, _ = env.step(s, seed)
    assert s.n_rows == 2
    m = env.materialise(s)
    assert m.rows_shown > 0 and not m.timed_out and env.stats["materialise_unordered"] == 1

    def boom(*a, **k):
        raise env_mod.QueryTimeout("x")
    monkeypatch.setattr(env_mod, "run", boom)
    m = env.materialise(s)
    assert m.timed_out and m.rows_shown == 0 and env.stats["materialise_timeouts"] == 1


def test_speed_profile_runs(data_dir):
    r = profile_db(_env(data_dir), [_inst()], budget_s=30, episodes=5, seed=0)
    assert r["episodes"] == 5 and r["crashes"] == 0
    assert r["lookahead"]["queries"] > 0 and r["lookahead"]["timeout_rate"] == 0


def test_manifest_lists_dbs_indexed_by_an_earlier_run(data_dir):
    """A re-run after an interruption creates nothing in an already-indexed DB; the DB must still
    be in `changed`, or the push would skip it."""
    args = ["--data-dir", str(data_dir), "--family", "spider_mmqa", "--workers", "1", "--min-rows", "0"]
    dry_out = data_dir / "plan.json"
    assert sqlindex_main(args + ["--dry-run", "--plan-out", str(dry_out)]) == 0
    plan = json.loads(dry_out.read_text())
    assert plan["changed"] == [] and plan["would_change"] == [REF] and plan["est_added_bytes"] > 0
    assert sqlindex_main(args) == 0
    assert sqlindex_main(args) == 0  # second run: nothing new to create
    manifest = json.loads((data_dir / "db_index_manifest.json").read_text())
    assert manifest["changed"] == [REF] and manifest["per_db"][0]["created"] == []
    assert manifest["per_db"][0]["mhx_indexes"] > 0


def test_env_lookahead_sql_uses_the_new_indexes(data_dir):
    import sqlite3

    from mhtable.data.schema import make_edge
    from mhtable.sqlrender import from_where

    db = data_dir / "db" / f"{REF}.sqlite"
    data = EnvData(data_dir)
    idx, schema = data.index(REF)[0], data.schema(REF)
    index_db(db, idx, schema, min_rows=0)
    from mhtable.types import Anchor

    # filter on head, then join management: without indexes both are full scans
    body = from_where(schema, ["head", "management"], [make_edge("head.head_id", "management.head_id")],
                      [Anchor("head.born_state", "=", ("Alabama",))])
    sql = f"SELECT COUNT(*) FROM (SELECT 1 {body} LIMIT 100000)"  # the env's look-ahead shape
    con = sqlite3.connect(str(db))
    plan = [r[-1] for r in con.execute("EXPLAIN QUERY PLAN " + sql)]
    con.close()
    assert not any("AUTOMATIC" in x for x in plan), plan
    assert any(x.startswith("SEARCH head USING") and "mhx_" in x for x in plan), plan
    assert any(x.startswith("SEARCH management USING") and "mhx_" in x for x in plan), plan


def test_analyze_on_rerun_covers_tables_indexed_earlier(data_dir):
    import sqlite3

    db = data_dir / "db" / f"{REF}.sqlite"
    data = EnvData(data_dir)
    idx, schema = data.index(REF)[0], data.schema(REF)
    before = _walk(_env(data_dir, lookahead_memo=0))
    index_db(db, idx, schema, min_rows=0)
    rep = index_db(db, idx, schema, min_rows=0, analyze=True)  # nothing new to create
    assert rep["created"] == [] and rep["analyzed"]["tables"] > 0
    con = sqlite3.connect(str(db))
    assert con.execute("SELECT COUNT(*) FROM sqlite_stat1").fetchone()[0] > 0
    con.close()
    assert _walk(_env(data_dir, lookahead_memo=0)) == before  # statistics change no result
