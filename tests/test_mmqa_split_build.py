import json
import sqlite3

from mhtable.data import mmqa_db, split
from mhtable.data.build import build
from mhtable.data.loaders import RawExample
from mhtable.data.raw import RawSource
from mhtable.data.schema import SchemaGraph

from conftest import SCHEMA_ENTRY

RECORD = {
    "id_": 0, "Question": "q", "SQL": "SELECT 1", "answer": "x",
    "table_names": ["department", "management"],
    "tables": [
        {"table_columns": ["Department_ID", "Name", "Budget"],
         "table_content": [[1, "MisÃ©rables", "9.96"], [2, "Treasury", 11.0], [3, None, "1,234"]]},
        {"table_columns": ["department_ID", "head_ID", "temporary_acting"],
         "table_content": [[2, "5", "Yes"]]},
    ],
    "primary_keys": ["department id"], "foreign_keys": ["department id"],
}


def test_fix_mojibake():
    assert mmqa_db.fix_mojibake("MisÃ©rables") == "Misérables"
    assert mmqa_db.fix_mojibake("Misérables") == "Misérables"  # correct text untouched
    assert mmqa_db.fix_mojibake("plain") == "plain"


def test_materialise_types_and_keys(tmp_path):
    p = tmp_path / "m.sqlite"
    mmqa_db.materialise(RECORD, p)
    con = sqlite3.connect(p)
    assert con.execute("SELECT Name FROM department WHERE Department_ID = 1").fetchone() == ("Misérables",)
    assert con.execute("SELECT typeof(Budget), Budget FROM department WHERE Department_ID = 3").fetchone() == ("real", 1234.0)
    assert con.execute("SELECT typeof(head_ID) FROM management").fetchone() == ("integer",)
    spider = {"department_management": SchemaGraph.from_tables_json(SCHEMA_ENTRY)}
    assert mmqa_db.spider_matches(RECORD, spider) == []  # Budget is not a column of the Spider table
    g, src = mmqa_db.schema_for(RECORD, p, "sig", None)
    assert src == "mmqa_lists" and g.fks  # fallback joins department <-> management on department_id
    assert mmqa_db.schema_signature(RECORD) == mmqa_db.schema_signature(json.loads(json.dumps(RECORD)))


def _ex(source, db, n, **extra):
    return RawExample(source, f"q{n}", db, f"{source}/{db}", "q", "SELECT 1", extra=extra)


def test_split_is_db_disjoint_and_stratified():
    exs = []
    for d in range(40):
        db = f"db{d}"
        exs += [_ex("spider", db, i) for i in range(3)]
        exs += [_ex("mmqa", db, 100 + i, mmqa_tables=2 + (d % 2), spider_candidates=[db]) for i in range(5)]
    exs += [_ex("spider", "spider_only", 0)]
    exs += [_ex("mmqa", "a", 900, mmqa_tables=2, spider_candidates=["a", "b"]), _ex("spider", "b", 901)]
    exs += [_ex("bird", "db0", 0, bird_split="dev"), _ex("bird", "x", 1, bird_split="train")]
    inst, dbs = split.assign(exs, seed=0)
    assert dbs["spider/spider_only"] == "train"
    assert dbs["spider/a"] == dbs["spider/b"]  # an ambiguous MMQA match merges the two databases
    assert dbs["bird/db0"] == "bird_dev" and dbs["spider/db0"] != "bird_dev"  # BIRD names are separate
    counts = {s: sum(1 for e in exs if e.source == "mmqa" and inst[e.instance_id] == s) for s in split.SPLITS}
    assert 0.5 < counts["train"] / sum(counts.values()) < 0.7 and counts["dev"] > 0 and counts["test"] > 0
    assert split.assign(exs, seed=0) == (inst, dbs)  # deterministic


def test_build_spider_only(tmp_path, spider_root, monkeypatch):
    from mhtable.data import build as build_mod
    from mhtable.data.loaders import spider

    monkeypatch.setattr(build_mod, "load_family",
                        lambda fam, raw, out, limit: {"spider": spider.load(spider_root, limit)})
    out = tmp_path / "out"
    stats = build("spider_mmqa", out, RawSource(), workers=1)
    gold = [json.loads(l) for l in open(out / "gold" / "spider.jsonl", encoding="utf-8")]
    insts = [json.loads(l) for l in open(out / "instances" / "spider.jsonl", encoding="utf-8")]
    assert gold[0]["gold_answer"] == [["1903"]]
    assert insts[0]["tier"] == {"strict": True, "relaxed": True}
    assert not insts[1]["tier"]["relaxed"] and "no_join" in gold[1]["fail_reasons"]
    assert "empty_answer" in gold[2]["fail_reasons"] and not insts[2]["tier"]["relaxed"]
    assert "gold_sql" not in insts[0]  # firewall: gold lives only in gold/
    assert (out / "db" / "spider" / "department_management.sqlite").exists()
    assert stats["sources"]["spider"]["relaxed"] == 1
    assert (out / "splits" / "spider_mmqa" / "train.txt").read_text().count("spider:") == 3
