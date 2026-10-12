import json

import pytest

from mhtable.types import Action, Anchor, GoldRecord, Instance


def test_instance_roundtrip_and_firewall():
    inst = Instance("mmqa:dept:0", "mmqa", "mmqa/abc", "department_management", "Which year?")
    assert Instance.from_dict(json.loads(json.dumps(inst.to_dict()))) == inst
    with pytest.raises(ValueError, match="gold"):
        Instance.from_dict({**inst.to_dict(), "gold_sql": "SELECT 1"})


def test_gold_record_roundtrip_through_json():
    edge = (("head.head_id", "management.head_id"),)
    gold = GoldRecord(
        "mmqa:dept:0", "SELECT 1", [["1903"]],
        gold_tables=["head", "management"],
        gold_edges=[edge],
        gold_anchors=[Anchor("head.born_state", "=", ("Alabama",)), Anchor("head.age", "between", (50, 60.5))],
        gold_entity_sets=[{"rows": 1, "truncated": False, "rowids": {"head": [1]}}],
        gold_trajectory=[Action("ANCHOR", "head", anchor=Anchor("head.born_state", "=", ("Alabama",)), lookahead_rows=1),
                         Action("JOIN", "management", edge=edge), Action.stop()],
        monotone=True,
    )
    assert GoldRecord.from_dict(json.loads(json.dumps(gold.to_dict()))) == gold


def test_action_and_anchor_keys():
    a = Action("ANCHOR", "head", anchor=Anchor("head.born_state", "=", ("Alabama",)), lookahead_rows=1)
    b = Action("ANCHOR", "head", anchor=Anchor("HEAD.born_state", "=", ("alabama ",)), lookahead_rows=9)
    assert a.key() == b.key()  # normalised values, case-insensitive column, look-ahead ignored
    assert a == Action("ANCHOR", "head", anchor=Anchor("head.born_state", "=", ("Alabama",)), lookahead_rows=5)
    year_str = Action("ANCHOR", "head", anchor=Anchor("head.year", "=", ("1903",)))
    year_int = Action("ANCHOR", "head", anchor=Anchor("head.year", "=", (1903,)))
    assert year_str.key() == year_int.key()
    assert Anchor("t.c", "in", ("b", "a")).key() == Anchor("t.c", "in", ("a", "b")).key()
    j1 = Action("JOIN", "m", edge=(("h.id", "m.hid"),))
    assert j1.key() == Action("JOIN", "m", edge=(("m.hid", "h.id"),)).key()  # edge direction irrelevant
