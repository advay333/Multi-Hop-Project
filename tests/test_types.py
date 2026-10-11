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
        gold_edges=[edge],
        gold_anchors=[Anchor("head.born_state", "=", "alabama")],
        gold_entity_sets=[["1"], ["7"]],
        gold_trajectory=[Action("ANCHOR", "head", anchor=Anchor("head.born_state", "=", "alabama")),
                         Action("JOIN", "management", edge=edge), Action.stop()],
    )
    assert GoldRecord.from_dict(json.loads(json.dumps(gold.to_dict()))) == gold
