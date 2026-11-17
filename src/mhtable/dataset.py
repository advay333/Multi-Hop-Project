"""
Training/evaluation question sets: (Instance, GoldRecord) pairs for a split and action tier.

    items = load_split(EnvData(data_dir, index_dir, revision), "train", "relaxed", ("spider", "mmqa"))

Gold-aware (it reads `gold/`), so it is used by training and evaluation code, never by the
environment, featuriser or reader. A question is kept when it is in the split, fits the tier,
and its oracle labels are usable (non-empty gold relation). Records built before data-v3 get
the same gold-analysis fix-ups the rebuild applies (`as_rebuilt`; a no-op on data-v3).
"""

from __future__ import annotations

import json
import random

from mhtable.analysis.coverage import usable
from mhtable.analysis.reach import as_rebuilt
from mhtable.env.env import EnvData
from mhtable.types import GoldRecord, Instance

FAMILY = {"spider": "spider_mmqa", "mmqa": "spider_mmqa", "bird": "bird"}


def load_split(data: EnvData, split: str, tier: str, sources=("spider", "mmqa"),
               limit: int | None = None, seed: int = 0) -> list[tuple[Instance, GoldRecord]]:
    """`limit` keeps a seeded random subset (for smoke runs)."""
    if tier not in ("strict", "relaxed"):
        raise ValueError(f"tier must be strict or relaxed, got {tier!r}")
    out = []
    for src in sources:
        ids = set(data._file(data.data_dir, f"splits/{FAMILY[src]}/{split}.txt").read_text().split())
        golds = {}
        with open(data._file(data.data_dir, f"gold/{src}.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["instance_id"] in ids:
                    golds[d["instance_id"]] = d
        with open(data._file(data.data_dir, f"instances/{src}.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["instance_id"] not in golds or not d["tier"].get(tier):
                    continue
                gold = GoldRecord.from_dict(golds[d["instance_id"]])
                if usable(gold):
                    out.append((Instance.from_dict(d), as_rebuilt(gold)))
    if limit is not None and limit < len(out):
        random.Random(seed).shuffle(out)
        out = out[:limit]
    return out
