"""
Database-level split (PLAN.md §0): no database appears in two splits.

Spider and MMQA share one namespace: MMQA was built from Spider databases, so an MMQA instance
belongs to the group of its Spider db_id (instances matching several Spider DBs merge those
groups; unmatched ones form their own group). Groups that contain MMQA instances are assigned
train/dev/test at 60/10/30, balanced separately over 2-table and 3-table MMQA instances so both
slices exist in every split. Groups with only Spider questions go to train (MMQA is the
evaluation target; they are extra training data). BIRD keeps its own division: BIRD-train
databases -> train, BIRD-dev databases -> bird_dev.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from mhtable.data.loaders import RawExample

SPLITS = ("train", "dev", "test")
FRACTIONS = {"train": 0.6, "dev": 0.1, "test": 0.3}


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _db_key(ex: RawExample) -> str:
    family = "bird" if ex.source == "bird" else "spider"  # Spider and MMQA share db names
    return f"{family}/{ex.db_id}"


def assign(examples: list[RawExample], seed: int = 0) -> tuple[dict[str, str], dict[str, str]]:
    """Returns ({instance_id: split}, {db_key: split})."""
    uf = _UnionFind()
    for ex in examples:
        uf.find(_db_key(ex))
        for other in ex.extra.get("spider_candidates", [])[1:]:
            uf.union(_db_key(ex), f"spider/{other}")

    strata: dict[str, Counter] = defaultdict(Counter)  # group -> {2: n, 3: n} MMQA instances
    for ex in examples:
        if ex.source == "mmqa":
            strata[uf.find(_db_key(ex))][ex.extra["mmqa_tables"]] += 1

    group_split: dict[str, str] = {}
    totals = Counter()
    for c in strata.values():
        totals.update(c)
    assigned = {s: Counter() for s in SPLITS}
    groups = sorted(strata)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -sum(strata[g].values()))  # stable: large groups first, random ties
    for g in groups:
        def need(s):
            return sum(n * (FRACTIONS[s] * totals[k] - assigned[s][k]) / totals[k] for k, n in strata[g].items())
        best = max(SPLITS, key=need)
        group_split[g] = best
        assigned[best].update(strata[g])

    inst_split, db_split = {}, {}
    for ex in examples:
        key = _db_key(ex)
        if ex.source == "bird":
            split = "bird_dev" if ex.extra.get("bird_split") == "dev" else "train"
        else:
            split = group_split.get(uf.find(key), "train")
        prev = db_split.setdefault(key, split)
        assert prev == split, f"database {key} assigned to both {prev} and {split}"
        inst_split[ex.instance_id] = split
    return inst_split, db_split
