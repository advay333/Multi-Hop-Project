"""
Mock reader: answers correctly iff the sub-table is sufficient (PLAN.md P4).

A free stand-in for the LLM reader while debugging RL: the reward then depends only on what the
policy selected. It needs gold information (the sufficiency verdict and the gold answer, passed
by the rollout in `meta`), so it is refused when the rollout runs with `oracle_mode="off"` and
must never produce a reported result.
"""

from __future__ import annotations

from mhtable.metrics import format_answer
from mhtable.readers.base import Reader
from mhtable.registry import register


@register("reader", "mock")
class MockReader(Reader):
    id = "mock"
    needs_gold = True

    def read_batch(self, questions, subtables, meta=None):
        if meta is None or any(m is None for m in meta):
            raise ValueError("the mock reader needs gold meta (sufficient, gold_answer); run the rollout "
                             "with oracle_mode='reward' or 'metric_only'")
        return [format_answer(m["gold_answer"]) if m["sufficient"] else "" for m in meta]

    def cache_extra(self, meta):
        return f"|sufficient={int(bool(meta and meta['sufficient']))}"
