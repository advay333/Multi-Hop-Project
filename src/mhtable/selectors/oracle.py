"""
Oracle selector: follows the gold trajectory (DESIGN.md §6.6). It gives the ceiling a perfect
selection policy could reach inside the action space — an upper bound for evaluation and a
reference line for training curves, never a reported system (it reads gold).

Where a gold action is not offered by the environment it stops (or takes the first candidate
if STOP is not legal yet), so unreachable questions count against the ceiling.
"""

from __future__ import annotations

from typing import Mapping

from mhtable.registry import register
from mhtable.rollout import Decision
from mhtable.types import GoldRecord


@register("selector", "oracle")
class OracleSelector:
    def __init__(self, golds: Mapping[str, GoldRecord]):
        self.golds = golds

    def act(self, states, cands, mode="argmax", generator=None):
        out = []
        for s, c in zip(states, cands):
            plan = self.golds[s.instance.instance_id].gold_trajectory
            want = plan[s.t].key() if s.t < len(plan) else None
            idx = next((i for i, a in enumerate(c) if a.key() == want), None)
            if idx is None:
                idx = next((i for i, a in enumerate(c) if a.kind == "STOP"), 0)
            out.append(Decision(idx))
        return out
