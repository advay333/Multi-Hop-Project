"""
Firewall check (PLAN.md §6 rule 4, DESIGN.md §8): gold SQL, gold answers, entity sets and gold
trajectories are training-time reward signals only. The featuriser and the reader prompt builder
call `assert_gold_free` on what they are given, so a gold record can never reach the policy's
input or the reader's prompt by accident.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from mhtable.types import GOLD_FIELDS


class FirewallError(AssertionError):
    pass


def assert_gold_free(obj, _depth: int = 0) -> None:
    """Raise FirewallError if `obj` is, or contains, something carrying gold fields."""
    if obj is None or isinstance(obj, (str, int, float, bool)) or _depth > 3:
        return
    if isinstance(obj, dict):
        names = set(obj)
    elif is_dataclass(obj):
        names = {f.name for f in fields(obj)}
    else:
        names = set(getattr(obj, "__dict__", {}))
    leaked = names & GOLD_FIELDS
    if leaked:
        raise FirewallError(f"gold fields {sorted(leaked)} reached a gold-free component ({type(obj).__name__})")
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple)):
        children = obj
    elif is_dataclass(obj):
        children = [getattr(obj, f.name) for f in fields(obj)]
    else:
        children = []
    for child in children:
        if isinstance(child, (dict, list, tuple)) or is_dataclass(child):
            assert_gold_free(child, _depth + 1)
