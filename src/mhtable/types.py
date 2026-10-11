"""
Shared record types (DESIGN.md §5).

`Instance` is public: safe to show the policy, featuriser and reader. `GoldRecord` is the
training signal and lives in a separate file (`gold/*.jsonl`), so code that only loads
instances cannot leak gold information — the firewall is structural.

Tensor-valued fields in `Step` are typed `Any` so this module does not import torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

# ((t1.c1, t2.c2), ...) — one join edge; several pairs for a composite key. Columns are
# "table.column" strings, lowercased.
Edge = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Anchor:
    column: str  # "head.born_state"
    op: str  # "=", "<", ">", and in the relaxed tier: "<=", ">=", "!=", "like", "in", ...
    value: str  # normalised with norm_value


ActionKind = Literal["SEED", "ANCHOR", "JOIN", "STOP"]


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    table: str | None = None  # SEED: the table; JOIN: the table being joined in
    anchor: Anchor | None = None
    edge: Edge | None = None
    lookahead_rows: int = 0  # COUNT(*) of the result if taken; gold-free cardinality estimate
    lookahead_tokens: int = 0

    @staticmethod
    def stop() -> "Action":
        return Action("STOP")


GOLD_FIELDS = frozenset({"gold_sql", "gold_answer", "gold_edges", "gold_anchors",
                         "gold_entity_sets", "gold_trajectory"})


@dataclass
class Instance:
    instance_id: str  # "{source}:{db_id}:{n}"
    source: Literal["spider", "bird", "mmqa"]
    db_ref: str  # storage key of the SQLite file, e.g. "mmqa/3fa2c1"
    db_id: str  # source database; used for splitting, never as a feature
    question: str
    evidence: str | None = None  # BIRD external-knowledge hint
    tables: list[str] = field(default_factory=list)  # tables available in the episode
    n_gold_tables: int = 0  # for reporting slices only
    category: str = ""  # Select / List / Count / Numerical
    tier: dict[str, bool] = field(default_factory=lambda: {"strict": False, "relaxed": False})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        names = {f.name for f in fields(cls)}
        extra = set(d) - names
        if extra & GOLD_FIELDS:
            raise ValueError(f"refusing to build an Instance from a record with gold fields: {extra & GOLD_FIELDS}")
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class GoldRecord:
    instance_id: str
    gold_sql: str
    gold_answer: list[str]  # executing gold_sql, normalised
    gold_edges: list[Edge] = field(default_factory=list)
    gold_anchors: list[Anchor] = field(default_factory=list)
    gold_entity_sets: list[list[str]] = field(default_factory=list)  # correct keys after each gold prefix
    gold_trajectory: list[Action] = field(default_factory=list)  # canonical sequence for BC
    fail_reasons: list[str] = field(default_factory=list)  # why it is outside a tier

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldRecord":
        def edge(e):
            return tuple(tuple(p) for p in e)

        def action(a):
            return Action(kind=a["kind"], table=a.get("table"),
                          anchor=Anchor(**a["anchor"]) if a.get("anchor") else None,
                          edge=edge(a["edge"]) if a.get("edge") else None,
                          lookahead_rows=a.get("lookahead_rows", 0),
                          lookahead_tokens=a.get("lookahead_tokens", 0))

        return cls(instance_id=d["instance_id"], gold_sql=d["gold_sql"], gold_answer=list(d["gold_answer"]),
                   gold_edges=[edge(e) for e in d.get("gold_edges", [])],
                   gold_anchors=[Anchor(**a) for a in d.get("gold_anchors", [])],
                   gold_entity_sets=[list(s) for s in d.get("gold_entity_sets", [])],
                   gold_trajectory=[action(a) for a in d.get("gold_trajectory", [])],
                   fail_reasons=list(d.get("fail_reasons", [])))


@dataclass
class State:
    instance: Instance
    t: int = 0
    visited: list[str] = field(default_factory=list)  # tables in the join tree, visit order
    joins: list[Edge] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    relation_sql: str = ""  # SQL defining the current intermediate relation R_t
    n_rows: int = 0
    est_tokens: int = 0


@dataclass
class Step:
    state_feat: Any
    action_feats: Any
    mask: Any
    action_idx: int
    logp: float
    value: float
    reward: float = 0.0  # filled by rewards.assemble
    phi: float = 0.0  # shaping potential, logged for diagnostics


@dataclass
class Trajectory:
    instance_id: str
    steps: list[Step] = field(default_factory=list)
    subtable: str = ""
    subtable_tokens: int = 0
    answer: str | None = None
    reader_called: bool = False  # False on a cache hit; only misses count against the call budget
    info: dict = field(default_factory=dict)  # metric_only diagnostics (per-hop recall, path match, ...)
