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
    """A filter `column op value(s)`. Values are raw SQL literals (str / int / float / None), so
    the filter can be re-executed exactly; compare anchors with `key()`, which normalises them."""

    column: str  # "head.born_state"
    op: str  # strict: "=", "<", ">"; relaxed adds "<=", ">=", "!=", "like", "not_like", "in",
    #          "not_in", "between", "is" (IS NULL), "not_is" (IS NOT NULL)
    values: tuple = ()  # one value for comparisons/LIKE, two for BETWEEN, n for IN, none for IS NULL

    def key(self) -> tuple:
        from mhtable.data.norm import norm_value

        vals = tuple(norm_value(v) for v in self.values)
        return (self.column.lower(), self.op, tuple(sorted(vals)) if self.op in ("in", "not_in") else vals)

    @property
    def table(self) -> str:
        return self.column.split(".", 1)[0]

    @classmethod
    def from_dict(cls, d: dict) -> "Anchor":
        return cls(d["column"], d["op"], tuple(d.get("values", ())))


ActionKind = Literal["SEED", "ANCHOR", "JOIN", "STOP"]


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    table: str | None = None  # SEED: the table; JOIN: the table being joined in
    anchor: Anchor | None = None
    edge: Edge | None = None
    # Cardinality look-ahead: features, not identity — excluded from equality and hashing.
    lookahead_rows: int = field(default=0, compare=False)  # COUNT(*) of the result if taken
    lookahead_tokens: int = field(default=0, compare=False)

    @staticmethod
    def stop() -> "Action":
        return Action("STOP")

    def key(self) -> tuple:
        """Identity for matching an enumerated action against a gold one (values normalised)."""
        return (self.kind, self.table, self.anchor.key() if self.anchor else None,
                tuple(sorted(tuple(sorted(p)) for p in self.edge)) if self.edge else None)

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(kind=d["kind"], table=d.get("table"),
                   anchor=Anchor.from_dict(d["anchor"]) if d.get("anchor") else None,
                   edge=tuple(tuple(p) for p in d["edge"]) if d.get("edge") else None,
                   lookahead_rows=d.get("lookahead_rows", 0), lookahead_tokens=d.get("lookahead_tokens", 0))


GOLD_FIELDS = frozenset({"gold_sql", "gold_answer", "gold_edges", "gold_anchors", "gold_tables",
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
    gold_answer: list[list[str]]  # rows from executing gold_sql, each cell normalised
    gold_tables: list[str] = field(default_factory=list)  # tables of the gold query (lowercase)
    gold_edges: list[Edge] = field(default_factory=list)
    gold_anchors: list[Anchor] = field(default_factory=list)
    # One entry per gold action except STOP: the intermediate relation after that action, as
    # {"rows": |R_t|, "truncated": bool, "rowids": {table: [rowid, ...]}} (rowids capped).
    gold_entity_sets: list[dict] = field(default_factory=list)
    gold_trajectory: list[Action] = field(default_factory=list)  # canonical sequence for BC
    monotone: bool | None = None  # False if aggregation / ORDER BY..LIMIT / negation (report A.7)
    fail_reasons: list[str] = field(default_factory=list)  # why it is outside a tier
    reference_answer: str | None = None  # answer text shipped by the source (MMQA), kept for audit

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldRecord":
        return cls(instance_id=d["instance_id"], gold_sql=d["gold_sql"],
                   gold_answer=[list(r) for r in d["gold_answer"]],
                   gold_tables=list(d.get("gold_tables", [])),
                   gold_edges=[tuple(tuple(p) for p in e) for e in d.get("gold_edges", [])],
                   gold_anchors=[Anchor.from_dict(a) for a in d.get("gold_anchors", [])],
                   gold_entity_sets=list(d.get("gold_entity_sets", [])),
                   gold_trajectory=[Action.from_dict(a) for a in d.get("gold_trajectory", [])],
                   monotone=d.get("monotone"),
                   fail_reasons=list(d.get("fail_reasons", [])),
                   reference_answer=d.get("reference_answer"))


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
