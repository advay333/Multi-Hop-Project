"""
Candidate-action enumeration (report C.3) — gold-free.

Masking rules (illegal actions are never offered, so there is nothing to penalise):
  t = 0        SEED(table) or ANCHOR on any episode table (an ANCHOR seeds its table)
  t >= 1       ANCHOR on visited tables (<= 2 anchors strict, <= 3 relaxed, no duplicates),
               JOIN from a visited table to an unvisited episode table, STOP
  t = H - 1    STOP only

Anchor candidates (all derived from the question text and the database, never from gold):
  categorical  `col = value` for the top-8 cell-index matches of the question (+ BIRD evidence)
  numeric      numbers in the question x top-3 numeric columns x ops. Columns rank by whether
               the number lies in the column's value range, then name similarity; operators
               rank by wording cues ("more than" -> >, "at least" -> >=, "not" -> !=), then "="
  relaxed adds ops <=, >=, !=, BETWEEN pairs of question numbers, IN over several matched
               values of one column, `!=` on matched text values when the question negates,
               LIKE '%phrase%' for quoted phrases (on question-like text columns) and for
               question tokens found inside matched values
Caps keep |A(s)| <= 25: <= 6 SEED, <= 14 ANCHOR, <= 6 JOIN, 1 STOP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mhtable.data.norm import norm_value
from mhtable.data.schema import SchemaGraph
from mhtable.env.index import CellSearch, DbIndex
from mhtable.env.text import LETTER, QUOTED, Similarity, dates, numbers, op_cues, tokens
from mhtable.types import Action, Anchor, State


@dataclass
class ActionConfig:
    horizon: int = 5
    tier: str = "relaxed"  # strict | relaxed
    edges: str = "inferred"  # inferred | declared
    edge_threshold: float = 0.6667  # minimum omega for inferred joins: keeps 98% of gold joins (train, data-v2)
    max_seed: int = 6
    max_categorical: int = 8
    max_numeric_columns: int = 3
    max_anchor_actions: int = 14
    max_join_actions: int = 6

    @property
    def max_anchors(self) -> int:
        return 2 if self.tier == "strict" else 3

    @property
    def numeric_ops(self) -> tuple[str, ...]:
        return ("=", "<", ">") if self.tier == "strict" else ("=", "<", ">", "<=", ">=", "!=")


@dataclass
class EpisodeContext:
    """Everything gold-free the enumerator needs for one episode, computed once at reset."""

    schema: SchemaGraph
    index: DbIndex
    query_text: str  # question (+ BIRD evidence)
    tables: list[str]  # episode tables
    cell_matches: list[tuple[str, object, float]] = field(default_factory=list)  # (column, value, score)
    query_numbers: list = field(default_factory=list)
    cues: set = field(default_factory=set)  # comparison operators suggested by the wording
    quoted: list = field(default_factory=list)  # quoted phrases in the question
    letters: list = field(default_factory=list)  # "contains the letter s"
    query_dates: list = field(default_factory=list)  # ISO date literals in the question
    column_scores: dict[str, float] = field(default_factory=dict)  # column key -> similarity to query
    table_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, schema: SchemaGraph, index: DbIndex, search: CellSearch, query_text: str,
              tables: list[str], sim: Similarity) -> "EpisodeContext":
        ctx = cls(schema, index, query_text, [t.lower() for t in tables])
        ctx.cell_matches = search.lookup(query_text, set(ctx.tables), k=40)
        ctx.query_numbers = numbers(query_text)
        ctx.cues = op_cues(query_text)
        ctx.quoted = QUOTED.findall(query_text)
        ctx.letters = LETTER.findall(query_text)
        ctx.query_dates = dates(query_text)
        # A year compared with a date column means its first day ("before the year 2000").
        ctx.query_dates += [f"{y}-01-01" for y in ctx.query_numbers
                            if isinstance(y, int) and 1900 <= y <= 2100 and f"{y}-01-01" not in ctx.query_dates]
        for t in ctx.tables:
            cols = schema.columns.get(t, [])
            ctx.table_scores[t] = sim.score(query_text, f"{t} {' '.join(cols)}")
            for c in cols:
                ctx.column_scores[f"{t}.{c.lower()}"] = sim.score(query_text, c)
        return ctx


def _anchor_candidates(s: State, ctx: EpisodeContext, cfg: ActionConfig, allowed: set[str]) -> list[tuple[float, Action]]:
    applied = {a.key() for a in s.anchors}
    out: list[tuple[float, Action]] = []

    def add(score, table, anchor):
        if anchor.key() not in applied:
            out.append((score, Action("ANCHOR", table, anchor=anchor)))

    # Categorical: question terms matched to cell values.
    matches = [m for m in ctx.cell_matches if m[0].split(".", 1)[0] in allowed]
    for col, val, score in matches[:cfg.max_categorical]:
        add(1.0 + score, col.split(".", 1)[0], Anchor(col, "=", (val,)))

    # Numeric: numbers mentioned in the question against plausible numeric columns.
    def in_range(col, n):
        st = ctx.index.columns[col]
        return st.min is not None and st.max is not None and st.min <= n <= st.max

    num_cols = [k for k, st in ctx.index.columns.items() if st.numeric and k.split(".", 1)[0] in allowed]
    nums = ctx.query_numbers
    num_cols.sort(key=lambda k: (-any(in_range(k, n) for n in nums), -ctx.column_scores.get(k, 0.0)))
    num_cols = num_cols[:cfg.max_numeric_columns]
    # Operators: "=", "<", ">" always; the relaxed-only ones (<=, >=, !=) only when the wording cues them.
    ops = [op for op in cfg.numeric_ops if op in ("=", "<", ">") or op in ctx.cues]
    for rank, col in enumerate(num_cols):
        for n in nums:
            base = 0.5 + 0.3 * in_range(col, n) + ctx.column_scores.get(col, 0.0) - 0.1 * rank
            for op in ops:
                bonus = 0.3 if op in ctx.cues else (0.1 if op == "=" else 0.0)
                add(base + bonus, col.split(".", 1)[0], Anchor(col, op, (n,)))

    # Dates written in the question, against date-like text columns (values compare as ISO strings).
    if ctx.query_dates:
        date_cols = [k for k, vals in ctx.index.cells.items() if k.split(".", 1)[0] in allowed
                     and vals and isinstance(vals[0], str) and vals[0][:4].isdigit() and vals[0][4:5] == "-"]
        date_cols.sort(key=lambda k: -ctx.column_scores.get(k, 0.0))
        for col in date_cols[:2]:
            for d in ctx.query_dates:
                for op in [o for o in cfg.numeric_ops if o in ("=", "<", ">") or o in ctx.cues]:
                    bonus = 0.3 if op in ctx.cues else (0.1 if op == "=" else 0.0)
                    add(0.8 + bonus, col.split(".", 1)[0], Anchor(col, op, (d,)))

    if cfg.tier == "relaxed":
        # BETWEEN two question numbers on the top numeric column.
        sorted_nums = sorted(nums)
        if num_cols and len(sorted_nums) >= 2:
            for i in range(len(sorted_nums)):
                for j in range(i + 1, len(sorted_nums)):
                    add(0.6, num_cols[0].split(".", 1)[0], Anchor(num_cols[0], "between", (sorted_nums[i], sorted_nums[j])))
        # IN: several matched values of one column.
        by_col: dict[str, list] = {}
        for col, val, score in matches:
            if score >= 1.0:  # value spelled out in the question
                by_col.setdefault(col, []).append(val)
        for col, vals in by_col.items():
            if len(vals) >= 2:
                add(1.5, col.split(".", 1)[0], Anchor(col, "in", tuple(vals)))
        # != on matched text values when the question negates ("did not end in San Francisco").
        if "!=" in ctx.cues:
            for col, val, score in matches[:3]:
                add(1.2 + score / 10, col.split(".", 1)[0], Anchor(col, "!=", (val,)))
        # LIKE: quoted phrases on the most question-like text columns ("whose name has 'Led'").
        text_cols = sorted((k for k, st in ctx.index.columns.items()
                            if not st.numeric and k.split(".", 1)[0] in allowed),
                           key=lambda k: -ctx.column_scores.get(k, 0.0))[:3]
        for phrase in ctx.quoted + ctx.letters:
            for col in text_cols:
                add(1.1, col.split(".", 1)[0], Anchor(col, "like", (f"%{phrase}%",)))
        # LIKE: a question token found inside (not equal to) a matched value.
        qtok = [t for t in tokens(ctx.query_text) if len(t) >= 4]
        for col, val, score in matches[:4]:
            if score < 1.0 and isinstance(val, str):
                nv = norm_value(val)
                for t in qtok:
                    if t in nv and t != nv:
                        add(0.9, col.split(".", 1)[0], Anchor(col, "like", (f"%{t}%",)))
                        break

    best: dict[tuple, tuple[float, Action]] = {}
    for score, a in out:
        k = a.key()
        if k not in best or score > best[k][0]:
            best[k] = (score, a)
    return sorted(best.values(), key=lambda x: -x[0])[:cfg.max_anchor_actions]


def _join_candidates(s: State, ctx: EpisodeContext, cfg: ActionConfig) -> list[Action]:
    visited = set(s.visited)
    episode = set(ctx.tables)
    out = []
    if cfg.edges == "declared":
        cands = [(e, 1.0) for e in ctx.schema.fks]
    else:
        cands = [(c.edge, c.omega) for c in ctx.index.edges if c.omega >= cfg.edge_threshold]
    for edge, score in cands:
        t1, t2 = (c.split(".", 1)[0] for c in edge[0])
        if t1 in visited and t2 not in visited and t2 in episode:
            out.append((score, Action("JOIN", t2, edge=edge)))
        elif t2 in visited and t1 not in visited and t1 in episode:
            out.append((score, Action("JOIN", t1, edge=edge)))
    out.sort(key=lambda x: -x[0])
    return [a for _, a in out[:cfg.max_join_actions]]


def enumerate_actions(s: State, ctx: EpisodeContext, cfg: ActionConfig) -> list[Action]:
    """Legal candidate actions in state `s`, without look-ahead counts (the env adds those)."""
    if s.t >= cfg.horizon - 1:
        return [Action.stop()]
    if s.t == 0:
        seeds = sorted(ctx.tables, key=lambda t: -ctx.table_scores.get(t, 0.0))[:cfg.max_seed]
        anchors = _anchor_candidates(s, ctx, cfg, set(ctx.tables))
        return [Action("SEED", t) for t in seeds] + [a for _, a in anchors]
    acts = []
    if len(s.anchors) < cfg.max_anchors:
        acts += [a for _, a in _anchor_candidates(s, ctx, cfg, set(s.visited))]
    acts += _join_candidates(s, ctx, cfg)
    return acts + [Action.stop()]
