"""
TableEnv: the deterministic, gold-free MDP environment (report C.3, DESIGN.md §6.4).

    env = TableEnv(EnvConfig(), EnvData(data_dir, index_dir))
    s = env.reset(instance)
    while True:
        acts = env.candidate_actions(s)          # legal actions with look-ahead counts
        s, done = env.step(s, acts[i])
        if done: break
    m = env.materialise(s)                       # sub-table for the reader, <= budget tokens

Transitions execute in SQLite via `sqlrender` (shared with the oracle). Every candidate carries
a look-ahead row count — a capped COUNT(*) of the relation it would produce — which is
cardinality estimation, as in a query optimiser, and uses no gold information.
This module never imports the oracle and never reads `gold/` (firewall, DESIGN.md §8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from mhtable.data.execute import QueryTimeout, connect_ro, run
from mhtable.data.schema import SchemaGraph
from mhtable.env.actions import ActionConfig, EpisodeContext, enumerate_actions
from mhtable.env.index import CellSearch, DbIndex
from mhtable.env.text import LexicalSimilarity, Similarity
from mhtable.env.tokens import make_counter
from mhtable.sqlrender import column_sql, from_where, table_sql
from mhtable.types import Action, Instance, State


@dataclass
class EnvConfig:
    actions: ActionConfig = field(default_factory=ActionConfig)
    budget_tokens: int = 4096
    chars_per_token: float = 3.5  # look-ahead token estimate; also the budget count when tokenizer is None
    tokenizer: str | None = None  # model whose tokenizer counts the sub-table budget (env/tokens.py)
    lookahead_cap: int = 100_000  # COUNT(*) is capped here (counts above are reported as the cap)
    lookahead_timeout_s: float = 2.0  # a look-ahead that times out reports -1 (unknown)
    project_top_columns: int = 4  # minimum extra question-relevant columns shown besides path columns
    project_all_columns: bool = True  # show every other column of visited tables while all rows fit B
    max_fetch_rows: int = 5000


class EnvData:
    """Schemas, indexes and DB files, from a local copy of the data repo or from HF."""

    def __init__(self, data_dir: str | Path | None = None, index_dir: str | Path | None = None,
                 revision: str | None = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.index_dir = Path(index_dir) if index_dir else self.data_dir
        self.revision = revision
        self._schemas: dict[str, dict] = {}
        self._index: dict[str, tuple[DbIndex, CellSearch]] = {}

    def _file(self, base: Path | None, rel: str) -> Path:
        if base is not None and (base / rel).exists():
            return base / rel
        from mhtable import storage

        return storage.fetch("data", rel, revision=self.revision)

    def schema(self, db_ref: str) -> SchemaGraph:
        src = db_ref.split("/", 1)[0]
        if src not in self._schemas:
            self._schemas[src] = json.loads(self._file(self.data_dir, f"schemas/{src}.json").read_text(encoding="utf-8"))
        return SchemaGraph.from_dict(self._schemas[src][db_ref])

    def index(self, db_ref: str) -> tuple[DbIndex, CellSearch]:
        if db_ref not in self._index:
            d = json.loads(self._file(self.index_dir, f"index/{db_ref}.json").read_text(encoding="utf-8"))
            idx = DbIndex.from_dict(d)
            self._index[db_ref] = (idx, CellSearch(idx))
        return self._index[db_ref]

    def db_file(self, db_ref: str) -> Path:
        return self._file(self.data_dir, f"db/{db_ref}.sqlite")


@dataclass
class Materialised:
    text: str
    tokens: int
    rows_total: int
    rows_shown: int
    columns: list[str]
    overflow: bool  # True if rows were cut to fit the budget


class TableEnv:
    def __init__(self, cfg: EnvConfig | None = None, data: EnvData | None = None, sim: Similarity | None = None):
        self.cfg = cfg or EnvConfig()
        self.data = data or EnvData()
        self.sim = sim or LexicalSimilarity()
        self.counter = make_counter(self.cfg.tokenizer, self.cfg.chars_per_token)
        self._ctx: dict[str, EpisodeContext] = {}
        self._con: dict[str, object] = {}

    # ------------------------------------------------------------------ episode setup
    def reset(self, inst: Instance, tables: list[str] | None = None) -> State:
        """Start an episode. `tables` overrides the episode's tables (e.g. gold + k distractors,
        chosen by the caller); default: every table of the instance's database."""
        schema = self.data.schema(inst.db_ref)
        index, search = self.data.index(inst.db_ref)
        query = inst.question + (f" ({inst.evidence})" if inst.evidence else "")
        ctx = EpisodeContext.build(schema, index, search, query, tables or inst.tables, self.sim)
        self._ctx[inst.instance_id] = ctx
        return State(instance=inst)

    def _context(self, s: State) -> EpisodeContext:
        return self._ctx[s.instance.instance_id]

    def _conn(self, db_ref: str):
        if db_ref not in self._con:
            self._con[db_ref] = connect_ro(self.data.db_file(db_ref))
        return self._con[db_ref]

    def close(self) -> None:
        for con in self._con.values():
            con.close()
        self._con.clear()

    # ------------------------------------------------------------------ transitions
    @staticmethod
    def apply(s: State, a: Action) -> State:
        """The next state's structure (no SQL executed)."""
        visited, joins, anchors = list(s.visited), list(s.joins), list(s.anchors)
        if a.kind in ("SEED", "ANCHOR") and not visited:
            visited.append(a.table)
        if a.kind == "JOIN":
            visited.append(a.table)
            joins.append(a.edge)
        if a.kind == "ANCHOR":
            anchors.append(a.anchor)
        return replace(s, t=s.t + 1, visited=visited, joins=joins, anchors=anchors)

    def _relation_count(self, s: State, schema: SchemaGraph) -> int:
        if not s.visited:
            return 0
        sql = (f"SELECT COUNT(*) FROM (SELECT 1 {from_where(schema, s.visited, s.joins, s.anchors)} "
               f"LIMIT {self.cfg.lookahead_cap})")
        try:
            return run(self._conn(s.instance.db_ref), sql, self.cfg.lookahead_timeout_s)[0][0]
        except QueryTimeout:
            return -1

    def _cols_tokens(self, cols: list[str], ctx: EpisodeContext) -> float:
        chars = sum(ctx.index.columns[c].avg_chars + 3 if c in ctx.index.columns else 8 for c in cols)
        return chars / self.cfg.chars_per_token

    def _row_tokens(self, s: State, ctx: EpisodeContext, n_rows: int | None = None) -> float:
        return self._cols_tokens(self.projection(s, ctx, n_rows), ctx)

    def candidate_actions(self, s: State) -> list[Action]:
        """Legal actions in `s`, each with look-ahead rows/tokens of the state it leads to."""
        ctx = self._context(s)
        out = []
        for a in enumerate_actions(s, ctx, self.cfg.actions):
            if a.kind == "STOP":
                out.append(Action("STOP", lookahead_rows=s.n_rows, lookahead_tokens=s.est_tokens))
                continue
            nxt = self.apply(s, a)
            n = self._relation_count(nxt, ctx.schema)
            tok = int(max(n, 0) * self._row_tokens(nxt, ctx, n))
            out.append(Action(a.kind, a.table, a.anchor, a.edge, lookahead_rows=n, lookahead_tokens=tok))
        return out

    def step(self, s: State, a: Action) -> tuple[State, bool]:
        if a.kind == "STOP":
            return replace(s, t=s.t + 1), True
        ctx = self._context(s)
        nxt = self.apply(s, a)
        n = a.lookahead_rows if a.lookahead_rows else self._relation_count(nxt, ctx.schema)
        nxt.n_rows = n
        nxt.est_tokens = int(max(n, 0) * self._row_tokens(nxt, ctx, n))
        nxt.relation_sql = from_where(ctx.schema, nxt.visited, nxt.joins, nxt.anchors)
        return nxt, nxt.t >= self.cfg.actions.horizon

    # ------------------------------------------------------------------ materialisation
    def projection(self, s: State, ctx: EpisodeContext | None = None, n_rows: int | None = None) -> list[str]:
        """Columns shown to the reader (report C.3): join keys and anchor columns, plus the other
        columns of the visited tables ranked by question relevance. With `project_all_columns`
        (default) as many of those are kept as let all `n_rows` rows fit the budget (estimated),
        never fewer than `project_top_columns`; otherwise exactly the top `project_top_columns`.
        Showing every column the budget allows keeps answer columns whose names do not match the
        question words (e.g. `people.name` for "names of entrepreneurs")."""
        ctx = ctx or self._context(s)
        cols = []
        for e in s.joins:
            for p in e:
                cols += list(p)
        cols += [a.column for a in s.anchors]
        extra = [f"{t}.{c.lower()}" for t in s.visited for c in ctx.schema.columns.get(t, [])]
        extra = [k for k in extra if k not in cols]
        extra.sort(key=lambda k: -ctx.column_scores.get(k, 0.0))  # stable: ties keep schema order
        k = min(self.cfg.project_top_columns, len(extra))
        if self.cfg.project_all_columns:
            if n_rows is None or n_rows < 0:
                n_rows = s.n_rows
            if n_rows >= 0:
                k = len(extra)
                while k > self.cfg.project_top_columns and \
                        n_rows * self._cols_tokens(cols + extra[:k], ctx) > self.cfg.budget_tokens:
                    k -= 1
        cols += extra[:k]
        seen, out = set(), []
        for t in s.visited:  # table visit order, then schema column order
            for c in ctx.schema.columns.get(t, []):
                k = f"{t}.{c.lower()}"
                if k in cols and k not in seen:
                    seen.add(k)
                    out.append(k)
        return out

    def materialise(self, s: State, budget: int | None = None) -> Materialised:
        budget = budget or self.cfg.budget_tokens
        ctx = self._context(s)
        if not s.visited:
            return Materialised("", 0, 0, 0, [], False)
        schema = ctx.schema
        cols = self.projection(s, ctx, s.n_rows)
        header = "| " + " | ".join(column_sql(schema, k).replace('"', "") for k in cols) + " |"
        lines = [header, "|" + "---|" * len(cols)]
        order = ", ".join(f"{table_sql(schema, t)}.rowid" for t in s.visited)
        body = from_where(schema, s.visited, s.joins, s.anchors)
        sel = ", ".join(column_sql(schema, k) for k in cols)
        con = self._conn(s.instance.db_ref)
        try:
            rows = run(con, f"SELECT {sel} {body} ORDER BY {order}", 30.0, self.cfg.max_fetch_rows + 1)
        except Exception:  # e.g. WITHOUT ROWID tables: fall back to ordering by the shown columns
            rows = run(con, f"SELECT {sel} {body} ORDER BY {', '.join(str(i + 1) for i in range(len(cols)))}",
                       30.0, self.cfg.max_fetch_rows + 1)
        total = s.n_rows if s.n_rows >= 0 else len(rows)
        text = "\n".join(lines)
        used = self.counter.count(text)
        shown, block, full = 0, 256, False
        for k in range(0, len(rows), block):  # count in blocks: the budget usually fills early
            part = ["| " + " | ".join("" if v is None else str(v).replace("|", "/").replace("\n", " ") for v in r) + " |"
                    for r in rows[k:k + block]]
            for line, cost in zip(part, self.counter.count_many(["\n" + x for x in part])):
                if used + cost > budget:
                    full = True
                    break
                text += "\n" + line
                used += cost
                shown += 1
            if full:
                break
        if self.counter.exact:  # per-line counts can differ from the joined text by a token at a seam
            used = self.counter.count(text)
            while used > budget and shown > 0:
                text = text.rsplit("\n", 1)[0]
                shown -= 1
                used = self.counter.count(text)
        return Materialised(text, int(used), max(total, len(rows)), shown, cols, shown < len(rows))
