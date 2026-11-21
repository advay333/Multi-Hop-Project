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

import heapq
import json
import sqlite3
import time
from collections import Counter
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
    fetch_timeout_s: float = 30.0  # materialise's row fetch; a timeout gives an empty sub-table
    lookahead_memo: int = 500_000  # memoised look-ahead counts per env (cleared when full); 0 = off


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
    timed_out: bool = False  # the row fetch hit fetch_timeout_s (sub-table left empty)


class TableEnv:
    def __init__(self, cfg: EnvConfig | None = None, data: EnvData | None = None, sim: Similarity | None = None):
        self.cfg = cfg or EnvConfig()
        self.data = data or EnvData()
        self.sim = sim or LexicalSimilarity()
        self.counter = make_counter(self.cfg.tokenizer, self.cfg.chars_per_token)
        self._ctx: dict[str, EpisodeContext] = {}
        self._con: dict[str, object] = {}
        self._count_memo: dict[tuple[str, str], int] = {}
        self.stats: Counter = Counter()  # look-ahead / materialise timings (analysis.speed reads them)
        self.slow_queries: list[tuple[float, str, str]] = []  # (seconds, db_ref, sql), the 10 slowest look-aheads

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
        key = (s.instance.db_ref, sql)
        if key in self._count_memo:  # the count depends only on the relation, not the question
            self.stats["lookahead_memo_hits"] += 1
            return self._count_memo[key]
        t0 = time.perf_counter()
        try:
            n = run(self._conn(s.instance.db_ref), sql, self.cfg.lookahead_timeout_s)[0][0]
        except QueryTimeout:
            n = -1
            self.stats["lookahead_timeouts"] += 1
        dt = time.perf_counter() - t0
        self.stats["lookahead_queries"] += 1
        self.stats["lookahead_seconds"] += dt
        item = (dt, s.instance.db_ref, sql)
        if len(self.slow_queries) < 10:
            heapq.heappush(self.slow_queries, item)
        elif dt > self.slow_queries[0][0]:
            heapq.heapreplace(self.slow_queries, item)
        if self.cfg.lookahead_memo:
            if len(self._count_memo) >= self.cfg.lookahead_memo:
                self._count_memo.clear()
            self._count_memo[key] = n
        return n

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
            t0 = time.perf_counter()
            n = self._relation_count(nxt, ctx.schema)
            self.stats[f"lookahead_{a.kind}_seconds"] += time.perf_counter() - t0
            self.stats[f"lookahead_{a.kind}_actions"] += 1
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

    def materialise_tables(self, inst: Instance, order: str = "rowid", budget: int | None = None,
                           seed: int = 0) -> Materialised:
        """Whole tables, no selection — the P6 baselines (report E.5):
            order="rowid"   B1 truncate: every episode table with all its columns, in schema
                            order, rows in rowid order, the whole text cut at the budget
            order="random"  B2 random rows: each table gets an equal share of the budget, filled
                            with uniformly sampled rows (seeded by `seed` and the question id)
        Same token counter as `materialise`; at most `max_fetch_rows` rows are read per table."""
        import random

        budget = budget or self.cfg.budget_tokens
        if order not in ("rowid", "random"):
            raise ValueError(f"order must be rowid or random, got {order!r}")
        schema = self.data.schema(inst.db_ref)
        con = self._conn(inst.db_ref)
        tables = [t.lower() for t in (inst.tables or sorted(schema.columns)) if t.lower() in schema.columns]
        share = budget / max(len(tables), 1)
        text, used, shown, total, cols_out, overflow = "", 0.0, 0, 0, [], False
        for t in tables:
            keys = [f"{t}.{c.lower()}" for c in schema.columns[t]]
            cols_out += keys
            sel = ", ".join(column_sql(schema, k) for k in keys)
            rows = run(con, f"SELECT {sel} FROM {table_sql(schema, t)}", 30.0, self.cfg.max_fetch_rows + 1)
            total += len(rows)
            if order == "random":
                rows = list(rows)
                random.Random(f"{seed}:{inst.instance_id}:{t}").shuffle(rows)
            head = (f"Table: {schema.table_names.get(t, t)}\n| " + " | ".join(schema.columns[t]) + " |\n|"
                    + "---|" * len(keys))
            block_cost = self.counter.count(("\n\n" if text else "") + head)
            limit = budget if order == "rowid" else used + share
            if used + block_cost > limit:
                overflow = True
                if order == "rowid":
                    break
                continue
            text += ("\n\n" if text else "") + head
            used += block_cost
            lines = ["| " + " | ".join("" if v is None else str(v).replace("|", "/").replace("\n", " ") for v in r) + " |"
                     for r in rows]
            for line, cost in zip(lines, self.counter.count_many(["\n" + x for x in lines])):
                if used + cost > limit:
                    overflow = True
                    break
                text += "\n" + line
                used += cost
                shown += 1
            if overflow and order == "rowid":
                break
        if self.counter.exact:
            used = self.counter.count(text)
            while used > budget and "\n" in text:
                text = text.rsplit("\n", 1)[0]
                used = self.counter.count(text)
        return Materialised(text, int(used), total, shown, cols_out, overflow)

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
        rows, timed_out = self._fetch(con, s, sel, body, order, len(cols))
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
        return Materialised(text, int(used), max(total, len(rows)), shown, cols, shown < len(rows), timed_out)

    def _fetch(self, con, s: State, sel: str, body: str, order: str, n_cols: int) -> tuple[list, bool]:
        """Rows of the sub-table, at most max_fetch_rows + 1, in a deterministic order.

        Relations below the look-ahead cap are sorted by the visited tables' rowids (small and
        fast; the order Track C used). Huge or unknown-size relations (n_rows at the cap or -1)
        are read in SQLite's scan order with a SQL LIMIT, so SQLite stops early instead of sorting
        millions of joined rows — deterministic for a fixed DB file, its indexes and SQLite build.
        A timeout gives an empty sub-table (timed_out=True) instead of an exception."""
        limit = self.cfg.max_fetch_rows + 1
        t0 = time.perf_counter()
        try:
            if 0 <= s.n_rows < self.cfg.lookahead_cap:
                try:
                    rows = run(con, f"SELECT {sel} {body} ORDER BY {order}", self.cfg.fetch_timeout_s, limit)
                except sqlite3.OperationalError:  # WITHOUT ROWID tables have no rowid: order by the shown columns
                    rows = run(con, f"SELECT {sel} {body} ORDER BY {', '.join(str(i + 1) for i in range(n_cols))}",
                               self.cfg.fetch_timeout_s, limit)
            else:
                self.stats["materialise_unordered"] += 1
                rows = run(con, f"SELECT {sel} {body} LIMIT {limit}", self.cfg.fetch_timeout_s)
            return rows, False
        except QueryTimeout:
            self.stats["materialise_timeouts"] += 1
            return [], True
        finally:
            self.stats["materialise_calls"] += 1
            self.stats["materialise_seconds"] += time.perf_counter() - t0
