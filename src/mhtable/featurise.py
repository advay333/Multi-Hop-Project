"""
State and candidate actions -> feature vectors for the policy (report C.3, DESIGN.md §6.5).

Everything is built from the question, the schema, the index and the environment's look-ahead
counts — never from gold (firewall: `assert_gold_free` on every input; this module imports
neither the oracle nor `GoldRecord`). No database-specific ids: tables, columns and values are
represented by the embeddings of their text, so a policy transfers to unseen databases.

A feature vector is a fixed number of embedding *slots* followed by scalars:

    state   [ question | mean of visited tables | mean of applied filters | 6 scalars ]
    action  [ what it acts on | filter value | target table | 25 scalars ]

Sets (visited tables, filters, the columns of a composite join) are mean-pooled, so the result
does not depend on their order. Scalars are scaled to about [0, 1]; counts go through log1p.
The policy normalises embeddings and scalars separately (PLAN.md §6 rule 7).

Texts looked up in the embedding cache (the P4 cache builder embeds exactly these, via
`required_texts`): the question (+ BIRD evidence), `table: col1, col2, ...` per table,
`table column` per column, and cell values. Filter values that are not in the cache (numbers
taken from the question, LIKE patterns) get a zero vector and a flag.

`header_ablation` replaces table and column names by `table_i` / `col_j` and zeroes the
name-similarity scalars (robustness study, PLAN.md §6 rule 10). Candidate generation in the
environment still reads the real names; the ablation covers the policy's inputs only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mhtable.encoders.cache import EmbCache
from mhtable.firewall import assert_gold_free
from mhtable.types import Action, Instance, State

KINDS = ("SEED", "ANCHOR", "JOIN", "STOP")
OP_CLASS = {"=": 0, "<": 1, ">": 1, "<=": 1, ">=": 1, "between": 1, "!=": 2, "not_like": 2, "not_in": 2,
            "not_is": 2, "like": 3, "in": 4, "is": 4}
N_OP = 5
STATE_SLOTS, STATE_SCALARS = 3, 6
ACTION_SLOTS, ACTION_SCALARS = 3, 25
MAX_VALUE_CHARS = 64


@dataclass
class FeatConfig:
    header_ablation: bool = False
    log_cap: float = 100_000.0  # row counts are capped here by the environment's look-ahead
    max_token_ratio: float = 4.0  # tokens / budget is clipped here


# ---------------------------------------------------------------- texts (shared with the cache builder)
def question_text(inst: Instance) -> str:
    return inst.question + (f" ({inst.evidence})" if inst.evidence else "")


def table_text(schema, table: str, ablate: bool = False) -> str:
    t = table.lower()
    cols = schema.columns.get(t, [])
    if ablate:
        return f"table_{sorted(schema.columns).index(t)}: " + ", ".join(f"col_{j}" for j in range(len(cols)))
    return f"{schema.table_names.get(t, t)}: " + ", ".join(cols)


def column_text(schema, key: str, ablate: bool = False) -> str:
    t, c = key.split(".", 1)
    cols = schema.columns.get(t, [])
    j = next((i for i, x in enumerate(cols) if x.lower() == c), -1)
    if ablate:
        return f"table_{sorted(schema.columns).index(t) if t in schema.columns else 0} col_{j}"
    return f"{schema.table_names.get(t, t)} {cols[j] if j >= 0 else c}"


def value_text(v) -> str:
    return str(v).strip("%")[:MAX_VALUE_CHARS]


def required_texts(inst: Instance, schema, index, ablate: bool = False) -> list[str]:
    """Every text the featuriser may look up for episodes of `inst` (for the cache builder)."""
    out = [question_text(inst)]
    for t in sorted(schema.columns):
        out.append(table_text(schema, t, ablate))
        out += [column_text(schema, f"{t}.{c.lower()}", ablate) for c in schema.columns[t]]
    for vals in index.cells.values():
        out += [value_text(v) for v in vals]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- featuriser
class Featuriser:
    def __init__(self, env, cache: EmbCache, cfg: FeatConfig | None = None):
        self.env = env
        self.cache = cache
        self.cfg = cfg or FeatConfig()
        self.d_emb = cache.dim
        self._zero = np.zeros(self.d_emb, dtype=np.float32)
        self._log_cap = math.log1p(self.cfg.log_cap)

    @property
    def spec(self) -> dict:
        """What a policy needs to know about the feature layout."""
        return {"d_emb": self.d_emb, "state_slots": STATE_SLOTS, "state_scalars": STATE_SCALARS,
                "action_slots": ACTION_SLOTS, "action_scalars": ACTION_SCALARS}

    @property
    def state_dim(self) -> int:
        return STATE_SLOTS * self.d_emb + STATE_SCALARS

    @property
    def action_dim(self) -> int:
        return ACTION_SLOTS * self.d_emb + ACTION_SCALARS

    # ------------------------------------------------------------ pieces
    def _rows(self, n: int) -> float:
        return math.log1p(min(max(n, 0), self.cfg.log_cap)) / self._log_cap

    def _tokens(self, tok: int) -> float:
        r = max(tok, 0) / max(self.env.cfg.budget_tokens, 1)
        return min(r, self.cfg.max_token_ratio) / self.cfg.max_token_ratio

    def _mean(self, vecs: list[np.ndarray]) -> np.ndarray:
        return np.mean(vecs, axis=0) if vecs else self._zero

    def _table(self, schema, t: str) -> np.ndarray:
        return self.cache.get(table_text(schema, t, self.cfg.header_ablation))

    def _column(self, schema, key: str) -> np.ndarray:
        return self.cache.get(column_text(schema, key, self.cfg.header_ablation))

    def _values(self, values: tuple) -> tuple[np.ndarray, bool]:
        found = [v for v in (self.cache.get(value_text(x), None) for x in values if x is not None) if v is not None]
        return self._mean(found), bool(found)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(a @ b) / (na * nb) if na > 0 and nb > 0 else 0.0

    # ------------------------------------------------------------ state
    def state_features(self, s: State) -> np.ndarray:
        assert_gold_free(s)
        ctx = self.env._context(s)
        schema = ctx.schema
        q = self.cache.get(question_text(s.instance))
        tables = self._mean([self._table(schema, t) for t in s.visited])
        filt = []
        for a in s.anchors:
            filt.append(self._column(schema, a.column))
            v, ok = self._values(a.values)
            if ok:
                filt.append(v)
        h = self.env.cfg.actions.horizon
        scalars = np.array([s.t / h, self._rows(s.n_rows), self._tokens(s.est_tokens), len(s.visited) / h,
                            len(s.anchors) / 3.0, float(s.n_rows < 0)], dtype=np.float32)
        return np.concatenate([q, tables, self._mean(filt), scalars]).astype(np.float32)

    # ------------------------------------------------------------ actions
    def action_features(self, s: State, actions: list[Action]) -> np.ndarray:
        assert_gold_free(s)
        ctx = self.env._context(s)
        schema, ablate = ctx.schema, self.cfg.header_ablation
        q = self.cache.get(question_text(s.instance))
        edge_stats = {c.edge: c for c in ctx.index.edges}
        out = np.zeros((len(actions), self.action_dim), dtype=np.float32)
        d = self.d_emb
        for i, a in enumerate(actions):
            assert_gold_free(a)
            what, value, target = self._zero, self._zero, self._zero
            sc = np.zeros(ACTION_SCALARS, dtype=np.float32)
            sc[KINDS.index(a.kind)] = 1.0
            col_score = 0.0
            if a.kind != "STOP":
                target = self._table(schema, a.table)
                sc[11] = 0.0 if ablate else ctx.table_scores.get(a.table, 0.0)
            if a.kind == "SEED":
                what = target
            elif a.kind == "ANCHOR":
                what = self._column(schema, a.anchor.column)
                value, has_value = self._values(a.anchor.values)
                sc[4 + OP_CLASS.get(a.anchor.op, 4)] = 1.0
                col_score = ctx.column_scores.get(a.anchor.column, 0.0)
                sc[23] = float(has_value)
                sc[24] = float(any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in a.anchor.values))
            elif a.kind == "JOIN":
                cols = [c for pair in a.edge for c in pair]
                what = self._mean([self._column(schema, c) for c in cols])
                col_score = max((ctx.column_scores.get(c, 0.0) for c in cols), default=0.0)
                st = edge_stats.get(a.edge)
                if st is not None:
                    sc[13:17] = (min(st.omega, 2.0) / 2.0, st.e, st.j, st.u)
                else:  # a declared or composite edge that the index did not score
                    sc[17] = 1.0
                sc[18] = float(len(a.edge) > 1)
            sc[9] = self._cos(q, what)
            sc[10] = self._cos(q, target)
            sc[12] = 0.0 if ablate else col_score
            sc[19] = self._rows(a.lookahead_rows)
            sc[20] = self._tokens(a.lookahead_tokens)
            sc[21] = float(a.lookahead_rows < 0)
            sc[22] = self._rows(a.lookahead_rows) - self._rows(s.n_rows)
            out[i, :d], out[i, d:2 * d], out[i, 2 * d:3 * d], out[i, 3 * d:] = what, value, target, sc
        return out

    def __call__(self, s: State, actions: list[Action]) -> tuple[np.ndarray, np.ndarray]:
        return self.state_features(s), self.action_features(s, actions)
