"""
Heuristic selector (PLAN.md P6: "greedy cosine + top-k") — a non-learned actor in the policy's
action space, so it shares the rollout, materialiser, reader and budget with every other selector.

Per question it embeds the question and every episode table (the same embedding cache the policy
uses — bge-m3 in real runs) and keeps the relevant tables R: those within `delta` cosine of the
best table, at most `max_tables` (MMQA questions use 2–3 tables). Then:

    t = 0   take the best ANCHOR on a table of R — score cos(question, column) + cos(question,
            value) minus a small penalty for the environment's own rank — or, if no anchor is
            offered on R, SEED the best table of R;
    t >= 1  JOIN the unvisited table of R with the highest cosine, while one is offered;
            otherwise STOP.

No gold, no learning; at most one filter (the start). It is the comparison point for the BC gate
("BC beats the heuristic on dev join-path accuracy").
"""

from __future__ import annotations

import numpy as np

from mhtable.featurise import column_text, question_text, table_text, value_text
from mhtable.registry import register
from mhtable.rollout import Decision


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else 0.0


@register("selector", "heuristic")
class HeuristicSelector:
    def __init__(self, env, cache, delta: float = 0.1, max_tables: int = 3, rank_penalty: float = 0.02):
        self.env, self.cache = env, cache
        self.delta, self.max_tables, self.rank_penalty = delta, max_tables, rank_penalty
        self._rel: dict[str, tuple] = {}

    def _relevant(self, s):
        iid = s.instance.instance_id
        if iid not in self._rel:
            schema = self.env._context(s).schema
            q = self.cache.get(question_text(s.instance))
            score = {t: _cos(q, self.cache.get(table_text(schema, t))) for t in self.env._context(s).tables}
            ranked = sorted(score, key=lambda t: -score[t])
            best = score[ranked[0]] if ranked else 0.0
            rel = [t for t in ranked if score[t] >= best - self.delta][:self.max_tables]
            self._rel[iid] = (q, score, rel)
        return self._rel[iid]

    def _anchor_score(self, s, q, a, rank: int) -> float:
        schema = self.env._context(s).schema
        col = _cos(q, self.cache.get(column_text(schema, a.anchor.column)))
        vals = [self.cache.get(value_text(v), None) for v in a.anchor.values if v is not None]
        val = max((_cos(q, v) for v in vals if v is not None), default=0.0)
        return col + val - self.rank_penalty * rank

    def act(self, states, cands, mode="argmax", generator=None):
        out = []
        for s, c in zip(states, cands):
            q, score, rel = self._relevant(s)
            stop = next((i for i, a in enumerate(c) if a.kind == "STOP"), 0)
            if s.t == 0:
                anchors = [(i, a) for i, a in enumerate(c) if a.kind == "ANCHOR" and a.table in rel]
                if anchors:
                    idx = max(((i, self._anchor_score(s, q, a, r)) for r, (i, a) in enumerate(anchors)),
                              key=lambda x: x[1])[0]
                else:
                    seeds = [(i, a) for i, a in enumerate(c) if a.kind == "SEED"]
                    pref = [x for x in seeds if x[1].table in rel] or seeds
                    idx = max(pref, key=lambda x: score.get(x[1].table, 0.0))[0] if pref else 0
            else:
                joins = [(i, a) for i, a in enumerate(c) if a.kind == "JOIN" and a.table in rel
                         and a.table not in s.visited]
                idx = max(joins, key=lambda x: score.get(x[1].table, 0.0))[0] if joins else stop
            out.append(Decision(idx))
        return out
