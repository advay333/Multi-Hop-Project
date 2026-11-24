"""
The one rollout collector, shared by training and every evaluation mode (PLAN.md §6 rule 5,
DESIGN.md §6.6). Behaviour differs only through flags:

    collect(env, actor, reader, instances,
            mode="sample" | "argmax",
            oracle_mode="reward" | "metric_only" | "off", golds={instance_id: GoldRecord},
            group_size=G,                 # G rollouts per question (GRPO, Monte-Carlo values)
            start_after_gold_steps=k)     # curriculum: start after k gold steps

All live episodes are stepped in lockstep (one batched policy forward per step); the reader is
called once, batched, at the end. An empty sub-table is not sent to the reader (it cannot be
answered and would waste a call); its answer is "".

`actor` is anything with
    act(states, candidate_lists, mode, generator) -> list[Decision]
`PolicyActor` wraps a policy network + featuriser; the P6 selectors (baselines, oracle)
implement the same method.

oracle_mode:
  reward       training: per-step rewards are written into the steps (needs `golds`)
  metric_only  evaluation: gold is used only for diagnostics in `traj.info`
  off          no gold at all; the mock reader (which needs gold) is refused
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from mhtable.featurise import Featuriser, question_text
from mhtable.metrics import answer_in_subtable, em_norm, item_counts, set_f1
from mhtable.readers.base import CachedReader, Reader
from mhtable.rewards import RewardConfig, assemble
from mhtable.sufficiency import entity_overlap, path_match, sufficient
from mhtable.types import Action, GoldRecord, Instance, State, Step, Trajectory

MODES = ("sample", "argmax")
ORACLE_MODES = ("reward", "metric_only", "off")


@dataclass
class Decision:
    action_idx: int
    logp: float = 0.0
    value: float = 0.0
    state_feat: np.ndarray | None = None
    action_feats: np.ndarray | None = None


def pad_actions(feats: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """[A_i, d] arrays -> padded [B, A_max, d] tensor and a [B, A_max] mask (True = real candidate)."""
    a_max, d = max(f.shape[0] for f in feats), feats[0].shape[1]
    dtype = torch.float16 if feats[0].dtype == np.float16 else torch.float32  # BC stores float16
    out = torch.zeros(len(feats), a_max, d, dtype=dtype)
    mask = torch.zeros(len(feats), a_max, dtype=torch.bool)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = torch.from_numpy(f)
        mask[i, : f.shape[0]] = True
    return out, mask


class PolicyActor:
    """A policy network acting through the featuriser."""

    def __init__(self, policy, featuriser: Featuriser):
        self.policy = policy
        self.featuriser = featuriser

    @torch.no_grad()
    def act(self, states: list[State], cands: list[list[Action]], mode: str,
            generator: torch.Generator | None = None) -> list[Decision]:
        feats = [self.featuriser(s, c) for s, c in zip(states, cands)]
        device = next(self.policy.parameters()).device
        state = torch.from_numpy(np.stack([f[0] for f in feats])).to(device)
        actions, mask = pad_actions([f[1] for f in feats])
        was_training = self.policy.training
        self.policy.eval()
        logits, value = self.policy(state, actions.to(device), mask.to(device))
        self.policy.train(was_training)
        logp, value = torch.log_softmax(logits, dim=-1).cpu(), value.cpu()  # sampling uses the CPU generator
        if mode == "argmax":
            idx = logp.argmax(dim=-1)
        else:
            idx = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        chosen = logp.gather(1, idx.unsqueeze(1)).squeeze(1)
        return [Decision(int(idx[i]), float(chosen[i]), float(value[i]), feats[i][0], feats[i][1])
                for i in range(len(states))]


def _candidates(env, s: State, cache: dict) -> list[Action]:
    """Candidate actions of `s`, memoised on (question, tier, state structure). The environment
    is deterministic, so rollouts that pass through the same state — the members of a group, or
    the same question in a later iteration — share one enumeration and its look-ahead counts."""
    key = (s.instance.instance_id, env.cfg.actions.tier, s.t, tuple(s.visited), tuple(s.joins),
           tuple(a.key() for a in s.anchors))
    if key not in cache:
        cache[key] = env.candidate_actions(s)
    return cache[key]


def _gold_prefix(env, s: State, gold: GoldRecord, k: int, cache: dict) -> State:
    """Follow up to k gold actions (never STOP); stop early if one is not offered."""
    for g in gold.gold_trajectory[:k]:
        if g.kind == "STOP":
            break
        match = next((a for a in _candidates(env, s, cache) if a.key() == g.key()), None)
        if match is None:
            break
        s, done = env.step(s, match)
        if done:
            break
    return s


def collect(env, actor, reader: Reader | CachedReader, instances: list[Instance], *,
            mode: str = "sample", oracle_mode: str = "off",
            golds: Mapping[str, GoldRecord] | None = None, group_size: int = 1,
            start_after_gold_steps: int = 0, reward_cfg: RewardConfig | None = None,
            generator: torch.Generator | None = None, cand_cache: dict | None = None,
            detail: bool = False) -> list[Trajectory]:
    """Run `group_size` episodes per instance. Trajectories come back grouped: the rollouts of
    instances[i] are at positions i*group_size .. (i+1)*group_size - 1. `cand_cache` (a dict the
    caller keeps) lets candidate enumerations be reused across calls; without it they are
    shared only inside this call. `detail=True` (needs gold) adds the full metric suite to
    `traj.info`: set_f1, answer_in_subtable, join-edge / filter counts, per-hop entity overlap."""
    cand_cache = {} if cand_cache is None else cand_cache
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if oracle_mode not in ORACLE_MODES:
        raise ValueError(f"oracle_mode must be one of {ORACLE_MODES}, got {oracle_mode!r}")
    use_gold = oracle_mode != "off"
    if use_gold and golds is None:
        raise ValueError(f"oracle_mode={oracle_mode!r} needs `golds`")
    if not use_gold and (golds is not None or getattr(reader, "needs_gold", False)):
        raise ValueError("oracle_mode='off' takes no gold: pass no `golds` and no mock reader")
    if start_after_gold_steps and oracle_mode != "reward":
        raise ValueError("the gold-prefix curriculum is a training device (oracle_mode='reward')")
    if not isinstance(reader, CachedReader):
        reader = CachedReader(reader)
    reward_cfg = reward_cfg or RewardConfig()
    budget = env.cfg.budget_tokens

    # ---- start every episode
    starts: dict[str, State] = {}
    states: list[list[State]] = []
    trajs: list[Trajectory] = []
    for inst in instances:
        if inst.instance_id not in starts:
            s = env.reset(inst)
            if start_after_gold_steps:
                s = _gold_prefix(env, s, golds[inst.instance_id], start_after_gold_steps, cand_cache)
            starts[inst.instance_id] = s
        for g in range(group_size):
            states.append([starts[inst.instance_id]])
            trajs.append(Trajectory(inst.instance_id, info={"group_member": g, "start_t": starts[inst.instance_id].t}))

    # ---- step all live episodes together
    live = list(range(len(trajs)))
    while live:
        cur = [states[i][-1] for i in live]
        cands = [_candidates(env, s, cand_cache) for s in cur]
        decisions = actor.act(cur, cands, mode, generator)
        still = []
        for i, s, c, d in zip(live, cur, cands, decisions):
            nxt, done = env.step(s, c[d.action_idx])
            trajs[i].steps.append(Step(d.state_feat, d.action_feats, np.ones(len(c), dtype=bool),
                                       d.action_idx, d.logp, d.value))
            trajs[i].info.setdefault("actions", []).append(c[d.action_idx].kind)
            states[i].append(nxt)
            if not done:
                still.append(i)
        live = still

    # ---- materialise, then one batched reader call
    mats, suff_cache = [], {}
    for tr, st in zip(trajs, states):
        final = st[-1]
        m = env.materialise(final)
        mats.append(m)
        tr.subtable, tr.subtable_tokens = m.text, m.tokens
        tr.info.update(visited=list(final.visited), n_rows=final.n_rows, rows_shown=m.rows_shown, overflow=m.overflow,
                       columns=list(m.columns))
        if use_gold:
            gold = golds[tr.instance_id]
            tr.info["sufficient"] = sufficient(env, final, m, gold, suff_cache.setdefault(final.instance.db_ref, {}))
            tr.info["path_match"] = path_match(final, gold)
    ask = [i for i, m in enumerate(mats) if m.rows_shown > 0]
    meta = None
    if getattr(reader, "needs_gold", False):
        meta = [{"sufficient": trajs[i].info["sufficient"], "gold_answer": golds[trajs[i].instance_id].gold_answer}
                for i in ask]
    answers, called = reader.read_batch([question_text(states[i][-1].instance) for i in ask],
                                        [mats[i].text for i in ask], meta) if ask else ([], [])
    for tr in trajs:
        tr.answer, tr.reader_called = "", False
    for i, ans, was_called in zip(ask, answers, called):
        trajs[i].answer, trajs[i].reader_called = ans, was_called

    # ---- gold-side scoring
    if use_gold:
        for tr, st in zip(trajs, states):
            gold = golds[tr.instance_id]
            tr.info["r_ans"] = em_norm(tr.answer, gold.gold_answer)
            if detail:  # the full metric suite (evaluation); costs one extra SQL query per episode
                final = st[-1]
                tr.info.update(set_f1=set_f1(tr.answer, gold.gold_answer),
                               answer_in_subtable=answer_in_subtable(tr.subtable, gold.gold_answer),
                               hops=entity_overlap(env, final, gold, suff_cache.setdefault(final.instance.db_ref, {})),
                               **item_counts(final.joins, final.anchors, gold.gold_edges, gold.gold_anchors))
            if oracle_mode == "reward":
                r = reward_cfg.answer_reward(tr.info["r_ans"], tr.info["sufficient"])
                assemble(tr, st, gold, r, reward_cfg, budget)
    return trajs
