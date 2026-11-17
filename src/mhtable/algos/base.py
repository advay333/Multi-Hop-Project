"""
Algorithm interface and the pieces every arm shares (DESIGN.md §6.6, report E.4).

An RL arm is one class with two methods:

    compute_advantages(trajs) -> Batch     how good was each step (the arm's baseline choice)
    update(batch) -> dict                  how the policy is changed (the arm's constraint)

Everything else — rollouts, rewards, the network — is shared, so arms differ only in these two.
Shared here: `collate` (trajectories -> padded tensors), `reward_to_go`, `gae`, and
`evaluate_actions` (log-probability, entropy and value of stored steps under the current policy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mhtable.rollout import pad_actions
from mhtable.types import Trajectory


@dataclass
class Batch:
    """All steps of a set of trajectories, flattened to N rows."""

    state: torch.Tensor  # [N, d_s]
    actions: torch.Tensor  # [N, A_max, d_a]
    mask: torch.Tensor  # [N, A_max] True = legal candidate
    action_idx: torch.Tensor  # [N]
    logp: torch.Tensor  # [N] log pi_old(a|s)
    value: torch.Tensor  # [N] V_old(s)
    reward: torch.Tensor  # [N]
    advantage: torch.Tensor  # [N] filled by compute_advantages
    returns: torch.Tensor  # [N] value targets, filled by compute_advantages
    traj: torch.Tensor  # [N] index of the trajectory each step belongs to
    group: torch.Tensor  # [N] index of the question (rollouts of one question share it)
    t: torch.Tensor  # [N] step index inside its trajectory

    def __len__(self) -> int:
        return self.state.shape[0]

    def select(self, idx: torch.Tensor) -> "Batch":
        return Batch(**{k: v[idx] for k, v in self.__dict__.items()})


def collate(trajs: list[Trajectory]) -> Batch:
    steps = [(i, t, s) for i, tr in enumerate(trajs) for t, s in enumerate(tr.steps)]
    if not steps:
        raise ValueError("no steps to collate")
    if any(s.state_feat is None for _, _, s in steps):
        raise ValueError("these trajectories carry no features (collected by a selector, not a policy)")
    groups: dict[str, int] = {}
    for tr in trajs:
        groups.setdefault(tr.instance_id, len(groups))
    actions, mask = pad_actions([s.action_feats for _, _, s in steps])
    f32 = lambda xs: torch.tensor(xs, dtype=torch.float32)  # noqa: E731
    n = len(steps)
    return Batch(
        state=torch.from_numpy(np.stack([s.state_feat for _, _, s in steps])),
        actions=actions, mask=mask,
        action_idx=torch.tensor([s.action_idx for _, _, s in steps], dtype=torch.long),
        logp=f32([s.logp for _, _, s in steps]), value=f32([s.value for _, _, s in steps]),
        reward=f32([s.reward for _, _, s in steps]),
        advantage=torch.zeros(n), returns=torch.zeros(n),
        traj=torch.tensor([i for i, _, _ in steps], dtype=torch.long),
        group=torch.tensor([groups[trajs[i].instance_id] for i, _, _ in steps], dtype=torch.long),
        t=torch.tensor([t for _, t, _ in steps], dtype=torch.long))


def reward_to_go(rewards: list[float], gamma: float = 1.0) -> list[float]:
    """G_t = r_t + gamma * G_{t+1} for one trajectory."""
    out, g = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def gae(rewards: list[float], values: list[float], gamma: float = 1.0,
        lam: float = 0.95) -> tuple[list[float], list[float]]:
    """Generalised advantage estimation for one trajectory, with V(terminal) = 0.
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t);   A_t = sum_l (gamma * lam)^l * delta_{t+l}
    Returns (advantages, value targets A_t + V(s_t))."""
    adv, last = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(rewards) else 0.0
        last = rewards[t] + gamma * nxt - values[t] + gamma * lam * last
        adv[t] = last
    return adv, [a + v for a, v in zip(adv, values)]


def evaluate_actions(policy, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Under the current policy: log pi(a|s) of the stored actions, entropy, V(s), and the full
    log-probabilities [N, A_max] (for KL terms)."""
    logits, value = policy(batch.state, batch.actions, batch.mask)
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(1, batch.action_idx.unsqueeze(1)).squeeze(1)
    p = logp_all.exp()
    entropy = -(p * logp_all.masked_fill(~batch.mask, 0.0)).sum(-1)
    return logp, entropy, value, logp_all


class Algorithm:
    """Base class of every training arm; subclasses are registered under kind "algo"."""

    def __init__(self, policy, **cfg):
        self.policy = policy
        self.cfg = cfg

    def compute_advantages(self, trajs: list[Trajectory]) -> Batch:
        raise NotImplementedError

    def update(self, batch: Batch) -> dict:
        raise NotImplementedError
