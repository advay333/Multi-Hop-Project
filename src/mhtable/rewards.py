"""
Rewards (report C.4, DESIGN.md §6.6) — gold-aware; called by the rollout in `oracle_mode="reward"`.

    r_t = beta * [Phi(s_{t+1}) - Phi(s_t)]  +  1[last step] * (R_ans - lambda_c * tok(S) / B)

Phi(s) is the Jaccard overlap between the joins and filters taken so far and the gold query's
(order-free, so valid alternative orders are not punished; a wrong item lowers it for good).
Phi of the terminal state is 0, so over a whole episode the shaping terms cancel
(Phi(s_T) - Phi(s_0) = 0 - 0): shaping moves credit earlier in time without changing which
policy is best. The token cost does change the objective, on purpose.

Variants: R0 sparse (beta = 0) | R1 shaped (default).
An episode started after k gold steps (curriculum) begins at Phi(s_k) > 0; its shaping terms
then sum to -beta * Phi(s_k), a constant for that start state.
"""

from __future__ import annotations

from dataclasses import dataclass

from mhtable.oracle import items, potential
from mhtable.types import GoldRecord, State, Trajectory


@dataclass
class RewardConfig:
    variant: str = "R1"  # R0 sparse | R1 shaped
    beta: float = 0.5
    lambda_cost: float = 0.1

    def __post_init__(self):
        if self.variant not in ("R0", "R1"):
            raise ValueError(f"unknown reward variant {self.variant!r} (R0 | R1)")

    @property
    def shaping(self) -> float:
        return 0.0 if self.variant == "R0" else self.beta


def gold_items(gold: GoldRecord) -> frozenset:
    return items(gold.gold_edges, gold.gold_anchors)


def phi_sequence(states: list[State], gold: GoldRecord) -> list[float]:
    """Phi(s_0), ..., Phi(s_T) for the states an episode went through; the last one is terminal."""
    g = gold_items(gold)
    return [potential(items(s.joins, s.anchors), g) for s in states[:-1]] + [0.0]


def assemble(traj: Trajectory, states: list[State], gold: GoldRecord, r_ans: float,
             cfg: RewardConfig, budget: int) -> None:
    """Write per-step rewards and potentials into `traj.steps`. `states` are the states the
    episode went through: states[t] is where step t was taken, states[-1] is terminal."""
    if len(states) != len(traj.steps) + 1:
        raise ValueError(f"{len(traj.steps)} steps need {len(traj.steps) + 1} states, got {len(states)}")
    phi = phi_sequence(states, gold)
    cost = cfg.lambda_cost * traj.subtable_tokens / max(budget, 1)
    for t, step in enumerate(traj.steps):
        step.phi = phi[t]
        step.reward = cfg.shaping * (phi[t + 1] - phi[t])
    if traj.steps:
        traj.steps[-1].reward += r_ans - cost
    traj.info.update(cost=cost, outcome=r_ans - cost, **{"return": sum(s.reward for s in traj.steps)})
