"""
GRPO, the critic-free arm (PLAN.md P9, report E.3): Group Relative Policy Optimisation.

    rollouts    G episodes of the same question (group_size, default 8), from the same start
    advantage   A_i = (R_i - mean(R_group)) / (std(R_group) + eps)   for every step of episode i
                R_i = the episode's total return (answer reward + shaping - token cost)
                std_normalize=False drops the division (the "Dr. GRPO" variant)
    update      the same clipped surrogate, entropy bonus, KL to BC and stability options as PPO
                (target_kl, freeze_temperature, adam_eps), with no critic: no value loss, and
                the value head is left out of the optimiser

This is exactly the research question's contrast: PPO's baseline is a learned V(s); GRPO's is the
mean return of the other rollouts of the same question.

Groups are consecutive blocks of `group_size` trajectories, as `rollout.collect` returns them
(the rollouts of instances[i] sit at i*G .. (i+1)*G - 1). A question that appears twice in one
iteration is two groups. A group whose G returns are all equal carries no learning signal (all
advantages 0); its share is logged as `zero_adv_frac` — with a sparse reward (R0) and a weak
policy this is most groups, which is the cost of having no critic.
"""

from __future__ import annotations

import torch

from mhtable.algos.base import Batch, collate
from mhtable.algos.ppo import PPO
from mhtable.registry import register
from mhtable.types import Trajectory


@register("algo", "grpo")
class GRPO(PPO):
    def __init__(self, policy, bc_policy=None, group_size: int = 8, std_normalize: bool = True,
                 adv_eps: float = 1e-6, lr_actor: float = 3e-4, freeze_temperature: bool = False,
                 adam_eps: float = 1e-8, normalize_adv: bool = False, **kw):
        kw.pop("lr_critic", None)
        kw.pop("value_coef", None)
        kw.pop("value_clip", None)
        kw.pop("lam", None)
        super().__init__(policy, bc_policy=bc_policy, lr_actor=lr_actor, value_coef=0.0,
                         freeze_temperature=freeze_temperature, adam_eps=adam_eps,
                         normalize_adv=normalize_adv, **kw)
        if group_size < 2:
            raise ValueError("GRPO needs group_size >= 2 (the group is the baseline)")
        self.group_size = group_size
        self.std_normalize, self.adv_eps = std_normalize, adv_eps
        actor = [p for p in policy.actor_parameters() if p.requires_grad]  # frozen temperature stays out
        self.opt = torch.optim.Adam(actor, lr=lr_actor, eps=adam_eps)  # actor only: no critic
        self._group_stats: dict = {}

    def compute_advantages(self, trajs: list[Trajectory]) -> Batch:
        g = self.group_size
        if len(trajs) % g:
            raise ValueError(f"{len(trajs)} trajectories is not a whole number of groups of {g}")
        batch = collate(trajs)
        adv, zero, stds = [], 0, []
        for k in range(0, len(trajs), g):
            group = trajs[k:k + g]
            rets = torch.tensor([sum(s.reward for s in tr.steps) for tr in group], dtype=torch.float64)
            std = rets.std(unbiased=False)
            stds.append(float(std))
            if std < self.adv_eps:
                zero += 1
                a = torch.zeros(g, dtype=torch.float64)
            else:
                a = rets - rets.mean()
                if self.std_normalize:
                    a = a / (std + self.adv_eps)
            for tr, ai in zip(group, a.tolist()):
                adv += [ai] * len(tr.steps)
        n_groups = len(trajs) // g
        self._group_stats = {"zero_adv_frac": zero / n_groups, "group_return_std": sum(stds) / n_groups}
        batch.advantage = torch.tensor(adv, dtype=torch.float32)
        batch.returns = torch.zeros(len(batch))  # no critic, no value targets
        return batch

    def update(self, batch: Batch) -> dict:
        stats = super().update(batch)
        for k in ("value_loss", "explained_variance"):  # the untrained value head means nothing here
            stats.pop(k, None)
        stats.update(self._group_stats)
        return stats
