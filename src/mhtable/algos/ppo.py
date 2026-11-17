"""
PPO-Clip, the main arm (PLAN.md P8, report E.3 and Table 4).

    advantage   GAE(gamma = 1, lambda = 0.95) with the learned critic V, V(terminal) = 0,
                normalised over the batch
    update      K epochs of minibatch steps on
                L = -L_clip + c_V * L_V - c_H * entropy + c_KL * KL(pi || pi_BC)
                L_clip  clipped probability ratio (epsilon = 0.2)
                L_V     squared error to the GAE return, with value clipping
                KL      keeps the policy near the behaviour-cloning start (only when a BC
                        policy is given)

The entropy coefficient is annealed linearly by `set_progress(fraction of training done)`.
Actor and critic have separate learning rates; they share no weights (policies/mlp.py).
"""

from __future__ import annotations

import torch

from mhtable.algos.base import Algorithm, Batch, collate, device_of, evaluate_actions, gae
from mhtable.registry import register
from mhtable.types import Trajectory


@register("algo", "ppo")
class PPO(Algorithm):
    group_size = 1

    def __init__(self, policy, bc_policy=None, lr_actor: float = 3e-4, lr_critic: float = 1e-3, clip: float = 0.2,
                 value_clip: float = 0.2, epochs: int = 4, batch_size: int = 256, gamma: float = 1.0,
                 lam: float = 0.95, entropy_coef: float = 0.01, entropy_coef_final: float = 0.001,
                 value_coef: float = 0.5, kl_bc_coef: float = 0.02, grad_clip: float = 0.5,
                 normalize_adv: bool = True, seed: int = 0):
        super().__init__(policy)
        self.bc_policy = bc_policy
        if bc_policy is not None:
            bc_policy.eval()
            for p in bc_policy.parameters():
                p.requires_grad_(False)
        self.clip, self.value_clip, self.epochs, self.batch_size = clip, value_clip, epochs, batch_size
        self.gamma, self.lam, self.value_coef, self.grad_clip = gamma, lam, value_coef, grad_clip
        self.kl_bc_coef = kl_bc_coef if bc_policy is not None else 0.0
        self.entropy_start, self.entropy_final, self.entropy_coef = entropy_coef, entropy_coef_final, entropy_coef
        self.normalize_adv = normalize_adv
        self.opt = torch.optim.Adam([{"params": list(policy.actor_parameters()), "lr": lr_actor},
                                     {"params": list(policy.critic_parameters()), "lr": lr_critic}])
        self.generator = torch.Generator().manual_seed(seed)

    def set_progress(self, frac: float) -> None:
        frac = min(max(frac, 0.0), 1.0)
        self.entropy_coef = self.entropy_start + (self.entropy_final - self.entropy_start) * frac

    def compute_advantages(self, trajs: list[Trajectory]) -> Batch:
        batch = collate(trajs)
        adv, ret = [], []
        for tr in trajs:
            a, r = gae([s.reward for s in tr.steps], [s.value for s in tr.steps], self.gamma, self.lam)
            adv += a
            ret += r
        batch.advantage = torch.tensor(adv, dtype=torch.float32)
        batch.returns = torch.tensor(ret, dtype=torch.float32)
        return batch

    def update(self, batch: Batch) -> dict:
        self.policy.train()
        batch = batch.to(device_of(self.policy))
        dev = batch.state.device
        adv = batch.advantage
        if self.normalize_adv and len(batch) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        bc_logp = None
        if self.kl_bc_coef > 0:
            with torch.no_grad():
                bc_logp = torch.log_softmax(self.bc_policy(batch.state, batch.actions, batch.mask)[0], dim=-1)
        stats = {}
        for _ in range(self.epochs):
            perm = torch.randperm(len(batch), generator=self.generator).to(dev)
            sums = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "kl_bc", "approx_kl", "clip_frac")}
            for k in range(0, len(batch), self.batch_size):
                idx = perm[k:k + self.batch_size]
                b = batch.select(idx)
                logp, entropy, value, logp_all = evaluate_actions(self.policy, b)
                ratio = (logp - b.logp).exp()
                a = adv[idx]
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                v_clipped = b.value + (value - b.value).clamp(-self.value_clip, self.value_clip)
                value_loss = torch.max((value - b.returns) ** 2, (v_clipped - b.returns) ** 2).mean()
                kl_bc = torch.zeros((), device=dev)
                if bc_logp is not None:
                    diff = (logp_all - bc_logp[idx]).masked_fill(~b.mask, 0.0)
                    kl_bc = (logp_all.exp() * diff).sum(-1).mean()
                loss = (policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                        + self.kl_bc_coef * kl_bc)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.opt.step()
                n = len(idx)
                sums["policy_loss"] += policy_loss.item() * n
                sums["value_loss"] += value_loss.item() * n
                sums["entropy"] += entropy.mean().item() * n
                sums["kl_bc"] += kl_bc.item() * n
                sums["approx_kl"] += (b.logp - logp).mean().item() * n
                sums["clip_frac"] += ((ratio - 1).abs() > self.clip).float().mean().item() * n
            stats = {k: v / len(batch) for k, v in sums.items()}  # the last epoch's averages
        var = batch.returns.var() if len(batch) > 1 else torch.zeros((), device=dev)
        stats["explained_variance"] = float(1 - (batch.returns - batch.value).var() / var) if var > 1e-12 else 0.0
        stats["entropy_coef"] = self.entropy_coef
        return stats
