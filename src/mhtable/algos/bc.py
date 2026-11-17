"""
Behaviour cloning (PLAN.md P7, report C.5 / E.5 baseline B7): supervised learning on the oracle's
canonical trajectories. It is a baseline in its own right and the starting point of every RL arm.

    demos, stats = demonstrations(env, featuriser, items)   # gold paths the environment offers
    batch = bc.compute_advantages(demos)                    # targets are the gold actions
    stats = bc.update(batch)                                # one epoch of cross-entropy

A question contributes demonstrations only if every one of its gold actions is among the
candidates the environment offers (otherwise there is no correct label to imitate); the rest
are counted in `stats["unreachable"]`. Only the actor is trained — the critic stays at its
initialisation and is learned by the RL arm that uses it.
"""

from __future__ import annotations

import numpy as np
import torch

from mhtable.algos.base import Algorithm, Batch, collate, device_of, evaluate_actions
from mhtable.registry import register
from mhtable.types import Step, Trajectory


def demonstrations(env, featuriser, items, progress: bool = False) -> tuple[list[Trajectory], dict]:
    """Follow each gold trajectory through the environment, recording features and the index of
    the gold action among the candidates at every step."""
    demos, stats = [], {"questions": len(items), "reachable": 0, "unreachable": 0, "errors": 0, "steps": 0}
    it = items
    if progress:
        from tqdm import tqdm

        it = tqdm(items, desc="demonstrations", mininterval=10)
    for inst, gold in it:
        try:
            s, steps = env.reset(inst), []
            for g in gold.gold_trajectory:
                cands = env.candidate_actions(s)
                idx = next((i for i, a in enumerate(cands) if a.key() == g.key()), None)
                if idx is None:
                    steps = None
                    break
                sf, af = featuriser(s, cands)
                steps.append(Step(sf, af, np.ones(len(cands), dtype=bool), idx, 0.0, 0.0))
                s, done = env.step(s, cands[idx])
                if done:
                    break
        except Exception:  # a database query failed or timed out: skip the question, keep going
            stats["errors"] += 1
            continue
        if not steps:
            stats["unreachable"] += 1
            continue
        stats["reachable"] += 1
        stats["steps"] += len(steps)
        demos.append(Trajectory(inst.instance_id, steps))
    return demos, stats


@register("algo", "bc")
class BC(Algorithm):
    def __init__(self, policy, lr: float = 3e-4, batch_size: int = 256, weight_decay: float = 0.0,
                 grad_clip: float = 0.5, label_smoothing: float = 0.0, seed: int = 0):
        super().__init__(policy, lr=lr, batch_size=batch_size, weight_decay=weight_decay, grad_clip=grad_clip,
                         label_smoothing=label_smoothing)
        self.batch_size, self.grad_clip, self.label_smoothing = batch_size, grad_clip, label_smoothing
        self.params = list(policy.actor_parameters()) if hasattr(policy, "actor_parameters") else list(policy.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        self.generator = torch.Generator().manual_seed(seed)

    def compute_advantages(self, trajs: list[Trajectory]) -> Batch:
        return collate(trajs)  # imitation has no advantages: the stored action is the target

    def _loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        logp, _, _, logp_all = evaluate_actions(self.policy, batch)
        nll = -logp
        if self.label_smoothing:  # spread a little probability over the other legal candidates
            uniform = -(logp_all.masked_fill(~batch.mask, 0.0)).sum(-1) / batch.mask.sum(-1)
            nll = (1 - self.label_smoothing) * nll + self.label_smoothing * uniform
        correct = logp_all.argmax(-1) == batch.action_idx
        return nll.mean(), correct

    def update(self, batch: Batch) -> dict:
        """One pass over the demonstrations in shuffled minibatches."""
        self.policy.train()
        batch = batch.to(device_of(self.policy))
        perm = torch.randperm(len(batch), generator=self.generator).to(batch.state.device)
        loss_sum = hit = 0.0
        for k in range(0, len(batch), self.batch_size):
            idx = perm[k:k + self.batch_size]
            loss, correct = self._loss(batch.select(idx))
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            self.opt.step()
            loss_sum += loss.item() * len(idx)
            hit += correct.sum().item()
        return {"loss": loss_sum / len(batch), "step_acc": hit / len(batch)}

    @torch.no_grad()
    def evaluate(self, batch: Batch) -> dict:
        """Teacher-forced: loss, per-step accuracy, and the share of trajectories with every step right."""
        self.policy.eval()
        loss_sum, correct = 0.0, []
        for k in range(0, len(batch), 1024):
            part = batch.select(torch.arange(k, min(k + 1024, len(batch)))).to(device_of(self.policy))
            loss, c = self._loss(part)
            loss_sum += loss.item() * len(part)
            correct.append(c.cpu())
        correct = torch.cat(correct)
        n_traj = int(batch.traj.max().item()) + 1
        wrong = torch.zeros(n_traj).index_add_(0, batch.traj, (~correct).float())
        return {"loss": loss_sum / len(batch), "step_acc": correct.float().mean().item(),
                "traj_acc": (wrong == 0).float().mean().item()}
