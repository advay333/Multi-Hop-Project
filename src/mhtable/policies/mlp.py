"""
MLP actor–critic (PLAN.md P5, report E.2 with MLP towers instead of the transformer).

    actor   z = StateTower(state)            phi_a = ActionTower(action a)
            logit(a) = <W z, phi_a> / sqrt(d) * exp(log_temperature)     masked softmax over A(s)
    critic  V(s) = head(own StateTower(state))    — its own tower, so the two losses do not
                                                    interfere through shared weights

A tower projects each embedding slot with one shared linear layer (+ LayerNorm and a learned
slot-type vector), sends the scalars through their own small MLP (so embeddings and scalars
are normalised separately), concatenates and applies an MLP. Each candidate is scored
independently of the others, so the logits do not depend on candidate order (permutation
invariance). The last action-tower layer and the value head start small, so the initial policy
is near uniform and V near 0.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mhtable.policies.base import MASKED, Policy
from mhtable.registry import register


class Tower(nn.Module):
    def __init__(self, d_emb: int, n_slots: int, n_scalars: int, d: int, hidden: int):
        super().__init__()
        self.d_emb, self.n_slots = d_emb, n_slots
        self.proj = nn.Linear(d_emb, d)
        self.norm = nn.LayerNorm(d)
        self.slot_type = nn.Parameter(torch.zeros(n_slots, d))
        self.scalars = nn.Sequential(nn.Linear(n_scalars, d), nn.ReLU(), nn.Linear(d, d))
        self.body = nn.Sequential(nn.Linear((n_slots + 1) * d, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.n_slots * self.d_emb
        emb = x[..., :split].reshape(*x.shape[:-1], self.n_slots, self.d_emb)
        h = self.norm(self.proj(emb)) + self.slot_type
        h = torch.cat([h.flatten(-2), self.scalars(x[..., split:])], dim=-1)
        return self.body(h)


@register("policy", "mlp")
class MLPPolicy(Policy):
    def __init__(self, spec: dict, d: int = 256, hidden: int = 512, init_scale: float = 0.01):
        super().__init__()
        self.d = d
        e = spec["d_emb"]
        self.state_tower = Tower(e, spec["state_slots"], spec["state_scalars"], d, hidden)
        self.action_tower = Tower(e, spec["action_slots"], spec["action_scalars"], d, hidden)
        self.query = nn.Linear(d, d)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.value_tower = Tower(e, spec["state_slots"], spec["state_scalars"], d, hidden)
        self.value_head = nn.Sequential(nn.ReLU(), nn.Linear(d, 1))
        with torch.no_grad():
            for last in (self.action_tower.body[-1], self.value_head[-1]):
                last.weight.mul_(init_scale)
                last.bias.zero_()

    def actor_parameters(self):
        yield from self.state_tower.parameters()
        yield from self.action_tower.parameters()
        yield from self.query.parameters()
        yield self.log_temperature

    def critic_parameters(self):
        yield from self.value_tower.parameters()
        yield from self.value_head.parameters()

    def forward(self, state_feat, action_feats, mask):
        state_feat, action_feats = state_feat.float(), action_feats.float()  # stored features may be float16
        z = self.query(self.state_tower(state_feat))  # [B, d]
        phi = self.action_tower(action_feats)  # [B, A, d]
        logits = (z.unsqueeze(1) * phi).sum(-1) / math.sqrt(self.d) * self.log_temperature.exp()
        logits = logits.masked_fill(~mask, MASKED)
        value = self.value_head(self.value_tower(state_feat)).squeeze(-1)
        return logits, value
