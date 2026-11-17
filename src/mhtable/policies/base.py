"""Policy interface (DESIGN.md §6.5): every policy network scores a variable-size, masked set of
candidate actions and estimates the value of the state."""

from __future__ import annotations

import torch
from torch import nn

MASKED = -1e9  # logit of a padded / illegal candidate


class Policy(nn.Module):
    def forward(self, state_feat: torch.Tensor, action_feats: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """state_feat [B, d_s], action_feats [B, A, d_a], mask [B, A] (True = legal candidate)
        -> logits [B, A] (masked entries = MASKED), value [B]."""
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
