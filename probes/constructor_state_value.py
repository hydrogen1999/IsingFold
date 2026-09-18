"""Independent, pre-action base-return critic for the constructive policy.

The actor's logits and observations are unchanged. Pooling every legal candidate
is a state observation: no sampled action or terminal label enters the critic.
The zero output head makes a fresh residual-LOO baseline equal ordinary LOO.
"""
import math

import torch
from torch import nn


class ConstructorStateValue(nn.Module):
    """Predict terminal base utility from legal rows and public state context."""

    def __init__(self, in_dim, width=32):
        super().__init__()
        for name, value in (("in_dim", in_dim), ("width", width)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.in_dim, self.width = in_dim, width
        self.net = nn.Sequential(nn.Linear(2 * in_dim + 3, width), nn.SiLU(),
                                 nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rows, *, remaining_fraction, progress):
        if (rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != self.in_dim
                or not torch.isfinite(rows).all()):
            raise ValueError("critic rows must be finite nonempty [candidates, in_dim]")
        for name, value in (("remaining_fraction", remaining_fraction), ("progress", progress)):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        # Detachment is also enforced here so a caller cannot accidentally couple
        # critic regression into the actor through learned candidate features.
        rows = rows.detach()
        context = torch.cat((rows.mean(0), rows.max(0).values, rows.new_tensor([
            math.log1p(len(rows)), remaining_fraction, progress])))
        return self.net(context).squeeze(-1)
