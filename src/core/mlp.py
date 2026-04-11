"""SwiGLU Feed-Forward Network."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    SwiGLU: FFN(x) = (SiLU(xW₁) ⊙ xW₃) W₂

    Hidden dim is scaled to 2/3 of d_ff and rounded to nearest 256 for
    hardware efficiency (tensor core alignment).
    """

    def __init__(self, config):
        super().__init__()
        hidden = int(2 * config.d_ff / 3)
        hidden = 256 * math.ceil(hidden / 256)  # align to 256

        self.w1 = nn.Linear(config.d_model, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)  # down
        self.w3 = nn.Linear(config.d_model, hidden, bias=False)  # up
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
