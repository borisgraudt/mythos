"""
optimizer.py — AdamW optimizer with weight decay.
"""

import torch
from torch.optim import AdamW


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """
    Build AdamW optimizer with decoupled weight decay.

    Separates parameters into those with and without weight decay:
    - 2D params (weights) get weight decay
    - 1D params (biases, norms) don't
    """
    # Separate parameters
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            # Weights (embedding, linear layers)
            decay_params.append(param)
        else:
            # Biases, norms
            no_decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )

    return optimizer
