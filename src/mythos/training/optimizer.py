"""AdamW optimizer with parameter-group weight decay."""

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Build AdamW with separate parameter groups:
      - Decay group:    matrices (weight decay applied)
      - No-decay group: biases, norms, embeddings (no weight decay)

    This follows the GPT-2/LLaMA convention where weight decay is only
    applied to weight matrices to avoid shrinking learned scale parameters.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
