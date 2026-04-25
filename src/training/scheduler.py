"""
scheduler.py — Learning rate scheduler with warmup and cosine decay.
"""

import math
import torch


class CosineDecayWithWarmup:
    """
    Cosine decay with linear warmup.

    Schedule:
      - Steps 0 to warmup_steps: linear increase from 0 to lr
      - Steps warmup_steps to max_steps: cosine decay to min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        base_lr: float = 3e-4,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_step = 0

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, step: int):
        """Update learning rate for all parameter groups."""
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step = step

    def get_last_lr(self) -> list:
        return [self.get_lr(self.current_step)]
