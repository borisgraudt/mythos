"""Cosine decay learning rate scheduler with linear warmup."""

import math
import torch


class CosineDecayWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup → cosine decay to min_lr.

    Used in: GPT-3, LLaMA, Chinchilla (DeepMind).

    Schedule:
        step < warmup_steps : lr = base_lr * step / warmup_steps
        step >= warmup_steps: lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(π*progress))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.min_lr       = min_lr
        self.base_lr      = base_lr
        self._current_step = 0
        super().__init__(optimizer, last_epoch)

    def step(self, step: int = None):  # type: ignore[override]
        if step is not None:
            self._current_step = step
        lr = self._compute_lr(self._current_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._current_step += 1

    def _compute_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / max(self.warmup_steps, 1)
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [self._compute_lr(max(self._current_step - 1, 0))]

    def get_lr(self):
        return [self._compute_lr(self._current_step)]
