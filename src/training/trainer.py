"""
trainer.py — Training loop for Mythos.
"""

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from ..utils.device import get_device


class Trainer:
    """
    Training manager for Mythos model.

    Handles:
      - Training steps (forward, backward, update)
      - Evaluation
      - Checkpointing
      - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: Optional[torch.device] = None,
        grad_clip: float = 1.0,
        gradient_accumulation: int = 1,
        bfloat16: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.grad_clip = grad_clip
        self.gradient_accumulation = gradient_accumulation
        self.bfloat16 = bfloat16 and self.device.type != "cpu"

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step with gradient accumulation.

        Returns: loss value
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Mixed precision
        if self.bfloat16:
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            logits, _ = self.model(input_ids)

            # Shift for next-token prediction
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / self.gradient_accumulation

        loss.backward()

        return loss.item() * self.gradient_accumulation

    def optimizer_step(self):
        """Gradient clipping and optimizer step."""
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.scheduler.step(self.global_step)
        self.global_step += 1

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluation on validation set."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _ = self.model(input_ids)

            # Shift for next-token prediction
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def maybe_save_checkpoint(
        self,
        val_loss: float,
        save_dir: Path,
        save_every: int,
        is_final: bool = False,
    ):
        """Save checkpoint if validation improved or at save interval."""
        save_current = is_final or (self.global_step % save_every == 0)
        is_best = val_loss < self.best_val_loss

        if is_best:
            self.best_val_loss = val_loss
            best_path = save_dir / "best.pt"
            save_checkpoint(self.model, self.optimizer, self.global_step, best_path)

        if save_current:
            if is_final:
                step_path = save_dir / "final.pt"
            else:
                step_path = save_dir / f"step_{self.global_step:06d}.pt"
            save_checkpoint(self.model, self.optimizer, self.global_step, step_path)
