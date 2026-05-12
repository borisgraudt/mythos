"""trainer.py — Training step, evaluation, and checkpointing."""

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from ..utils.device import get_device

logger = logging.getLogger(__name__)


class Trainer:
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
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels    = batch["labels"].to(self.device)

        ctx = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16) if self.bfloat16 else nullcontext()

        with ctx:
            logits, _ = self.model(input_ids)
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / self.gradient_accumulation

        loss.backward()
        return loss.item() * self.gradient_accumulation

    def optimizer_step(self) -> float:
        """Clip gradients, step optimizer and scheduler. Returns the gradient norm
        (before clipping) — useful for training-stability logging in papers."""
        grad_norm = 0.0
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            ).item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step(self.global_step)
        self.global_step += 1
        return grad_norm

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, max_batches: int = 50) -> float:
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels    = batch["labels"].to(self.device)
            logits, _ = self.model(input_ids)
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            total_loss += self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
            num_batches += 1
            if max_batches and num_batches >= max_batches:
                break
        return total_loss / max(num_batches, 1)

    def maybe_save_checkpoint(self, val_loss: float, save_dir: Path, save_every: int, is_final: bool = False):
        save_current = is_final or (self.global_step % save_every == 0)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_checkpoint(self.model, self.optimizer, self.global_step, save_dir / "best.pt")
        if save_current:
            path = save_dir / "final.pt" if is_final else save_dir / f"step_{self.global_step:06d}.pt"
            save_checkpoint(self.model, self.optimizer, self.global_step, path)
