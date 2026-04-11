"""
loop.py — Main training loop.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .trainer import Trainer
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def training_loop(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    max_steps: int,
    save_dir: Path,
    save_every: int = 1000,
    eval_every: int = 500,
    log_every: int = 50,
    resume_step: int = 0,
) -> dict:
    """
    Main training loop.

    Returns:
        Dictionary with training metrics
    """
    logger.info(f"Starting training for {max_steps} steps")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Mixed precision (bfloat16): {trainer.bfloat16}")

    # Tracking
    train_losses = []
    val_losses = []
    tokens_processed = 0
    start_time = time.time()

    # Create progress bar
    pbar = tqdm(
        total=max_steps,
        initial=resume_step,
        desc="Training",
        ncols=100,
    )

    train_iter = iter(train_loader)

    for step in range(resume_step, max_steps):
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        loss = trainer.train_step(batch)
        trainer.optimizer_step()

        train_losses.append(loss)

        # Track tokens
        tokens_processed += batch["input_ids"].numel()

        # Logging
        if (step + 1) % log_every == 0:
            avg_loss = sum(train_losses[-log_every:]) / log_every
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / max(elapsed, 1)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{trainer.scheduler.get_last_lr()[0]:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
            })

        # Evaluation
        if val_loader is not None and (step + 1) % eval_every == 0:
            val_loss = trainer.evaluate(val_loader)
            val_losses.append(val_loss)
            logger.info(f"Step {step + 1}/{max_steps} | val_loss: {val_loss:.4f}")
            trainer.maybe_save_checkpoint(val_loss, save_dir, save_every)
        elif (step + 1) % save_every == 0:
            avg_loss = sum(train_losses[-save_every:]) / min(len(train_losses), save_every)
            logger.info(f"Step {step + 1}/{max_steps} | train_loss: {avg_loss:.4f}")
            trainer.maybe_save_checkpoint(avg_loss, save_dir, save_every)

        pbar.update(1)

    pbar.close()

    # Final checkpoint
    if val_loader is not None:
        val_loss = trainer.evaluate(val_loader)
        trainer.maybe_save_checkpoint(val_loss, save_dir, save_every, is_final=True)
    else:
        trainer.maybe_save_checkpoint(float("inf"), save_dir, save_every, is_final=True)

    elapsed = time.time() - start_time

    logger.info(f"Training complete in {elapsed / 3600:.2f} hours")
    logger.info(f"Total tokens processed: {tokens_processed / 1e9:.2f}B")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "total_tokens": tokens_processed,
        "elapsed_hours": elapsed / 3600,
    }
