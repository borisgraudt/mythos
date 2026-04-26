"""
checkpoint.py — Save and load training checkpoints.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    config: Optional[dict] = None,
):
    """Save model, optimizer, and training state to a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if config is None and hasattr(model, "config"):
        config = model.config.to_dict()

    state = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "config":    config or {},
    }

    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int]:
    """Load checkpoint and restore model, optimizer, and step."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    step = state.get("step", 0)
    logger.info("Checkpoint loaded: %s (step %d)", path, step)

    return model, optimizer, step
