#!/usr/bin/env python3
"""
train.py — Mythos pretraining entry point.

Usage:
  # Debug run (1M model, 100 steps)
  python scripts/train.py --mode debug

  # Full 500M training
  python scripts/train.py --model configs/model/base_500m.yaml --training configs/training/base.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

# Make sure src/ is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.transformer import Mythos, ModelConfig
from src.training.trainer import Trainer
from src.training.loop import training_loop
from src.training.optimizer import build_optimizer
from src.training.scheduler import CosineDecayWithWarmup
from src.training.checkpoint import load_checkpoint
from src.utils.device import get_device, set_seed
from src.utils.logging import setup_logging
from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ── Debug config (1M model, tiny data) ──────────────────────────────────────
DEBUG_MODEL = {
    "vocab_size": 100,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "d_ff": 128,
    "max_seq_len": 64,
    "dropout": 0.0,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
}

DEBUG_TRAINING = {
    "batch_size": 2,
    "gradient_accumulation": 1,
    "seq_len": 32,
    "learning_rate": 1.0e-3,
    "min_lr": 1.0e-4,
    "warmup_steps": 10,
    "max_steps": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "bfloat16": False,
    "gradient_checkpointing": False,
    "log_every": 10,
    "eval_every": 50,
    "save_every": 100,
    "checkpoint_dir": "checkpoints/debug",
}


def create_dummy_dataloader(seq_len: int, batch_size: int, num_samples: int = 1000) -> DataLoader:
    """
    Create dummy dataloader for debug training.
    Generates random token IDs in range [0, vocab_size).
    """
    from torch.utils.data import TensorDataset

    # Random tokens (vocab_size = 100 for debug)
    input_ids = torch.randint(0, 100, (num_samples, seq_len + 1))
    dataset = TensorDataset(input_ids)

    def collate_fn(batch):
        input_ids = torch.stack([b[0] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def create_real_dataloader(data_dir: Path, seq_len: int, batch_size: int) -> DataLoader:
    """
    Create dataloader from prepared data.
    """
    from data.dataset import ShardDataset

    # Load train split
    train_file = data_dir / "splits" / "train.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"Train split not found: {train_file}")

    with open(train_file) as f:
        shard_paths = [Path(p.strip()) for p in f if p.strip()]

    if not shard_paths:
        raise ValueError(f"No shard paths found in {train_file}")

    dataset = ShardDataset(shard_paths, seq_len=seq_len)

    def collate_fn(batch):
        input_ids = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return {"input_ids": input_ids, "labels": labels}

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def parse_args():
    parser = argparse.ArgumentParser(description="Mythos pretraining")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "data", "full"],
        default="debug",
        help="Debug (1M model, dummy data), Data (150M on real data), or Full (500M)",
    )
    parser.add_argument("--model", type=Path, help="Model config YAML")
    parser.add_argument("--training", type=Path, help="Training config YAML")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "debug", help="Data directory")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=" * 60)
    logger.info("Mythos Training")
    logger.info("=" * 60)

    # Load configs
    if args.mode == "debug":
        model_cfg = DEBUG_MODEL
        training_cfg = DEBUG_TRAINING
        logger.info("Mode: DEBUG (1M model, dummy data)")
    elif args.mode == "data":
        # Auto-detect vocab_size from tokenizer if present
        tok_path = args.data / "tokenizer" / "tokenizer.json"
        if tok_path.exists():
            try:
                from tokenizers import Tokenizer as _Tok
                _vocab = _Tok.from_file(str(tok_path)).get_vocab_size()
                logger.info("Detected tokenizer vocab_size: %d", _vocab)
            except Exception:
                _vocab = 32000
        else:
            _vocab = 32000

        model_cfg = load_config(ROOT / "configs" / "model" / "150m.yaml").get("model", {})
        model_cfg["vocab_size"] = _vocab

        training_cfg = DEBUG_TRAINING.copy()
        training_cfg.update({
            "batch_size": 4,
            "seq_len": 256,
            "max_steps": 5000,
            "warmup_steps": 200,
            "checkpoint_dir": f"checkpoints/150m_debug_v2",
            "bfloat16": True,
            "log_every": 50,
            "eval_every": 500,
            "save_every": 1000,
        })
        logger.info("Mode: DATA (150M model, vocab=%d)", _vocab)
    else:
        if not args.model or not args.training:
            raise ValueError("--model and --training required for full mode")
        model_cfg = load_config(args.model).get("model", {})
        training_cfg = load_config(args.training).get("training", {})
        logger.info(f"Mode: FULL")
        logger.info(f"Model config: {args.model}")
        logger.info(f"Training config: {args.training}")

    # Set seed
    set_seed(42)

    # Create model
    logger.info(f"Initializing model...")
    model = Mythos.from_config_dict(model_cfg)
    num_params = model.get_num_params()
    logger.info(f"Model created: {num_params / 1e6:.2f}M parameters")

    # Create optimizer
    optimizer = build_optimizer(
        model,
        lr=training_cfg.get("learning_rate", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.1),
    )

    # Create scheduler
    scheduler = CosineDecayWithWarmup(
        optimizer,
        warmup_steps=training_cfg.get("warmup_steps", 1000),
        max_steps=training_cfg.get("max_steps", 100000),
        min_lr=training_cfg.get("min_lr", 3e-5),
        base_lr=training_cfg.get("learning_rate", 3e-4),
    )

    # Create dataloaders
    seq_len = training_cfg.get("seq_len", 512)
    batch_size = training_cfg.get("batch_size", 4)

    if args.mode == "debug":
        train_loader = create_dummy_dataloader(seq_len=seq_len, batch_size=batch_size)
        val_loader = None
        logger.info("Using dummy data (debug mode)")
    else:
        train_loader = create_real_dataloader(args.data, seq_len=seq_len, batch_size=batch_size)
        val_loader = None  # TODO: add validation
        logger.info(f"Using real data from: {args.data}")

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=training_cfg.get("grad_clip", 1.0),
        gradient_accumulation=training_cfg.get("gradient_accumulation", 1),
        bfloat16=training_cfg.get("bfloat16", True) and get_device().type != "cpu",
    )

    # Resume from checkpoint
    resume_step = 0
    if args.resume and args.resume.exists():
        model, optimizer, resume_step = load_checkpoint(args.resume, model, optimizer)
        trainer.global_step = resume_step
        logger.info(f"Resumed from step {resume_step}")

    # Setup checkpoint directory
    save_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints/debug"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training loop...")
    logger.info("=" * 60)

    metrics = training_loop(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=training_cfg.get("max_steps", 100000),
        save_dir=save_dir,
        save_every=training_cfg.get("save_every", 1000),
        eval_every=training_cfg.get("eval_every", 500),
        log_every=training_cfg.get("log_every", 50),
        resume_step=resume_step,
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Final checkpoint: {save_dir / 'final.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
