#!/usr/bin/env python3
"""
train.py — Mythos pretraining entry point.

Usage:
  python scripts/train.py --mode debug                         # 1M model, dummy data, 100 steps
  python scripts/train.py --mode data --data data/medium       # 150M on real data
  python scripts/train.py --mode full --model configs/model/base_500m.yaml --training configs/training/base.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mythos.core.transformer import Mythos, ModelConfig
from mythos.training.trainer import Trainer
from mythos.training.loop import training_loop
from mythos.training.optimizer import build_optimizer
from mythos.training.scheduler import CosineDecayWithWarmup
from mythos.training.checkpoint import load_checkpoint
from mythos.utils.device import get_device, set_seed
from mythos.utils.logging import setup_logging
from mythos.utils.config import load_config

logger = logging.getLogger(__name__)


# ── Debug config (1M model, dummy data) ─────────────────────────────────────

DEBUG_MODEL = {
    "vocab_size": 100, "d_model": 64, "n_layers": 2, "n_heads": 4,
    "n_kv_heads": 2, "d_ff": 128, "max_seq_len": 64, "dropout": 0.0,
    "norm_eps": 1e-5, "rope_theta": 10000.0,
}

DEBUG_TRAINING = {
    "batch_size": 2, "gradient_accumulation": 1, "seq_len": 32,
    "learning_rate": 1.0e-3, "min_lr": 1.0e-4, "warmup_steps": 10,
    "max_steps": 100, "weight_decay": 0.1, "grad_clip": 1.0,
    "bfloat16": False, "gradient_checkpointing": False,
    "log_every": 10, "eval_every": 50, "save_every": 100,
    "checkpoint_dir": "checkpoints/debug",
}


def create_dummy_dataloader(seq_len: int, batch_size: int) -> DataLoader:
    from torch.utils.data import TensorDataset
    input_ids = torch.randint(0, 100, (1000, seq_len + 1))
    dataset = TensorDataset(input_ids)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        return {"input_ids": ids, "labels": ids.clone()}

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)


def create_real_dataloader(data_dir: Path, seq_len: int, batch_size: int) -> DataLoader:
    from mythos.data.dataset import ShardDataset

    train_file = data_dir / "splits" / "train.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"Train split not found: {train_file}\nRun: python3.11 scripts/prepare_data.py --medium")

    with open(train_file) as f:
        shard_paths = [Path(p.strip()) for p in f if p.strip()]

    dataset = ShardDataset(shard_paths, seq_len=seq_len)

    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        labels    = torch.stack([b[1] for b in batch])
        return {"input_ids": input_ids, "labels": labels}

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)


def parse_args():
    parser = argparse.ArgumentParser(description="Mythos pretraining")
    parser.add_argument("--mode", choices=["debug", "data", "full"], default="debug")
    parser.add_argument("--model",    type=Path, help="Model config YAML (full mode)")
    parser.add_argument("--training", type=Path, help="Training config YAML (full mode)")
    parser.add_argument("--data",     type=Path, default=ROOT / "data" / "medium", help="Data directory")
    parser.add_argument("--resume",    type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument("--max_steps", type=int,  default=None, help="Override max training steps")
    parser.add_argument("--wandb",         action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="mythos")
    parser.add_argument("--wandb_entity",  type=str, default=None, help="W&B username or team")
    parser.add_argument("--run_name",      type=str, default=None, help="W&B run name / checkpoint dir suffix")
    parser.add_argument("--verbose",   action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=" * 60)
    logger.info("Mythos Training")
    logger.info("=" * 60)

    # ── Load configs ─────────────────────────────────────────────────────
    if args.mode == "debug":
        model_cfg    = DEBUG_MODEL
        training_cfg = DEBUG_TRAINING
        logger.info("Mode: DEBUG (1M model, dummy data, 100 steps)")

    elif args.mode == "data":
        # Auto-detect vocab_size from tokenizer
        tok_path = args.data / "tokenizer" / "tokenizer.json"
        if tok_path.exists():
            try:
                from tokenizers import Tokenizer as _Tok
                _vocab = _Tok.from_file(str(tok_path)).get_vocab_size()
                logger.info("Detected vocab_size: %d", _vocab)
            except Exception:
                _vocab = 32000
        else:
            _vocab = 32000

        model_cfg = load_config(ROOT / "configs" / "model" / "150m.yaml").get("model", {})
        model_cfg["vocab_size"] = _vocab

        training_cfg = {
            **DEBUG_TRAINING,
            "batch_size":      4,
            "seq_len":         512,
            "max_steps":       100000,
            "warmup_steps":    2000,
            "learning_rate":   3.0e-4,
            "min_lr":          3.0e-5,
            "bfloat16":        True,
            "log_every":       100,
            "eval_every":      2000,
            "save_every":      2000,
            "checkpoint_dir":  "checkpoints/150m_v2",
        }
        if args.max_steps:
            training_cfg["max_steps"] = args.max_steps
        logger.info("Mode: DATA | 150M | vocab=%d | steps=%d | data=%s",
                    _vocab, training_cfg["max_steps"], args.data)

    else:
        if not args.model or not args.training:
            raise ValueError("--model and --training required for full mode")
        model_cfg    = load_config(args.model).get("model", {})
        training_cfg = load_config(args.training).get("training", {})
        logger.info("Mode: FULL | model=%s | training=%s", args.model, args.training)

    # ── Build model ──────────────────────────────────────────────────────
    set_seed(42)
    model = Mythos.from_config_dict(model_cfg)
    logger.info("Model: %.2fM parameters", model.get_num_params() / 1e6)

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        lr=training_cfg.get("learning_rate", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.1),
    )
    scheduler = CosineDecayWithWarmup(
        optimizer,
        warmup_steps=training_cfg.get("warmup_steps", 1000),
        max_steps=training_cfg.get("max_steps", 100000),
        base_lr=training_cfg.get("learning_rate", 3e-4),
        min_lr=training_cfg.get("min_lr", 3e-5),
    )

    # ── Data ─────────────────────────────────────────────────────────────
    seq_len    = training_cfg.get("seq_len", 512)
    batch_size = training_cfg.get("batch_size", 4)

    if args.mode == "debug":
        train_loader = create_dummy_dataloader(seq_len, batch_size)
        val_loader   = None
        logger.info("Data: dummy (debug)")
    else:
        train_loader = create_real_dataloader(args.data, seq_len, batch_size)
        val_loader   = None
        logger.info("Data: %s", args.data)

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=training_cfg.get("grad_clip", 1.0),
        gradient_accumulation=training_cfg.get("gradient_accumulation", 1),
        bfloat16=training_cfg.get("bfloat16", True) and get_device().type != "cpu",
    )

    # Resume
    resume_step = 0
    if args.resume and args.resume.exists():
        model, optimizer, resume_step = load_checkpoint(args.resume, model, optimizer)
        trainer.global_step = resume_step
        logger.info("Resumed from step %d", resume_step)

    # ── Training ─────────────────────────────────────────────────────────
    save_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints/debug"))
    if args.run_name:
        save_dir = save_dir.parent / f"{save_dir.name}_{args.run_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # W&B init (optional)
    wandb_run = None
    if args.wandb:
        try:
            import os, wandb
            os.environ.setdefault("WANDB_SILENT", "true")  # suppresses broken _print_logged_in_message
            entity = args.wandb_entity
            if entity is None:
                try:
                    entity = wandb.Api().viewer.username
                    logger.info("W&B entity auto-detected: %s", entity)
                except Exception as api_err:
                    logger.warning("Could not auto-detect W&B entity: %s", api_err)
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=entity,
                name=args.run_name,
                config={
                    "model": model_cfg,
                    "training": training_cfg,
                    "n_params_M": round(model.get_num_params() / 1e6, 1),
                    "device": str(get_device()),
                },
                dir=str(save_dir),
            )
            logger.info("W&B enabled: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — pip install wandb. Continuing without it.")
        except Exception as e:
            logger.warning("W&B init failed (%s) — continuing without it. "
                           "Check your entity name at wandb.ai/settings", e)

    # Dump resolved configs next to the checkpoints (reproducibility)
    import json
    (save_dir / "model_config.json").write_text(json.dumps(model_cfg, indent=2))
    (save_dir / "training_config.json").write_text(json.dumps(training_cfg, indent=2))

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    training_loop(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=training_cfg.get("max_steps", 100000),
        save_dir=save_dir,
        save_every=training_cfg.get("save_every", 1000),
        eval_every=training_cfg.get("eval_every", 500),
        log_every=training_cfg.get("log_every", 50),
        resume_step=resume_step,
        accum_steps=training_cfg.get("gradient_accumulation", 1),
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Done. Checkpoint: %s", save_dir / "final.pt")


if __name__ == "__main__":
    main()
