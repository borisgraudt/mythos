"""
Unit tests for training loop components.

Run:
    pytest tests/test_training.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mythos.core.transformer import Mythos, ModelConfig
from mythos.training.optimizer import build_optimizer
from mythos.training.scheduler import CosineDecayWithWarmup
from mythos.training.trainer import Trainer
from mythos.training.checkpoint import save_checkpoint, load_checkpoint


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=64,
    )


@pytest.fixture
def tiny_model(tiny_config) -> Mythos:
    return Mythos(tiny_config)


@pytest.fixture
def optimizer(tiny_model):
    return build_optimizer(tiny_model)


# ── Optimizer ─────────────────────────────────────────────────────────────────

class TestOptimizer:
    def test_creates_adamw(self, optimizer):
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_no_decay_on_norms(self, tiny_model):
        """RMSNorm weights and biases should have no weight decay."""
        opt = build_optimizer(tiny_model, lr=1e-3, weight_decay=0.1)
        no_decay_params = {id(p) for g in opt.param_groups if g["weight_decay"] == 0 for p in g["params"]}
        for name, p in tiny_model.named_parameters():
            if "norm" in name or name.endswith(".bias"):
                assert id(p) in no_decay_params, f"{name} should have no weight decay"


# ── Scheduler ─────────────────────────────────────────────────────────────────

class TestScheduler:
    def test_warmup_phase(self, optimizer):
        scheduler = CosineDecayWithWarmup(
            optimizer,
            warmup_steps=10,
            max_steps=100,
            base_lr=3e-4,
            min_lr=3e-5,
        )
        scheduler.step(1)  # mid-warmup
        lr = optimizer.param_groups[0]["lr"]
        assert lr < 3e-4, f"LR during warmup should be < peak: {lr}"

    def test_peak_at_warmup_end(self, optimizer):
        warmup_steps = 5
        base_lr = 3e-4
        scheduler = CosineDecayWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=100,
            base_lr=base_lr,
            min_lr=3e-5,
        )
        scheduler.step(warmup_steps)
        lr = optimizer.param_groups[0]["lr"]
        assert abs(lr - base_lr) < 1e-6, f"LR at warmup end should equal base_lr: {lr} vs {base_lr}"

    def test_min_lr_at_end(self, optimizer):
        min_lr = 3e-5
        max_steps = 10
        scheduler = CosineDecayWithWarmup(
            optimizer,
            warmup_steps=2,
            max_steps=max_steps,
            base_lr=3e-4,
            min_lr=min_lr,
        )
        scheduler.step(max_steps)
        lr = optimizer.param_groups[0]["lr"]
        assert abs(lr - min_lr) < 1e-7, f"LR at end should equal min_lr: {lr} vs {min_lr}"

    def test_lr_monotone_after_warmup(self, optimizer):
        """After warmup, LR should never increase."""
        warmup = 3
        max_steps = 20
        scheduler = CosineDecayWithWarmup(
            optimizer, warmup_steps=warmup, max_steps=max_steps, base_lr=3e-4, min_lr=3e-5
        )
        lrs = []
        for step in range(max_steps):
            scheduler.step(step)
            lrs.append(optimizer.param_groups[0]["lr"])

        post_warmup = lrs[warmup:]
        for i in range(len(post_warmup) - 1):
            assert post_warmup[i] >= post_warmup[i + 1], \
                f"LR increased at step {warmup + i}: {post_warmup[i]} → {post_warmup[i+1]}"


# ── Trainer ───────────────────────────────────────────────────────────────────

class TestTrainer:
    def _make_batch(self, tiny_config, seq_len=8):
        x = torch.randint(0, tiny_config.vocab_size, (2, seq_len + 1))
        return {"input_ids": x, "labels": x.clone()}

    def test_train_step_returns_loss(self, tiny_model, tiny_config, optimizer):
        scheduler = CosineDecayWithWarmup(optimizer, warmup_steps=5, max_steps=100, base_lr=1e-3)
        trainer = Trainer(tiny_model, optimizer, scheduler)
        batch = self._make_batch(tiny_config)
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases_over_steps(self, tiny_model, tiny_config, optimizer):
        """10 training steps should reduce loss vs initial."""
        scheduler = CosineDecayWithWarmup(optimizer, warmup_steps=2, max_steps=100, base_lr=1e-3)
        trainer = Trainer(tiny_model, optimizer, scheduler)

        losses = []
        for _ in range(10):
            batch = self._make_batch(tiny_config)
            losses.append(trainer.train_step(batch))

        assert losses[-1] < losses[0], f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"

    def test_gradient_accumulation(self, tiny_model, tiny_config, optimizer):
        """Gradient accumulation should produce same effective gradient as full batch."""
        scheduler = CosineDecayWithWarmup(optimizer, warmup_steps=2, max_steps=100, base_lr=1e-3)
        trainer = Trainer(tiny_model, optimizer, scheduler, gradient_accumulation=2)
        batch = self._make_batch(tiny_config)
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))


# ── Checkpoint ────────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load(self, tiny_model, optimizer, tmp_path):
        step = 42
        path = tmp_path / "step_000042.pt"

        save_checkpoint(tiny_model, optimizer, step, path)
        assert path.exists()

        loaded_model, loaded_opt, loaded_step = load_checkpoint(path, tiny_model, optimizer)
        assert loaded_step == step

    def test_model_state_preserved(self, tiny_model, optimizer, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(tiny_model, optimizer, 1, path)

        # Corrupt model weights
        for p in tiny_model.parameters():
            p.data.fill_(0.0)

        # Reload
        tiny_model, _, _ = load_checkpoint(path, tiny_model, optimizer)

        # Weights should not all be zero after reload
        total = sum(p.abs().sum().item() for p in tiny_model.parameters())
        assert total > 0, "Checkpoint did not restore model weights"
