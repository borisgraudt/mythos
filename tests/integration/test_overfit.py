"""Integration smoke test: verify the model can overfit a tiny batch.

A model that cannot overfit a single batch has a training bug.
This test is the canonical sanity check before any full training run.
"""
import torch
import pytest
from mythos.core.transformer import Mythos, ModelConfig
from mythos.training.trainer import Trainer
from mythos.training.optimizer import build_optimizer
from mythos.training.scheduler import CosineDecayWithWarmup


@pytest.fixture
def tiny_model():
    config = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
    )
    return Mythos(config)


def _make_batch(vocab_size: int = 256, seq_len: int = 16, batch_size: int = 2) -> dict:
    torch.manual_seed(42)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
    return {
        "input_ids": ids[:, :-1],
        "labels": ids[:, 1:],
    }


def test_overfit_single_batch(tiny_model):
    """Loss should drop significantly when repeatedly training on the same batch."""
    device = torch.device("cpu")
    optimizer = build_optimizer(tiny_model, lr=1e-3, weight_decay=0.0)
    scheduler = CosineDecayWithWarmup(optimizer, warmup_steps=0, max_steps=100, base_lr=1e-3, min_lr=1e-4)
    trainer = Trainer(tiny_model, optimizer, scheduler, device=device, bfloat16=False)

    batch = _make_batch()
    initial_loss = trainer.train_step(batch)

    for _ in range(50):
        trainer.train_step(batch)
        trainer.optimizer_step()

    final_loss = trainer.train_step(batch)
    assert final_loss < initial_loss * 0.5, (
        f"Expected loss to drop by >50% on fixed batch; "
        f"got initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_gradient_flows_to_all_layers(tiny_model):
    """Every parameter should receive a gradient after a forward+backward pass."""
    device = torch.device("cpu")
    optimizer = build_optimizer(tiny_model, lr=1e-3, weight_decay=0.0)
    scheduler = CosineDecayWithWarmup(optimizer, warmup_steps=0, max_steps=10, base_lr=1e-3, min_lr=1e-4)
    trainer = Trainer(tiny_model, optimizer, scheduler, device=device, bfloat16=False)

    batch = _make_batch()
    trainer.train_step(batch)

    no_grad = [name for name, p in tiny_model.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"
