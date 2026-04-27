"""
Unit tests for Mythos model components.

Run:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v --tb=short  # compact traceback
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mythos.core.transformer import Mythos, ModelConfig, TransformerBlock
from mythos.core.attention import Attention
from mythos.core.mlp import FeedForward
from mythos.core.norms import RMSNorm
from mythos.core.rope import build_rope_cache, apply_rope
from mythos.utils.config import load_config


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_config() -> ModelConfig:
    """Minimal config for fast unit tests (< 1M params)."""
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


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == x.shape

    def test_unit_rms(self):
        """After normalization, RMS along last dim should be ~1."""
        norm = RMSNorm(64)
        x = torch.randn(4, 64) * 10  # large values
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_no_nan(self):
        """Handles near-zero inputs without NaN."""
        norm = RMSNorm(32)
        x = torch.zeros(2, 32) + 1e-8
        out = norm(x)
        assert not out.isnan().any()


# ── RoPE ──────────────────────────────────────────────────────────────────────

class TestRoPE:
    def test_cache_shape(self):
        cos, sin = build_rope_cache(seq_len=16, head_dim=64)
        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)

    def test_apply_rope_shape(self):
        cos, sin = build_rope_cache(seq_len=8, head_dim=32)
        q = torch.randn(2, 4, 8, 32)   # (B, heads, T, head_dim)
        k = torch.randn(2, 4, 8, 32)
        q_r, k_r = apply_rope(q, k, cos[:8], sin[:8])
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_rope_is_rotation(self):
        """RoPE should preserve vector norm (rotation doesn't change length)."""
        cos, sin = build_rope_cache(seq_len=4, head_dim=16)
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        q_r, _ = apply_rope(q, k, cos[:4], sin[:4])
        norms_before = q.norm(dim=-1)
        norms_after  = q_r.norm(dim=-1)
        assert torch.allclose(norms_before, norms_after, atol=1e-5)


# ── Attention ─────────────────────────────────────────────────────────────────

class TestAttention:
    def test_output_shape(self, tiny_config):
        attn = Attention(tiny_config)
        B, T, d = 2, 8, tiny_config.d_model
        x = torch.randn(B, T, d)
        cos, sin = build_rope_cache(tiny_config.max_seq_len, tiny_config.d_model // tiny_config.n_heads)
        out, kv = attn(x, cos[:T], sin[:T])
        assert out.shape == (B, T, d)

    def test_kv_cache_returned(self, tiny_config):
        """Attention returns (output, (k, v)) — k/v shapes should be correct."""
        attn = Attention(tiny_config)
        B, T = 1, 6
        x = torch.randn(B, T, tiny_config.d_model)
        cos, sin = build_rope_cache(tiny_config.max_seq_len, tiny_config.d_model // tiny_config.n_heads)
        out, (k, v) = attn(x, cos[:T], sin[:T])
        head_dim = tiny_config.d_model // tiny_config.n_heads
        assert k.shape == (B, tiny_config.n_kv_heads, T, head_dim)
        assert v.shape == (B, tiny_config.n_kv_heads, T, head_dim)

    def test_kv_cache_incremental(self, tiny_config):
        """Incremental decoding with KV-cache produces same logits as full forward."""
        model = Mythos(tiny_config)
        model.eval()
        B, T = 1, 6

        x = torch.randint(0, tiny_config.vocab_size, (B, T))

        with torch.no_grad():
            # Full forward (no cache)
            logits_full, _ = model(x)

            # Incremental: prefill first T-1 tokens, then decode last token
            logits_prefix, past_kvs = model(x[:, :-1], use_cache=True)
            logits_last, _ = model(x[:, -1:], past_kvs=past_kvs, use_cache=True)

        # Last position logits should match
        assert torch.allclose(logits_full[:, -1], logits_last[:, -1], atol=1e-4), \
            "KV-cache produced different logits than full forward pass"

    def test_gqa_kv_heads(self, tiny_config):
        """n_kv_heads < n_heads → GQA active."""
        assert tiny_config.n_kv_heads < tiny_config.n_heads
        attn = Attention(tiny_config)
        assert attn.n_rep == tiny_config.n_heads // tiny_config.n_kv_heads

    def test_no_nan_in_output(self, tiny_config):
        attn = Attention(tiny_config)
        x = torch.randn(2, 4, tiny_config.d_model)
        cos, sin = build_rope_cache(tiny_config.max_seq_len, tiny_config.d_model // tiny_config.n_heads)
        out, _ = attn(x, cos[:4], sin[:4])
        assert not out.isnan().any()


# ── FeedForward ───────────────────────────────────────────────────────────────

class TestFeedForward:
    def test_output_shape(self, tiny_config):
        ff = FeedForward(tiny_config)
        x = torch.randn(2, 8, tiny_config.d_model)
        assert ff(x).shape == x.shape

    def test_swiglu_gating(self, tiny_config):
        """Gate path: zero gate → zero output (SwiGLU gate collapses to 0)."""
        ff = FeedForward(tiny_config)
        with torch.no_grad():
            ff.w1.weight.zero_()  # gate projection → 0
        x = torch.randn(1, 4, tiny_config.d_model)
        out = ff(x)
        assert out.abs().max() < 1e-4


# ── Full Model ─────────────────────────────────────────────────────────────────

class TestMythos:
    def test_forward_shape(self, tiny_model, tiny_config):
        B, T = 2, 16
        x = torch.randint(0, tiny_config.vocab_size, (B, T))
        logits, _ = tiny_model(x)
        assert logits.shape == (B, T, tiny_config.vocab_size)

    def test_forward_no_cache(self, tiny_model, tiny_config):
        """Without use_cache, second return value is None."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 8))
        logits, kvs = tiny_model(x, use_cache=False)
        assert kvs is None

    def test_forward_with_cache(self, tiny_model, tiny_config):
        """With use_cache, second return value is a list of per-layer KV tuples."""
        x = torch.randint(0, tiny_config.vocab_size, (1, 8))
        logits, kvs = tiny_model(x, use_cache=True)
        assert kvs is not None
        assert len(kvs) == tiny_config.n_layers
        assert len(kvs[0]) == 2  # (k, v)

    def test_weight_tying(self, tiny_model):
        """Embedding and output weights must be the same tensor."""
        assert tiny_model.embedding.weight is tiny_model.output.weight

    def test_param_count(self, tiny_model):
        """Sanity check that param count is plausible."""
        n = tiny_model.get_num_params()
        assert n > 0
        assert n < 10_000_000  # tiny model < 10M

    def test_autoregressive_causality(self, tiny_model, tiny_config):
        """
        Changing token at position i should NOT affect logits at positions < i.
        (Causal masking check)
        """
        B, T = 1, 10
        x = torch.randint(0, tiny_config.vocab_size, (B, T))
        x2 = x.clone()
        x2[0, -1] = (x[0, -1] + 1) % tiny_config.vocab_size  # change last token

        with torch.no_grad():
            logits1, _ = tiny_model(x)
            logits2, _ = tiny_model(x2)

        # All positions before T-1 must be identical
        assert torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5), \
            "Causal masking broken: logits changed at positions before modified token"

    def test_no_nan_in_logits(self, tiny_model, tiny_config):
        x = torch.randint(0, tiny_config.vocab_size, (2, 8))
        logits, _ = tiny_model(x)
        assert not logits.isnan().any()

    def test_loss_decreasing(self, tiny_model, tiny_config):
        """Overfit on a fixed batch: loss must decrease over 20 steps."""
        torch.manual_seed(42)
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-2)
        tiny_model.train()

        x = torch.randint(0, tiny_config.vocab_size, (4, 9))
        first_loss = None
        last_loss = None

        for step in range(20):
            logits, _ = tiny_model(x[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, tiny_config.vocab_size),
                x[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 0:
                first_loss = loss.item()
            last_loss = loss.item()

        assert last_loss < first_loss, f"Loss did not decrease over 20 steps: {first_loss:.4f} → {last_loss:.4f}"


# ── Config loading ─────────────────────────────────────────────────────────────

class TestConfigs:
    @pytest.mark.parametrize("config_path", [
        "configs/model/debug.yaml",
        "configs/model/150m.yaml",
        "configs/model/base_500m.yaml",
    ])
    def test_config_loads(self, config_path):
        raw = load_config(ROOT / config_path)
        cfg = ModelConfig.from_dict(raw["model"])
        assert cfg.d_model > 0
        assert cfg.n_layers > 0
        assert cfg.n_heads % cfg.n_kv_heads == 0

    @pytest.mark.parametrize("config_path", [
        "configs/model/debug.yaml",
        "configs/model/150m.yaml",
    ])
    def test_model_instantiates(self, config_path):
        raw = load_config(ROOT / config_path)
        cfg = ModelConfig.from_dict(raw["model"])
        model = Mythos(cfg)
        assert model.get_num_params() > 0
