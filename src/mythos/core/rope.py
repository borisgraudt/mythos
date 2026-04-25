"""
Rotary Positional Embeddings (real-valued — MPS compatible).

Real-valued implementation avoids torch.view_as_complex which is
not supported on Apple MPS.
"""

from typing import Optional

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) tensors of shape (seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)            # (seq_len, half)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k : (B, n_heads, T, head_dim)
    cos, sin : (T, head_dim)  →  broadcast to (1, 1, T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
