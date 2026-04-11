"""Multi-Head Attention with Grouped Query Attention (GQA) and KV-cache."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope

# Type alias for cached (key, value) tensors — shape (B, n_kv_heads, T_past, head_dim)
KVCache = Tuple[torch.Tensor, torch.Tensor]


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0,    "d_model must be divisible by n_heads"
        assert config.n_heads % config.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads  # GQA repeat factor
        self.head_dim   = config.d_model // config.n_heads
        self.dropout_p  = config.dropout

        self.wq = nn.Linear(config.d_model, config.n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model,    bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Args:
            x:       (B, T, d_model) — T=1 during cached decoding
            cos/sin: (T, head_dim/2) — sliced at the correct positions by the caller
            mask:    optional attention mask
            past_kv: (past_k, past_v) with shape (B, n_kv_heads, T_past, head_dim)

        Returns:
            output:  (B, T, d_model)
            kv:      (k, v) after appending current step — pass back on next call
        """
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE is applied only to the NEW tokens (positions already in cache are pre-rotated)
        q, k = apply_rope(q, k, cos, sin)

        # Append new K, V to cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, n_kv_heads, T_past+T, head_dim)
            v = torch.cat([past_v, v], dim=2)

        current_kv: KVCache = (k, v)

        # GQA: broadcast KV heads to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # is_causal only during prefill (T > 1); during decoding T=1, no mask needed
        is_causal = (mask is None) and (T > 1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), current_kv
