"""
Mythos — full Transformer model.

Design choices:
  - RMSNorm       : cheaper than LayerNorm, no mean-centering (Zhang & Sennrich 2019)
  - RoPE          : relative position via rotation, extrapolates to longer seqs (Su et al. 2021)
  - SwiGLU FFN    : gated activation, ~10% better than ReLU at equal FLOP (Shazeer 2020)
  - GQA           : grouped-query attention — fewer KV heads → less memory (Ainslie et al. 2023)
  - Weight tying  : share embedding ↔ output matrix → −vocab_size×d_model params free
  - No bias       : empirically cleaner at scale
  - Pre-norm      : residual stream stays clean; easier to train deep nets
"""

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import Attention, KVCache
from .mlp import FeedForward
from .norms import RMSNorm
from .rope import build_rope_cache


@dataclass
class ModelConfig:
    vocab_size:  int   = 32000
    d_model:     int   = 1024
    n_layers:    int   = 40
    n_heads:     int   = 16
    n_kv_heads:  int   = 8
    d_ff:        int   = 4096
    max_seq_len: int   = 2048
    dropout:     float = 0.0
    norm_eps:    float = 1e-5
    rope_theta:  float = 10000.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerBlock(nn.Module):
    """Pre-norm residual block: norm → sublayer → residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention    = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attn_norm    = RMSNorm(config.d_model, config.norm_eps)
        self.ffn_norm     = RMSNorm(config.d_model, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        attn_out, new_kv = self.attention(self.attn_norm(x), cos, sin, mask, past_kv)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv


class Mythos(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers    = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm      = RMSNorm(config.d_model, config.norm_eps)
        self.output    = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: output projection reuses embedding matrix.
        self.output.weight = self.embedding.weight

        # RoPE cache — registered as buffer so it moves with .to(device)
        cos, sin = build_rope_cache(config.max_seq_len, config.d_model // config.n_heads, config.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self):
        """GPT-style init with residual scaling."""
        std = 0.02
        residual_scale = std / math.sqrt(2 * self.config.n_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        for block in self.layers:
            nn.init.normal_(block.attention.wo.weight,    mean=0.0, std=residual_scale)
            nn.init.normal_(block.feed_forward.w2.weight, mean=0.0, std=residual_scale)

    def forward(
        self,
        x: torch.Tensor,
        past_kvs: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCache]]]:
        """
        Args:
            x:         (B, T) token ids
            past_kvs:  list of per-layer KV caches from the previous step (inference only)
            use_cache: if True, return updated KV caches alongside logits

        Returns:
            logits:    (B, T, vocab_size)
            new_kvs:   updated per-layer caches (only when use_cache=True, else None)
        """
        B, T = x.shape
        assert T <= self.config.max_seq_len, f"Input length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # Determine starting position (non-zero when reusing cache)
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0

        h   = self.embedding(x)
        cos = self.rope_cos[past_len : past_len + T]
        sin = self.rope_sin[past_len : past_len + T]

        new_kvs: List[KVCache] = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            h, new_kv = layer(h, cos, sin, past_kv=past_kv)
            if use_cache:
                new_kvs.append(new_kv)

        h = self.norm(h)
        logits = self.output(h)

        return logits, (new_kvs if use_cache else None)

    def get_num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embedding.weight.numel()
        return n

    @classmethod
    def from_config_dict(cls, d: dict) -> "Mythos":
        return cls(ModelConfig.from_dict(d))
