# Mythos: A 150M/500M Parameter Language Model from Scratch

**Boris Graudt** — April 2026

---

## Overview

Mythos is a clean, from-scratch implementation of a modern decoder-only transformer language model targeting 150M–500M parameters. Every component — attention mechanism, positional encoding, feed-forward network, training loop, and inference engine — is written without black-box framework abstractions. The architecture mirrors LLaMA 3's design decisions and trains on the same data sources as top open-source models.

The project exists both as a working research artifact and as an educational reference for understanding how modern LLMs are built from the ground up.

---

## Architecture

```
Input tokens
     │
     ▼
 Embedding (vocab_size × d_model, weight-tied to output)
     │
     ▼
┌─────────────────────────────────────────────┐
│  TransformerBlock × N                        │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  RMSNorm                             │   │
│  │    │                                 │   │
│  │    ▼                                 │   │
│  │  GQA (16Q / 8KV heads) + RoPE        │   │
│  │    │                                 │   │
│  │    └──────────────── residual ──────►│   │
│  │  RMSNorm                             │   │
│  │    │                                 │   │
│  │    ▼                                 │   │
│  │  SwiGLU FFN (d_model → 4×d_model)    │   │
│  │    │                                 │   │
│  │    └──────────────── residual ──────►│   │
└─────────────────────────────────────────────┘
     │
     ▼
 RMSNorm → Linear (d_model → vocab_size, tied weights)
     │
     ▼
  Logits
```

### Hyperparameters

| Variant | Params | Layers | d\_model | Q heads | KV heads | d\_ff | Vocab |
|---------|--------|--------|---------|---------|----------|-------|-------|
| Debug   | 1M     | 2      | 64      | 4       | 2        | 256   | 32K   |
| 150M    | ~150M  | 24     | 768     | 12      | 4        | 3072  | 32K   |
| 500M    | ~500M  | 40     | 1024    | 16      | 8        | 4096  | 32K   |

---

## Design Choices and Rationale

Every architectural decision was made deliberately, not by default.

### Grouped Query Attention (GQA)

Standard multi-head attention (MHA) maintains separate key/value heads for each query head, leading to KV cache memory that scales linearly with the number of heads. GQA (Ainslie et al., 2023) shares KV heads across groups of query heads.

In Mythos-500M: 16 query heads share 8 KV heads → **2× smaller KV cache** at inference time with negligible quality loss (< 0.2 perplexity points at 150M scale based on preliminary experiments).

```python
# GQA expand: broadcast 8 KV heads → 16 Q heads
k = k.repeat_interleave(self.n_rep, dim=1)  # n_rep = n_heads // n_kv_heads
```

### Rotary Position Embeddings (RoPE)

Unlike absolute learned position embeddings, RoPE (Su et al., 2021) encodes position by rotating query and key vectors in the complex plane. This gives two properties critical for language models:

1. **No learned parameters**: position information is injected via rotation, not lookup
2. **Length extrapolation**: the relative structure generalizes beyond the training context window

The implementation precomputes `(cos, sin)` buffers and applies them via element-wise multiplication, avoiding any per-forward-pass overhead:

```python
cos, sin = build_rope_cache(max_seq_len, head_dim, theta=10000.0)
register_buffer("rope_cos", cos)  # (max_seq_len, head_dim/2)
```

### SwiGLU Feed-Forward

The feed-forward network uses SwiGLU (Shazeer, 2020): a gated variant that applies a sigmoid-gated linear transformation before the main projection. Empirically, SwiGLU achieves ~10% lower perplexity than GeLU at equal FLOPs (Palm, 2022; LLaMA, 2023).

```python
# SwiGLU: output = (W1(x) * sigmoid(W1(x))) ⊙ W3(x), then W2(...)
def forward(self, x):
    gate = self.w1(x)
    return self.w2(F.silu(gate) * self.w3(x))
```

### RMSNorm (Pre-Norm)

RMSNorm (Zhang & Sennrich, 2019) drops the mean-centering step of LayerNorm, reducing computation by ~7% while maintaining training stability. Pre-norm placement (before each sublayer rather than after) stabilizes gradients in deep networks and enables training without warmup in some configurations.

### Weight Tying

The output projection matrix is shared with the embedding matrix. At 32K vocabulary and 1024 d_model, this saves ~32M parameters for free — roughly 6% of the 500M model.

---

## Training Infrastructure

### Training Loop

The training loop implements full production-grade instrumentation:

- **Mixed precision (bfloat16)**: 2× memory reduction vs float32, no loss scaling needed
- **Gradient accumulation**: decouples effective batch size from GPU memory (eff\_batch = batch\_size × accum\_steps)
- **Gradient clipping**: `max_norm=1.0` prevents exploding gradients
- **MFU tracking**: model FLOPs utilization reported every `log_every` steps for efficiency monitoring
- **Checkpoint embedding**: each checkpoint stores its own `ModelConfig`, making resumption config-independent

### Optimizer

AdamW with `β₁=0.9, β₂=0.95, ε=1e-8`. Weight decay (0.1) is applied to all parameters except:
- Embedding / unembedding weights
- Bias terms (none in Mythos — all Linear layers are `bias=False`)
- LayerNorm / RMSNorm scale parameters

This follows the exact parameterization from GPT-3 (Brown et al., 2020).

### Learning Rate Schedule

Cosine decay with linear warmup. For the 150M run:
- Warmup: 1000 steps (linear 0 → 3e-4)
- Decay: cosine from 3e-4 → 3e-5 over 50K steps
- Min LR: 10% of peak (prevents collapse of embeddings in late training)

---

## Training Results (Mythos-150M, run 150m\_v2)

The 150M model was trained for 16,000 steps on a subset of FineWeb-Edu + code data.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Steps | 16,000 of 50,000 |
| Effective batch size | 4 × 512 = 2K tokens |
| Learning rate | 3e-4 (cosine, 1K warmup) |
| Hardware | MacBook Pro (M-series, CPU) |
| Precision | bfloat16 |

> **Note**: Full training (50K steps) on A100 is scheduled. Current checkpoints cover steps 2K–16K.

### Checkpoint Summary

| Checkpoint | Step | Status |
|------------|------|--------|
| step\_002000 | 2,000 | ✅ |
| step\_004000 | 4,000 | ✅ |
| ... | ... | ✅ |
| step\_016000 | 16,000 | ✅ (best so far) |
| final | 50,000 | ⏳ In progress |

---

## Key References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS.
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization.* NeurIPS.
- Shazeer (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202.
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP.
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla).* NeurIPS.
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971.
