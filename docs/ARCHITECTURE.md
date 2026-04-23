# Mythos Architecture Reference

## Model Overview

Mythos is a **decoder-only transformer** (autoregressive language model) implementing the best practices from LLaMA, PaLM, and Chinchilla research. Every design decision is backed by empirical evidence.

```
Input tokens (B, T)
    ↓
Token Embedding  [vocab_size → d_model]
    ↓
× 40 TransformerBlocks:
    ├─ RMSNorm → GQA (+ RoPE) → residual
    └─ RMSNorm → SwiGLU FFN   → residual
    ↓
RMSNorm
    ↓
Output Projection  [d_model → vocab_size]  (weight-tied with embedding)
    ↓
Logits (B, T, vocab_size)
```

---

## Hyperparameters

### 500M Model

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `vocab_size` | 32,000 | BPE; ~4.7 bytes/token; standard across LLaMA/Mistral |
| `d_model` | 1,024 | Hidden dimension |
| `n_layers` | 40 | Depth; more layers = richer representations |
| `n_heads` | 16 | Query heads; head_dim = 64 |
| `n_kv_heads` | 8 | GQA: 2× smaller KV cache |
| `d_ff` | 2,816 | SwiGLU hidden; maintains FLOP budget vs 4×d_model |
| `max_seq_len` | 2,048 | Context window |
| `dropout` | 0.0 | No dropout in pretraining (Chinchilla practice) |
| `rope_theta` | 10,000 | RoPE frequency base (LLaMA default) |

**Parameter count**: ~505M (embedding weight-tied → net ~472M unique params)

---

## Components

### 1. Grouped Query Attention (`src/core/attention.py`)

Reduces KV cache memory by grouping multiple query heads per KV head:

```
Q: (B, T, 16 heads, 64 dim)   ← n_heads = 16
K: (B, T,  8 heads, 64 dim)   ← n_kv_heads = 8
V: (B, T,  8 heads, 64 dim)   ← same
```

Each pair of Q heads shares one (K, V) head. K and V are repeated at inference:

```python
k = k.repeat_interleave(n_heads // n_kv_heads, dim=2)  # 8 → 16
```

**Trade-off**: KV cache is 2× smaller. No perplexity loss (Ainslie et al. 2023).

---

### 2. Rotary Position Embeddings (`src/core/rope.py`)

Position is encoded by rotating Q and K vectors in complex space:

```python
cos, sin = build_rope_cache(max_seq_len, head_dim, rope_theta)
q = apply_rope(q, cos[:T], sin[:T])
k = apply_rope(k, cos[:T], sin[:T])
```

- Zero learnable parameters
- Attention score between tokens at positions m, n depends only on |m - n|
- Tested to generalize beyond training length (2048 → 4096+)

---

### 3. SwiGLU Feed-Forward (`src/core/mlp.py`)

```python
def forward(self, x):
    gate = F.silu(self.w1(x))  # (B, T, d_ff)
    up   = self.w3(x)          # (B, T, d_ff)
    return self.w2(gate * up)  # element-wise gate, then project back
```

Three linear projections: `w1` (gate), `w3` (up), `w2` (down).  
`d_ff = 2816` chosen so `3 × d_model × d_ff` ≈ `2 × d_model × 4×d_model`.

**Quality gain**: +10% over GeLU at the same FLOP budget (Shazeer 2020).

---

### 4. RMSNorm (`src/core/norms.py`)

```python
def forward(self, x):
    rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
    return x / rms * self.weight
```

- No bias, no mean subtraction
- 10–15% faster than LayerNorm
- Same training stability at modern scales (Zhang & Sennrich 2019)

Applied **pre-norm** (before attention and FFN, not after).

---

### 5. Weight Tying

```python
self.output.weight = self.embedding.weight  # shares same tensor
```

Input embedding and output projection share the same matrix. Saves 32M parameters with no quality loss.

---

### 6. Weight Initialization (`transformer.py: _init_weights`)

```python
std = 0.02
residual_scale = std / math.sqrt(2 * n_layers)

nn.init.normal_(linear.weight, mean=0, std=std)

# Output projections of attention and FFN scaled down:
nn.init.normal_(block.attention.wo.weight, mean=0, std=residual_scale)
nn.init.normal_(block.feed_forward.w2.weight, mean=0, std=residual_scale)
```

Residual scaling prevents variance explosion in deep networks (GPT-2 technique, Radford et al. 2019).

---

## Scaling Variants

| Config | Params | Layers | `d_model` | File |
|--------|--------|--------|-----------|------|
| debug  | ~1M    | 2      | 64        | `configs/model/debug.yaml` |
| 150M   | ~150M  | 12     | 768       | `configs/model/150m.yaml` |
| 300M   | ~300M  | 24     | 896       | `configs/model/300m.yaml` |
| 500M   | ~505M  | 40     | 1024      | `configs/model/base_500m.yaml` |

Same architecture, same code — only YAML config changes.

---

## References

- Su et al. (2021) — RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Ainslie et al. (2023) — GQA: "GQA: Training Generalized Multi-Query Transformers"
- Zhang & Sennrich (2019) — RMSNorm: "Root Mean Square Layer Normalization"
- Shazeer (2020) — SwiGLU: "GLU Variants Improve Transformer"
- Touvron et al. (2023) — LLaMA architecture reference
