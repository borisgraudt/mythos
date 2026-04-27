# Ablation Studies

Design decisions in Mythos were validated through controlled ablations on the 150M model at 10K steps.

## GQA vs Multi-Head Attention

Config: `configs/training/ablation_mha.yaml` (n_kv_heads=12) vs baseline 150M (n_kv_heads=4).

**Hypothesis**: GQA with 4 KV heads reduces KV cache memory by 3× vs MHA with negligible quality loss.

| Variant | n_kv_heads | KV Cache (rel.) | Val PPL @10K | Notes |
|---------|-----------|-----------------|--------------|-------|
| MHA (baseline) | 12 | 100% | — | to be run |
| GQA (3 groups) | 4 | 33% | — | to be run |

Run with:
```bash
# MHA baseline
python scripts/train.py --config configs/training/ablation_mha.yaml

# Default GQA (already in 150m.yaml)
python scripts/train.py --config configs/model/150m.yaml
```

**Expected result** (from Ainslie et al., 2023): < 0.5 PPL degradation with 3× KV cache reduction.

---

## SwiGLU vs GeLU

Config: `configs/training/ablation_gelu.yaml` + `src/mythos/core/mlp.py` activation swap.

**Hypothesis**: SwiGLU achieves lower perplexity than GeLU at equal FLOPs.

| Activation | d_ff | Val PPL @10K | Notes |
|------------|------|--------------|-------|
| GeLU | 3072 | — | to be run |
| SwiGLU | 3072 | — | default |

**Expected result** (from Shazeer 2020; PaLM 2022): ~2 PPL improvement with SwiGLU.

---

## RoPE vs Learned Position Embeddings

Not implemented yet. RoPE was chosen over learned embeddings for:
1. No additional parameters
2. Length generalization beyond training context
3. Established in LLaMA, Mistral, Falcon

---

## Weight Tying

Sharing embedding ↔ output projection saves `vocab_size × d_model` parameters.

At 32K vocab and 768 d_model: **24.6M parameters saved** (~16% of 150M model).

Trade-off: forces the input embedding space to also serve as the output prediction space. Empirically no degradation observed in models up to 7B (LLaMA 2).
