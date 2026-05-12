# Limitations

A deliberate, honest account of what Mythos does **not** claim. Updated 2026-04-29.

## Training state

- The shipped 150M checkpoint (`150m_v2`, also on HuggingFace) is **undertrained by ~90×** relative to the Chinchilla-optimal token budget for a 150M model (~33M tokens seen vs ~3B optimal). It generates locally fluent text but is **not** a quality claim at the 150M scale.
- All `150m_v2` training was done on MacBook Air M3 (MPS, bf16). Throughput is bounded by Apple Silicon memory bandwidth; the same code on an A100 is ~30–40× faster.

## Evaluation

- `scripts/eval.py` includes hooks for LAMBADA and MMLU, but published numbers would be misleading at the current training state. They are intentionally omitted from the README and model card.
- Validation perplexity is reported only on the held-out FineWeb-Edu split. No cross-domain perplexity (Wikipedia, code, dialogue) is reported yet.

## Ablations

- `docs/ABLATIONS.md` currently lists hypotheses and configs (GQA vs MHA, SwiGLU vs GeLU) but the result columns are placeholders. Filling them requires the 30M Chinchilla-optimal scaling-laws run; the smoke-test 150M is not a useful ablation substrate (signal would be dominated by undertraining noise, not architectural variation).

## Architectural scope

- No multi-GPU / distributed training path. FSDP or DDP would be required for a serious 500M run; the README's `base_500m.yaml` config is a definition, not a tested recipe.
- No FlashAttention or torch.compile integration yet. Reported throughput is plain PyTorch + MPS.
- No KV-cache implementation in `inference/`. Generation is currently O(T²) per token.
- No quantization-aware training. GGUF q4_k_m export is post-training quantization only.

## Data

- Training data is a small subset of FineWeb-Edu and codeparrot/github-code. No deduplication is run *across* the two sources, only within each. C4-style quality filtering is minimal.
- The tokenizer is trained on the same subset used for training. Vocabulary coverage on out-of-distribution domains (e.g., non-English text) is poor.

## Reproducibility caveats

- Determinism is not guaranteed across hardware (MPS vs CUDA). Bit-exact reproduction requires same OS + PyTorch version + chip.
- The `wandb` logs in `checkpoints/150m_v2_150m_medium_v1/wandb/` reflect ten partial runs from iteration on hyperparameters; the final reported run is the one with the longest checkpoint sequence.

## What this project *does* claim

- A clean, from-scratch implementation of a modern LLaMA-class transformer with full test coverage.
- A reproducible end-to-end pipeline (data → tokenizer → train → eval → HF/GGUF export → demo).
- An upcoming small-scale Chinchilla scaling-laws reproduction (10M / 30M / 80M) — see [`SCALING.md`](SCALING.md). This is the primary scientific artifact.
