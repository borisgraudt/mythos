<div align="center">

# Mythos

**A 500M-parameter decoder-only language model, built from scratch.**

[![Tests](https://github.com/borisgraudt/mythos/actions/workflows/tests.yml/badge.svg)](https://github.com/borisgraudt/mythos/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org)

</div>

---

Mythos is a clean, research-grade implementation of a modern transformer language model. Every component — attention, tokenizer, training loop, inference engine — is written from scratch with no black-box dependencies.

The goal is not a toy: the full 500M model uses the same architectural techniques as LLaMA 3 (GQA, SwiGLU, RoPE) and trains on the same data sources as top open-source models.

## Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Attention | Grouped Query Attention (16Q / 8KV) | 2× smaller KV cache vs MHA |
| Activation | SwiGLU | +~10% perplexity over GeLU at equal FLOPs |
| Position | RoPE (θ = 10,000) | No learned params; length extrapolation |
| Normalization | RMSNorm (pre-norm) | 10–15% faster than LayerNorm |
| Weight tying | Embedding ↔ output matrix | Saves 33M params at 32K vocab |

Full hyperparameter breakdowns are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).  
Design rationale with citations is in [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Model Variants

| Model | Params | Layers | d_model | Heads (Q/KV) | Config |
|-------|--------|--------|---------|--------------|--------|
| Debug | 1M | 2 | 64 | 4/2 | built-in |
| Mythos-150M | 150M | 18 | 768 | 12/6 | `configs/model/150m.yaml` |
| Mythos-500M | 500M | 40 | 1024 | 16/8 | `configs/model/base_500m.yaml` |

## Requirements

```
Python >= 3.11
PyTorch >= 2.5
```

```bash
git clone https://github.com/borisgraudt/mythos
cd mythos
pip install -e ".[dev]"
```

## Quick Start

**Verify the architecture (1M model, 100 steps, dummy data):**
```bash
python scripts/train.py --mode debug
```

**Train on real data (150M model):**
```bash
# Step 1: download and process data
python scripts/prepare_data.py --debug   # small dataset for testing
python scripts/prepare_data.py           # full FineWeb + Code pipeline

# Step 2: train
python scripts/train.py --mode data --data data/debug
```

**Generate text:**
```bash
python scripts/infer.py \
  --checkpoint checkpoints/150m_debug_v2/final.pt \
  --tokenizer data/debug/tokenizer/tokenizer.json \
  --prompt "Once upon a time"
```

**Run all tests:**
```bash
make test   # 34 tests
```

## Project Layout

```
mythos/
├── src/
│   ├── core/         # transformer.py, attention.py, mlp.py, norms.py, rope.py
│   ├── training/     # trainer.py, loop.py, optimizer.py, scheduler.py, checkpoint.py
│   ├── inference/    # generate.py, sampler.py
│   └── utils/        # config.py, device.py, logging.py
├── data/
│   ├── dataset.py    # ShardDataset + DataPipeline
│   └── pipelines/    # clean.py, deduplicate.py, tokenize.py
├── scripts/
│   ├── train.py      # training entry point
│   ├── infer.py      # interactive generation
│   ├── prepare_data.py  # download → clean → tokenize → encode → split
│   ├── eval.py       # benchmarks (perplexity, LAMBADA, MMLU)
│   ├── export_hf.py       # upload weights to HuggingFace Hub (custom format)
│   ├── export_llama_hf.py # upload as LlamaForCausalLM (transformers / vLLM / TGI)
│   └── export_gguf.py     # convert to GGUF for Ollama / llama.cpp
├── configs/
│   ├── model/        # debug.yaml, 150m.yaml, base_500m.yaml
│   └── training/     # base.yaml
├── tests/
│   ├── test_model.py    # 23 architecture tests
│   └── test_training.py # 11 training loop tests
├── notebooks/
│   └── demo.ipynb          # load from HF + generation + attention viz
├── app.py                  # Gradio demo for HuggingFace Spaces
└── docs/
    ├── ARCHITECTURE.md
    ├── RESEARCH.md
    ├── TRAINING.md         # reproducible 500M training recipe
    └── results.md
```

## Data Pipeline

```
HuggingFace (wikimedia/wikipedia, codeparrot/github-code)
     │
     ▼
  download_and_clean()     ← language filter, dedup, quality score
     │
     ▼
  train_tokenizer()        ← BPE, 32K vocab (tokenizers library)
     │
     ▼
  encode_shards()          ← tokenize all shards → binary .bin files
     │
     ▼
  split()                  ← 80% train / 10% val / 10% test
```

## Hardware

The full 500M model trains in ~3 days on a MacBook M2 16GB (bfloat16, gradient checkpointing).

| Configuration | Memory | Steps/sec |
|--------------|--------|-----------|
| fp32 | ~8 GB | ~0.7 |
| bf16 | ~4 GB | ~1.4 |
| bf16 + grad checkpoint | ~2.5 GB | ~0.9 |

_100K steps, batch_size=4, seq_len=512 on Apple M2._

## Export

**HuggingFace Hub (LLaMA-compatible — recommended):**
```bash
python scripts/export_llama_hf.py \
  --checkpoint checkpoints/base_500m/final.pt \
  --tokenizer data/medium/tokenizer/tokenizer.json \
  --repo your-username/mythos
```
This uploads the weights as `LlamaForCausalLM`, unlocking `AutoModelForCausalLM.from_pretrained(...)`,
HuggingFace Inference API, vLLM, TGI, and the HF "Try it" widget — no custom code needed.

**Ollama (via GGUF):**
```bash
python scripts/export_gguf.py --checkpoint checkpoints/base_500m/final.pt --quantize q4_k_m
ollama create mythos -f Modelfile
ollama run mythos
```

## Citation

```bibtex
@software{graudt2026mythos,
  author  = {Graudt, Boris},
  title   = {Mythos: A 500M Parameter Language Model from Scratch},
  year    = {2026},
  url     = {https://github.com/borisgraudt/mythos},
  license = {MIT}
}
```

## License

MIT — see [LICENSE](LICENSE).
