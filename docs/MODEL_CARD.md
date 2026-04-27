---
license: mit
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - mythos
  - llama
  - causal-lm
  - from-scratch
  - research
  - gqa
  - swiglu
  - rope
  - rmsnorm
  - pretrained
model-index:
  - name: mythos-150m
    results:
      - task:
          type: text-generation
          name: Causal Language Modeling
        dataset:
          name: WikiText-103 (val)
          type: wikitext
        metrics:
          - type: perplexity
            value: TBD
            name: Perplexity
extra_gated_heading: "Access Mythos on Hugging Face"
extra_gated_description: >-
  Mythos is a research-grade language model. Read the model card before use.
---

<div align="center">

# Mythos

**A decoder-only language model, built from scratch.**

<img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" />
<img src="https://img.shields.io/badge/params-150M%20%2F%20500M-success" alt="Params" />
<img src="https://img.shields.io/badge/arch-LLaMA--style-orange" alt="Architecture" />
<img src="https://img.shields.io/badge/precision-bfloat16-purple" alt="Precision" />
<img src="https://img.shields.io/badge/context-2048-lightgrey" alt="Context" />

</div>

> **Mythos** is a clean-room implementation of a modern decoder-only transformer.
> Every component — attention, tokenizer, training loop, inference engine — is written from scratch with no black-box dependencies, then exported to the LLaMA format for full compatibility with `transformers`, `vLLM` and `TGI`.

---

## ✨ Highlights

- **LLaMA 3-class architecture** — Grouped Query Attention, SwiGLU, RoPE, RMSNorm, weight tying.
- **Drop-in compatible** — loads via `AutoModelForCausalLM` as `LlamaForCausalLM`.
- **Reproducible** — full training recipe, configs, and ablations open-sourced.
- **Two sizes** — 150M (research / education) and 500M (base model).
- **Permissive** — MIT license, weights and code.

---

## 🧠 Model Description

| | |
|---|---|
| **Developed by** | Boris Graudt |
| **Model type** | Causal language model (decoder-only transformer) |
| **Language(s)** | English |
| **License** | MIT |
| **Parent architecture** | LLaMA 3 family |
| **Repository** | [github.com/borisgraudt/mythos](https://github.com/borisgraudt/mythos) |
| **Paper / Report** | [`REPORT.md`](https://github.com/borisgraudt/mythos/blob/main/REPORT.md) |

---

## 🏗 Architecture

| Component | Mythos-150M | Mythos-500M |
|-----------|:-----------:|:-----------:|
| Parameters | ~150 M | ~500 M |
| Layers | 18 | 40 |
| Hidden size (`d_model`) | 768 | 1 024 |
| FFN dim | 3 072 | 4 096 |
| Attention heads (Q / KV) | 12 / 6 | 16 / 8 |
| Head dim | 64 | 64 |
| Vocabulary | 32 000 (BPE) | 32 000 (BPE) |
| Max context | 2 048 | 2 048 |
| Position encoding | RoPE (θ = 10 000) | RoPE (θ = 10 000) |
| Activation | SwiGLU | SwiGLU |
| Normalization | RMSNorm (pre-norm) | RMSNorm (pre-norm) |
| Tied embeddings | ✅ | ✅ |

Design rationale and citations: [`docs/RESEARCH.md`](https://github.com/borisgraudt/mythos/blob/main/docs/RESEARCH.md).

---

## 📚 Training Data

| Source | Notes |
|---|---|
| **FineWeb-Edu** | High-quality educational web text (English subset) |
| **CodeParrot / GitHub Code** | Programming-language pretraining mix |

The data pipeline (`scripts/prepare_data.py`) handles download → language filter → dedup → BPE → binary shards → 80/10/10 split.

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β₁ = 0.9, β₂ = 0.95, wd = 0.1) |
| Learning rate | 3 × 10⁻⁴, cosine decay, 2 K warmup |
| Batch | 4 × 512 tokens / step (gradient accumulation supported) |
| Gradient clipping | 1.0 |
| Precision | bfloat16 (mixed) |
| Gradient checkpointing | optional |
| Hardware | Apple M2 16 GB (150M) / single A100 80 GB (500M) |

Reproducible recipe: [`docs/TRAINING.md`](https://github.com/borisgraudt/mythos/blob/main/docs/TRAINING.md).

---

## 🚀 Usage

### With 🤗 Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "borisgraudt/mythos-150m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Native (from the `mythos` package)

```python
from mythos.core import Mythos
from mythos.training.checkpoint import load_checkpoint
from mythos.inference.generate import generate

model, config = load_checkpoint("checkpoints/150m_v2/best.pt")
model.eval()

ids = generate(model, prompt_ids, max_new_tokens=200, temperature=0.8, top_p=0.9)
```

### Run locally with Ollama (GGUF)

```bash
ollama run borisgraudt/mythos
```

---

## 📈 Evaluation

| Benchmark | Metric | Mythos-150M | Mythos-500M |
|---|---|:---:|:---:|
| WikiText-103 (val) | Perplexity ↓ | _TBD_ | _TBD_ |
| LAMBADA | Accuracy ↑ | _TBD_ | _TBD_ |
| HellaSwag | Accuracy ↑ | _TBD_ | _TBD_ |

_Evaluation harness: `scripts/eval.py`. Numbers will be filled in once full pretraining completes._

---

## 🎯 Intended Use

- **Research** — ablation platform for architectural and training experiments.
- **Education** — concrete reference for how a modern LLM is built end-to-end.
- **Prototyping** — fast iteration before scaling to larger models.

### Out-of-Scope Use

- Production user-facing assistants (no RLHF / instruction tuning).
- Safety-critical decisions, medical or legal advice.
- Generating content in languages other than English.

---

## ⚠️ Limitations & Bias

- Pretrained on a **small subset** of English web and code data; not competitive with frontier models.
- **English-only**, with web and code biases inherited from the source corpora.
- **No alignment** — raw next-token predictor; outputs may be inaccurate, offensive, or unsafe.
- Maximum context **2 048 tokens**.

---

## 🌱 Environmental Impact

| | |
|---|---|
| Hardware | 1 × A100 80 GB / Apple M2 |
| Estimated training time | ~3 days (500M, bf16, grad-ckpt) |
| Cloud provider | RunPod / local |
| Carbon emissions | Estimated with [ML CO₂ Impact](https://mlco2.github.io/impact/) |

---

## 📝 Citation

```bibtex
@software{graudt2026mythos,
  author  = {Graudt, Boris},
  title   = {Mythos: A Language Model from Scratch},
  year    = {2026},
  url     = {https://github.com/borisgraudt/mythos},
  license = {MIT}
}
```

---

## 🙏 Acknowledgements

Mythos draws on the open-source work of LLaMA (Meta AI), the FineWeb dataset (HuggingFace), and the broader open-LLM community.
