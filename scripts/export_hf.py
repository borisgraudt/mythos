#!/usr/bin/env python3
"""
export_hf.py — Export Mythos to the Hugging Face Hub in LlamaForCausalLM format.

Mythos is architecturally identical to LLaMA (GQA + SwiGLU + RoPE half-rotation +
RMSNorm pre-norm + weight tying). Every Mythos tensor maps 1:1 to a
LlamaForCausalLM tensor, so the uploaded repo is natively compatible with:

  - transformers.AutoModelForCausalLM.from_pretrained("borisgraudt/mythos")
  - HF Inference API widget ("Try it" panel on the model page)
  - vLLM / TGI / text-generation-inference
  - llama.cpp / Ollama (via GGUF conversion)

This is the same export path used by Google Gemma, Meta LLaMA, and Mistral
checkpoints on the Hub: a Llama-shaped config + safetensors weights + tokenizer.

Usage:
  huggingface-cli login
  python scripts/export_hf.py \\
      --checkpoint checkpoints/150m_v2/final.pt \\
      --tokenizer data/debug/tokenizer/tokenizer.json \\
      --repo borisgraudt/mythos
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mythos.core.transformer import Mythos, ModelConfig
from mythos.training.checkpoint import load_checkpoint


# ─── Weight remap: Mythos → LlamaForCausalLM ────────────────────────────────
#
# Mythos layer naming                       HF Llama naming
# ────────────────────────────────          ─────────────────────────────────
# embedding.weight                          model.embed_tokens.weight
# layers.N.attn_norm.weight                 model.layers.N.input_layernorm.weight
# layers.N.attention.wq.weight              model.layers.N.self_attn.q_proj.weight
# layers.N.attention.wk.weight              model.layers.N.self_attn.k_proj.weight
# layers.N.attention.wv.weight              model.layers.N.self_attn.v_proj.weight
# layers.N.attention.wo.weight              model.layers.N.self_attn.o_proj.weight
# layers.N.ffn_norm.weight                  model.layers.N.post_attention_layernorm.weight
# layers.N.feed_forward.w1.weight           model.layers.N.mlp.gate_proj.weight
# layers.N.feed_forward.w3.weight           model.layers.N.mlp.up_proj.weight
# layers.N.feed_forward.w2.weight           model.layers.N.mlp.down_proj.weight
# norm.weight                               model.norm.weight
# output.weight (tied to embedding)         lm_head.weight (tied)

def remap_to_llama(state_dict: dict, n_layers: int) -> dict:
    out = {}
    out["model.embed_tokens.weight"] = state_dict["embedding.weight"]
    out["model.norm.weight"] = state_dict["norm.weight"]

    for i in range(n_layers):
        src = f"layers.{i}"
        dst = f"model.layers.{i}"
        out[f"{dst}.input_layernorm.weight"]           = state_dict[f"{src}.attn_norm.weight"]
        out[f"{dst}.post_attention_layernorm.weight"]  = state_dict[f"{src}.ffn_norm.weight"]
        out[f"{dst}.self_attn.q_proj.weight"] = state_dict[f"{src}.attention.wq.weight"]
        out[f"{dst}.self_attn.k_proj.weight"] = state_dict[f"{src}.attention.wk.weight"]
        out[f"{dst}.self_attn.v_proj.weight"] = state_dict[f"{src}.attention.wv.weight"]
        out[f"{dst}.self_attn.o_proj.weight"] = state_dict[f"{src}.attention.wo.weight"]
        out[f"{dst}.mlp.gate_proj.weight"] = state_dict[f"{src}.feed_forward.w1.weight"]
        out[f"{dst}.mlp.up_proj.weight"]   = state_dict[f"{src}.feed_forward.w3.weight"]
        out[f"{dst}.mlp.down_proj.weight"] = state_dict[f"{src}.feed_forward.w2.weight"]

    # Weight tying: lm_head shares embedding. Keep both so the safetensors saver
    # does not complain about shared memory — transformers will dedupe on load.
    return out


def build_llama_config(mcfg: ModelConfig) -> dict:
    """Mythos ModelConfig → HF LlamaConfig JSON."""
    # Mythos SwiGLU rounds hidden to a multiple of 256:
    hidden = int(2 * mcfg.d_ff / 3)
    intermediate = 256 * math.ceil(hidden / 256)

    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": mcfg.d_model,
        "intermediate_size": intermediate,
        "num_hidden_layers": mcfg.n_layers,
        "num_attention_heads": mcfg.n_heads,
        "num_key_value_heads": mcfg.n_kv_heads,
        "head_dim": mcfg.d_model // mcfg.n_heads,
        "vocab_size": mcfg.vocab_size,
        "max_position_embeddings": mcfg.max_seq_len,
        "rms_norm_eps": mcfg.norm_eps,
        "rope_theta": mcfg.rope_theta,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "tie_word_embeddings": True,
        "use_cache": True,
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "float32",
        "transformers_version": "4.46.0",
    }


def build_tokenizer_config() -> dict:
    """Minimal tokenizer_config.json expected by HF transformers / widget."""
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
        "model_max_length": 2048,
    }


def build_generation_config() -> dict:
    return {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 256,
        "transformers_version": "4.46.0",
    }


def build_model_card(mcfg: ModelConfig, step: int, repo_id: str, is_debug: bool) -> str:
    n_params_m = sum([
        mcfg.vocab_size * mcfg.d_model,
        mcfg.n_layers * 4 * mcfg.d_model * mcfg.d_model,
        mcfg.n_layers * 3 * mcfg.d_model * (256 * math.ceil(int(2 * mcfg.d_ff / 3) / 256)),
    ]) / 1e6
    size_tag = f"{round(n_params_m)}M"

    status = (
        "> ⚠️ **Research preview.** Debug checkpoint — trained on ~21 M tokens "
        "with vocab 3 252 for 5 000 steps. Intended to verify the architecture, "
        "not for downstream use. A production 500 M checkpoint will supersede it."
        if is_debug else
        "> **Production release.** Full pre-training run."
    )

    return f"""---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- pytorch
- causal-lm
- llama
- from-scratch
- pretraining
- gqa
- swiglu
- rope
- rmsnorm
model-index:
- name: Mythos-{size_tag}
  results: []
widget:
- text: "The history of artificial intelligence begins with"
  example_title: "History"
- text: "A transformer is a neural network that"
  example_title: "Architecture"
inference:
  parameters:
    temperature: 0.8
    top_p: 0.9
    max_new_tokens: 128
---

<div align="center">

# Mythos-{size_tag}

**A decoder-only language model built from scratch — LLaMA-compatible weights.**

[![GitHub](https://img.shields.io/badge/GitHub-borisgraudt/mythos-24292e?logo=github)](https://github.com/borisgraudt/mythos)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/borisgraudt/mythos/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg?logo=pytorch)](https://pytorch.org)
[![transformers](https://img.shields.io/badge/🤗%20transformers-compatible-yellow)](https://github.com/huggingface/transformers)

</div>

---

{status}

## Model Summary

Mythos is a LLaMA-style autoregressive transformer implemented **from first principles**
in pure PyTorch — no `transformers` inheritance, no `nn.TransformerBlock`, no shortcuts.
Every component (attention, rotary embeddings, SwiGLU, RMSNorm, the training loop, the
BPE tokenizer, the data pipeline, the KV-cache inference engine) is hand-written in the
reference repository.

This release packages the weights in the **`LlamaForCausalLM`** format so that the model
is natively usable via the standard `transformers`, `vLLM`, `TGI`, and `llama.cpp`
toolchains — no custom code or `trust_remote_code` required.

| | |
|---|---|
| **Developed by** | Boris Graudt |
| **Model type** | Decoder-only causal transformer |
| **Language** | English |
| **License** | MIT |
| **Compatible with** | 🤗 `transformers`, vLLM, TGI, llama.cpp, Ollama |
| **Reference implementation** | [github.com/borisgraudt/mythos](https://github.com/borisgraudt/mythos) |

## Architecture

| Component | Choice | Value |
|---|---|---:|
| Parameters | — | **{round(n_params_m)} M** |
| Hidden layers | Pre-norm decoder blocks | {mcfg.n_layers} |
| Hidden size | `d_model` | {mcfg.d_model} |
| Intermediate size | SwiGLU hidden | {256 * math.ceil(int(2 * mcfg.d_ff / 3) / 256)} |
| Attention heads | Multi-head | {mcfg.n_heads} |
| Key / value heads | **Grouped-Query Attention** | {mcfg.n_kv_heads} |
| Head dim | `d_model / n_heads` | {mcfg.d_model // mcfg.n_heads} |
| Positional encoding | **Rotary (RoPE)** | θ = {int(mcfg.rope_theta):,} |
| Normalization | **RMSNorm** (pre-norm) | ε = {mcfg.norm_eps} |
| Activation | **SwiGLU** | — |
| Tied embeddings | Embedding ↔ LM head | ✅ |
| Vocabulary | ByteLevel BPE | {mcfg.vocab_size:,} |
| Context length | Max sequence | {mcfg.max_seq_len:,} |

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

inputs = tokenizer("The history of artificial intelligence begins with", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.8, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Serving with vLLM

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model {repo_id}
```

### Serving with llama.cpp

```bash
# Convert to GGUF (one-time)
python llama.cpp/convert_hf_to_gguf.py {repo_id.split('/')[-1]}
./llama-cli -m ggml-model-f16.gguf -p "Hello"
```

## Training

### Data

{"- **Corpus:** Wikipedia (English 20231101 snapshot) — 5 000 articles, ~21 M tokens" if is_debug else "- **Corpus:** mixed web + code (details in the GitHub repo)"}
- **Tokenizer:** ByteLevel BPE trained from scratch, vocab size **{mcfg.vocab_size:,}**
- **Training context:** 512 tokens

### Hyperparameters

| | |
|---|---:|
| Steps | {step:,} |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| LR schedule | Cosine decay, 2 000-step warmup |
| Peak learning rate | 3 × 10⁻⁴ |
| Precision | bfloat16 mixed |
| Hardware | {"Apple M2 (MPS)" if is_debug else "A100 40 GB"} |

## Limitations and Intended Use

- **Base model only** — no instruction tuning, no RLHF, no safety alignment.
- English-only; non-English performance is poor.
- May reproduce biases and factual errors from the training distribution.
{"- Tiny vocabulary (3 252 tokens) severely caps fluency — intended as an architecture demo." if is_debug else ""}
- Not suitable for medical, legal, financial, or other high-stakes applications.

## Citation

```bibtex
@software{{graudt2026mythos,
  author  = {{Graudt, Boris}},
  title   = {{Mythos: A Decoder-Only Language Model Built From Scratch}},
  year    = {{2026}},
  url     = {{https://github.com/borisgraudt/mythos}},
  license = {{MIT}}
}}
```

## Acknowledgements

Architecture inspired by **LLaMA** (Touvron et al., 2023) and **Mistral 7B**
(Jiang et al., 2023). Data pipeline follows the **FineWeb** methodology
(Penedo et al., 2024).
"""


# ─── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json (HF tokenizers format)")
    p.add_argument("--repo", type=str, default=None, help="HF repo id, e.g. bgraudt/mythos")
    p.add_argument("--out_dir", type=Path, default=ROOT / "export/hf_llama")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading checkpoint…")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_cfg = raw.get("config")
    if not saved_cfg:
        raise SystemExit("Checkpoint has no embedded config.")

    mcfg = ModelConfig.from_dict(saved_cfg)
    print(f"  config → d_model={mcfg.d_model}  n_layers={mcfg.n_layers}  vocab={mcfg.vocab_size}")

    # Build the model and load weights (to verify the checkpoint is healthy)
    model = Mythos(mcfg)
    model, _, step = load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"  loaded step {step}  |  {model.get_num_params() / 1e6:.1f} M params")

    # Remap state_dict to Llama naming
    print("Remapping tensors → LlamaForCausalLM…")
    llama_state = remap_to_llama(model.state_dict(), mcfg.n_layers)

    # Ensure contiguous tensors (safetensors requirement)
    llama_state = {k: v.detach().contiguous() for k, v in llama_state.items()}

    # Write everything
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for f in ["model.safetensors", "config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "README.md"]:
        (out / f).unlink(missing_ok=True)

    # 1. Weights
    from safetensors.torch import save_file
    save_file(llama_state, out / "model.safetensors", metadata={"format": "pt"})
    print(f"  model.safetensors  ({sum(v.numel() for v in llama_state.values()) / 1e6:.1f} M tensors)")

    # 2. Llama config
    llama_cfg = build_llama_config(mcfg)
    (out / "config.json").write_text(json.dumps(llama_cfg, indent=2))
    print("  config.json")

    # 3. Generation config
    (out / "generation_config.json").write_text(json.dumps(build_generation_config(), indent=2))
    print("  generation_config.json")

    # 4. Tokenizer (bring over as-is; HF tokenizers format works directly)
    shutil.copy(args.tokenizer, out / "tokenizer.json")
    (out / "tokenizer_config.json").write_text(json.dumps(build_tokenizer_config(), indent=2))
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<s>", "eos_token": "</s>",
        "pad_token": "<pad>", "unk_token": "<unk>",
    }, indent=2))
    print("  tokenizer.json / tokenizer_config.json / special_tokens_map.json")

    # 5. Model card
    is_debug = mcfg.vocab_size < 10_000
    repo_id = args.repo or "bgraudt/mythos"
    (out / "README.md").write_text(build_model_card(mcfg, step, repo_id, is_debug))
    print("  README.md")

    print(f"\nLocal export ready: {out.resolve()}")

    if args.dry_run or not args.repo:
        print("Dry run — skipping upload.  Pass --repo bgraudt/mythos to publish.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.repo, exist_ok=True, repo_type="model")
    print(f"\nUploading to https://huggingface.co/{args.repo} …")
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model")
    print(f"Done → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
