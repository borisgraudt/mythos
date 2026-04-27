#!/usr/bin/env python3
"""
export_gguf.py — Export Mythos to GGUF format for Ollama / llama.cpp.

Steps:
  1. Export weights to safetensors
  2. Run llama.cpp convert script → GGUF
  3. Quantize (Q4_K_M recommended for 500M)

Usage:
  python scripts/export_gguf.py \\
      --checkpoint checkpoints/base_500m/best.pt \\
      --model_config configs/model/base_500m.yaml \\
      --tokenizer data/tokenizer/tokenizer.json \\
      --quantize q4_k_m

Then add to Ollama:
  ollama create mythos -f Modelfile
  ollama run mythos
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mythos.core.transformer import Mythos, ModelConfig
from mythos.training.checkpoint import load_checkpoint
from mythos.utils.config import load_config


# HuggingFace-style config.json that llama.cpp understands
def write_hf_config(config: ModelConfig, out_dir: Path):
    hf_cfg = {
        "architectures": ["LlamaForCausalLM"],   # compatible architecture
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": config.d_model,
        "intermediate_size": config.d_ff,
        "max_position_embeddings": config.max_seq_len,
        "model_type": "llama",
        "num_attention_heads": config.n_heads,
        "num_hidden_layers": config.n_layers,
        "num_key_value_heads": config.n_kv_heads,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "vocab_size": config.vocab_size,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(hf_cfg, f, indent=2)
    print(f"HF config written: {out_dir / 'config.json'}")


def export_safetensors(model: Mythos, out_dir: Path):
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Install: pip install safetensors")
        sys.exit(1)

    # Remap keys to HuggingFace LLaMA naming so llama.cpp can convert
    state = model.state_dict()
    hf_state = {}

    hf_state["model.embed_tokens.weight"] = state["embedding.weight"]
    hf_state["model.norm.weight"] = state["norm.weight"]
    # lm_head shares weights with embed_tokens (weight tying)
    hf_state["lm_head.weight"] = state["embedding.weight"]

    for i in range(model.config.n_layers):
        prefix = f"layers.{i}"
        hf_prefix = f"model.layers.{i}"

        hf_state[f"{hf_prefix}.input_layernorm.weight"] = state[f"{prefix}.attn_norm.weight"]
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = state[f"{prefix}.ffn_norm.weight"]

        hf_state[f"{hf_prefix}.self_attn.q_proj.weight"] = state[f"{prefix}.attention.wq.weight"]
        hf_state[f"{hf_prefix}.self_attn.k_proj.weight"] = state[f"{prefix}.attention.wk.weight"]
        hf_state[f"{hf_prefix}.self_attn.v_proj.weight"] = state[f"{prefix}.attention.wv.weight"]
        hf_state[f"{hf_prefix}.self_attn.o_proj.weight"] = state[f"{prefix}.attention.wo.weight"]

        hf_state[f"{hf_prefix}.mlp.gate_proj.weight"] = state[f"{prefix}.feed_forward.w1.weight"]
        hf_state[f"{hf_prefix}.mlp.down_proj.weight"] = state[f"{prefix}.feed_forward.w2.weight"]
        hf_state[f"{hf_prefix}.mlp.up_proj.weight"]   = state[f"{prefix}.feed_forward.w3.weight"]

    out_file = out_dir / "model.safetensors"
    save_file(hf_state, str(out_file))
    print(f"Safetensors saved: {out_file}")
    return out_file


def convert_to_gguf(hf_dir: Path, gguf_dir: Path, llama_cpp_path: Path):
    """Run llama.cpp convert script."""
    gguf_dir.mkdir(parents=True, exist_ok=True)
    out_file = gguf_dir / "mythos-500m-f16.gguf"

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"llama.cpp convert script not found: {convert_script}")
        print("Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
        sys.exit(1)

    cmd = [
        sys.executable, str(convert_script),
        str(hf_dir),
        "--outfile", str(out_file),
        "--outtype", "f16",
    ]
    print(f"Converting to GGUF: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"GGUF saved: {out_file}")
    return out_file


def quantize_gguf(gguf_file: Path, llama_cpp_path: Path, quant: str = "q4_k_m") -> Path:
    """Quantize GGUF with llama.cpp quantize binary."""
    quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print(f"llama-quantize not found. Build llama.cpp first: cd llama.cpp && cmake -B build && cmake --build build")
        sys.exit(1)

    out_file = gguf_file.parent / f"mythos-500m-{quant}.gguf"
    cmd = [str(quantize_bin), str(gguf_file), str(out_file), quant.upper()]
    print(f"Quantizing ({quant}): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Quantized: {out_file} ({out_file.stat().st_size / 1e6:.0f} MB)")
    return out_file


def parse_args():
    parser = argparse.ArgumentParser(description="Export Mythos to GGUF for Ollama")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_config", type=Path, default=ROOT / "configs/model/base_500m.yaml")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--llama_cpp", type=Path, default=Path("../llama.cpp"), help="Path to llama.cpp repo")
    parser.add_argument("--quantize", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization type (q4_k_m = best quality/size ratio)")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "export")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading model...")
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_config = raw_ckpt.get("config", {})
    if saved_config:
        config = ModelConfig.from_dict(saved_config)
        print(f"Config loaded from checkpoint (vocab={config.vocab_size})")
    else:
        raw = load_config(args.model_config)
        config = ModelConfig.from_dict(raw["model"])
        print(f"Config loaded from YAML (vocab={config.vocab_size})")
    model = Mythos(config)
    model, _, step = load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Loaded step {step} | {model.get_num_params()/1e6:.1f}M params")

    hf_dir = args.out_dir / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    write_hf_config(config, hf_dir)
    export_safetensors(model, hf_dir)

    if args.tokenizer and args.tokenizer.exists():
        import shutil
        shutil.copy(args.tokenizer, hf_dir / "tokenizer.json")

    gguf_dir = args.out_dir / "gguf"
    gguf_f16 = convert_to_gguf(hf_dir, gguf_dir, args.llama_cpp)

    if args.quantize != "f16":
        quantize_gguf(gguf_f16, args.llama_cpp, args.quantize)

    print(f"""
Done! Next steps:
  1. ollama create mythos -f Modelfile
  2. ollama run mythos
  3. ollama push borisgraudt/mythos
""")


if __name__ == "__main__":
    main()
