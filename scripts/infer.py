#!/usr/bin/env python3
"""
infer.py — Mythos interactive text generation.

Usage:
  # Interactive REPL
  python scripts/infer.py --checkpoint checkpoints/debug/step_000100.pt --mode debug

  # Single prompt
  python scripts/infer.py --checkpoint checkpoints/base_500m/best.pt --prompt "Once upon a time"

  # With custom sampling params
  python scripts/infer.py --checkpoint checkpoints/base_500m/best.pt \\
      --temperature 0.8 --top_p 0.9 --max_tokens 300
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mythos.core.transformer import ModelConfig, Mythos
from mythos.inference.generate import generate, generate_stream
from mythos.training.checkpoint import load_checkpoint
from mythos.utils.config import load_config
from mythos.utils.device import get_device

# ── Debug tokenizer (character-level, for testing without real tokenizer) ────

class CharTokenizer:
    """Minimal character-level tokenizer for debug runs."""
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 100 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if 32 <= i + 32 < 127)


def load_tokenizer(tokenizer_path: Path):
    """Load tokenizer — HuggingFace tokenizers format."""
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(tokenizer_path))
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Mythos text generation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model", type=Path, default=None, help="Model config YAML (optional, auto-detected from checkpoint)")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (omit for interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0=disabled)")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream output token by token")
    return parser.parse_args()


def load_model(args) -> tuple[Mythos, object]:
    device = get_device()

    # Load raw checkpoint to read saved config
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_config = raw_ckpt.get("config", {})

    # Priority: 1) --model flag  2) config saved in checkpoint  3) fallback yaml
    if args.model:
        raw = load_config(args.model)
        config = ModelConfig.from_dict(raw["model"])
    elif saved_config:
        config = ModelConfig.from_dict(saved_config)
    else:
        # Last resort: look for yaml next to checkpoint
        ckpt_dir = args.checkpoint.parent
        candidates = list(ckpt_dir.glob("*.yaml"))
        cfg_path = candidates[0] if candidates else ROOT / "configs/model/base_500m.yaml"
        raw = load_config(cfg_path)
        config = ModelConfig.from_dict(raw["model"])

    model = Mythos(config)
    model, _, step = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tok_path = args.tokenizer or (args.checkpoint.parent / "tokenizer.json")
    tokenizer = load_tokenizer(tok_path) if tok_path and tok_path.exists() else CharTokenizer()

    print(f"Model loaded | step {step} | {model.get_num_params()/1e6:.1f}M params | device: {device}")
    return model, tokenizer


def encode(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text)
        if hasattr(result, "ids"):
            return result.ids  # HuggingFace Tokenizer
        return result


def decode(tokenizer, ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids)
    return str(ids)


def run_generation(model, tokenizer, prompt: str, args):
    ids = encode(tokenizer, prompt)
    input_ids = torch.tensor([ids])

    print(f"\n{'─'*60}")
    print(f"Prompt: {prompt}")
    print(f"{'─'*60}")
    print(prompt, end="", flush=True)

    if args.stream:
        generated = []
        for token_id in generate_stream(
            model, input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        ):
            generated.append(token_id)
            text = decode(tokenizer, [token_id])
            print(text, end="", flush=True)
        print(f"\n{'─'*60}")
    else:
        output_ids = generate(
            model, input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        new_ids = output_ids[0, len(ids):].tolist()
        print(decode(tokenizer, new_ids))
        print(f"{'─'*60}")


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, tokenizer = load_model(args)

    if args.prompt:
        run_generation(model, tokenizer, args.prompt, args)
    else:
        # Interactive REPL
        print("\nMythos Interactive Generation")
        print("Type your prompt and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                prompt = input(">>> ").strip()
                if not prompt:
                    continue
                run_generation(model, tokenizer, prompt, args)
            except KeyboardInterrupt:
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
