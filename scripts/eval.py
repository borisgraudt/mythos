#!/usr/bin/env python3
"""
eval.py — Mythos evaluation: perplexity on held-out data.

Usage:
  # Evaluate on validation split
  python scripts/eval.py \\
      --checkpoint checkpoints/150m_debug_v2/final.pt \\
      --data data/debug \\
      --tokenizer data/debug/tokenizer/tokenizer.json

  # Evaluate on specific text file
  python scripts/eval.py \\
      --checkpoint checkpoints/base_500m/final.pt \\
      --text path/to/test.txt
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mythos.core.transformer import ModelConfig, Mythos
from mythos.training.checkpoint import load_checkpoint
from mythos.utils.device import get_device
from mythos.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_perplexity(
    model: Mythos,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    """
    Compute perplexity on a dataloader.

    Perplexity = exp(mean cross-entropy loss over all tokens).
    Lower is better. A random model with vocab_size=32000 has PPL ~32000.

    Returns:
        {"perplexity": float, "loss": float, "tokens": int, "batches": int}
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        else:
            input_ids = batch["input_ids"].to(device)

        logits, _ = model(input_ids)

        # Next-token prediction: shift by 1
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()

        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
        total_tokens += targets.numel()
        num_batches += 1

        if max_batches > 0 and num_batches >= max_batches:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))  # cap at exp(20) to avoid overflow

    return {
        "perplexity": ppl,
        "loss": avg_loss,
        "tokens": total_tokens,
        "batches": num_batches,
    }


def load_text_as_loader(text_path: Path, tokenizer_path: Path, seq_len: int, batch_size: int) -> DataLoader:
    """Tokenize a text file and return a DataLoader."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = text_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids

    # Chunk into fixed-length sequences
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    n_chunks = len(ids_tensor) // (seq_len + 1)
    ids_tensor = ids_tensor[: n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    dataset = TensorDataset(ids_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_data_split_loader(data_dir: Path, split: str, seq_len: int, batch_size: int) -> DataLoader:
    """Load a pre-encoded data split."""
    from mythos.data.dataset import ShardDataset

    split_file = data_dir / "splits" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        shard_paths = [Path(p.strip()) for p in f if p.strip()]

    dataset = ShardDataset(shard_paths, seq_len=seq_len)

    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        return {"input_ids": input_ids}

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def parse_args():
    parser = argparse.ArgumentParser(description="Mythos evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="Data directory with splits/")
    parser.add_argument("--text", type=Path, default=None, help="Raw text file to evaluate on")
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json path")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0, help="Limit batches (0=all)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    device = get_device()
    logger.info("Device: %s", device)

    # Load model (auto-detects config from checkpoint)
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_config = raw_ckpt.get("config", {})
    if not saved_config:
        raise ValueError("Checkpoint has no embedded config. Pass --model_config.")

    config = ModelConfig.from_dict(saved_config)
    model = Mythos(config)
    model, _, step = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    logger.info("Model: %dM params | step %d", model.get_num_params() // 1_000_000, step)

    # Build dataloader
    if args.text is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer required when using --text")
        loader = load_text_as_loader(args.text, args.tokenizer, args.seq_len, args.batch_size)
        source = f"text file ({args.text.name})"
    elif args.data is not None:
        loader = load_data_split_loader(args.data, args.split, args.seq_len, args.batch_size)
        source = f"{args.split} split ({args.data})"
    else:
        raise ValueError("Provide --data or --text")

    logger.info("Evaluating on: %s", source)

    results = compute_perplexity(model, loader, device, max_batches=args.max_batches)

    print("\n" + "─" * 50)
    print(f"  Perplexity : {results['perplexity']:.2f}")
    print(f"  Loss       : {results['loss']:.4f}")
    print(f"  Tokens     : {results['tokens']:,}")
    print(f"  Batches    : {results['batches']}")
    print("─" * 50)

    return results


if __name__ == "__main__":
    main()
