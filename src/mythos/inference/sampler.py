"""Sampling strategies for text generation."""

import torch
import torch.nn.functional as F


def sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability >= top_p,
    then samples from that set.
    """
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens once cumulative prob exceeds top_p
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
    sorted_probs /= sorted_probs.sum()

    next_token = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)]
    return next_token.item()


def sample_top_k(logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> int:
    """Keep only top-k tokens, then sample."""
    if temperature != 1.0:
        logits = logits / temperature

    if top_k < logits.size(-1):
        indices_to_remove = logits < torch.topk(logits, top_k).values[..., -1, None]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def greedy(logits: torch.Tensor) -> int:
    return logits.argmax(-1).item()
