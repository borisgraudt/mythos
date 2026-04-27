"""
deduplicate.py — Near-deduplication for Mythos corpus using MinHash LSH.

Near-deduplication removes documents that are suspiciously similar (>80% Jaccard
similarity on word n-grams). This is critical for LLM training:
  - Prevents memorisation of repeated boilerplate
  - Improves generalisation (model sees diverse text)
  - Reduces dataset size by ~15–30% without quality loss

Algorithm: MinHash + Locality Sensitive Hashing (LSH)
  1. Represent each document as a set of word shingles (n-grams)
  2. Estimate Jaccard similarity via MinHash signatures
  3. Use LSH to find candidate pairs without O(N²) comparison
  4. Keep one document per near-duplicate cluster
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def _shingles(text: str, n: int = 5) -> Set[str]:
    """Extract word-level n-gram shingles from text."""
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def _minhash_signature(shingles: Set[str], num_perm: int = 128) -> List[int]:
    """
    Compute MinHash signature: num_perm minimum hash values.
    Uses the standard (a*x + b) mod p hash family.
    """
    import hashlib

    if not shingles:
        return [0] * num_perm

    sig = []
    for i in range(num_perm):
        min_hash = float("inf")
        for s in shingles:
            # Deterministic per-permutation hash
            h = int(hashlib.md5(f"{i}:{s}".encode()).hexdigest(), 16)
            if h < min_hash:
                min_hash = h
        sig.append(min_hash)
    return sig


def _lsh_buckets(signature: List[int], bands: int, rows: int) -> List[str]:
    """
    Locality Sensitive Hashing: split signature into bands.
    Two documents land in the same bucket iff their band is identical —
    a fast proxy for high Jaccard similarity.
    """
    buckets = []
    for b in range(bands):
        band = tuple(signature[b * rows : (b + 1) * rows])
        buckets.append(f"{b}:{hash(band)}")
    return buckets


def deduplicate_shards(
    shard_paths: List[Path],
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> List[Path]:
    """
    Remove near-duplicate documents across shards.

    Args:
        shard_paths: list of .txt shard files (one document per line)
        threshold:   Jaccard similarity threshold; docs above this are duplicates
        num_perm:    number of MinHash permutations (higher = more accurate)
        shingle_size: word n-gram size for shingling

    Returns:
        List of paths to deduplicated output shards (written alongside originals
        as shard_*.dedup.txt)
    """
    # LSH parameters: bands × rows = num_perm, threshold ≈ (1/bands)^(1/rows)
    # For threshold=0.8, num_perm=128: bands=32, rows=4 → threshold≈0.80
    bands = 32
    rows = num_perm // bands

    # ── Pass 1: build LSH index ──────────────────────────────────────────
    logger.info("Deduplication pass 1: building MinHash index over %d shards", len(shard_paths))

    # bucket → list of (shard_idx, doc_idx)
    bucket_to_docs: Dict[str, List] = {}
    all_docs: List[List[str]] = []  # all_docs[shard_idx] = list of lines

    for shard_idx, shard_path in enumerate(shard_paths):
        lines = shard_path.read_text(encoding="utf-8").splitlines()
        all_docs.append(lines)

        for doc_idx, line in enumerate(lines):
            if not line.strip():
                continue
            shingles = _shingles(line, shingle_size)
            sig = _minhash_signature(shingles, num_perm)
            for bucket in _lsh_buckets(sig, bands, rows):
                bucket_to_docs.setdefault(bucket, []).append((shard_idx, doc_idx))

    # ── Pass 2: find duplicate clusters ──────────────────────────────────
    logger.info("Deduplication pass 2: finding near-duplicate clusters")

    removed: Set = set()  # (shard_idx, doc_idx) pairs to remove
    seen_pairs: Set = set()

    for bucket, docs in bucket_to_docs.items():
        if len(docs) < 2:
            continue
        # Check all pairs in the same bucket
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, b = docs[i], docs[j]
                pair = (min(a, b), max(a, b))
                if pair in seen_pairs or b in removed:
                    continue
                seen_pairs.add(pair)

                # Exact Jaccard check for confirmation
                si = _shingles(all_docs[a[0]][a[1]], shingle_size)
                sj = _shingles(all_docs[b[0]][b[1]], shingle_size)
                if si and sj:
                    jaccard = len(si & sj) / len(si | sj)
                    if jaccard >= threshold:
                        removed.add(b)  # keep a, remove b

    logger.info("Removed %d near-duplicate documents", len(removed))

    # ── Pass 3: write deduplicated shards ────────────────────────────────
    out_paths = []
    for shard_idx, (shard_path, lines) in enumerate(zip(shard_paths, all_docs)):
        kept = [
            line for doc_idx, line in enumerate(lines)
            if (shard_idx, doc_idx) not in removed
        ]
        out_path = shard_path.with_suffix(".dedup.txt")
        out_path.write_text("\n".join(kept), encoding="utf-8")
        out_paths.append(out_path)
        logger.debug("Shard %s: %d → %d docs", shard_path.name, len(lines), len(kept))

    total_before = sum(len(d) for d in all_docs)
    total_after = total_before - len(removed)
    logger.info(
        "Deduplication complete: %d → %d docs (%.1f%% removed)",
        total_before, total_after, 100 * len(removed) / max(total_before, 1),
    )

    return out_paths
