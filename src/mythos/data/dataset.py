"""
dataset.py — end-to-end data preparation for Mythos.

Stages:
  1. Download  — stream datasets from HuggingFace Hub (never loads fully into RAM)
  2. Clean     — apply TextCleaner quality filters
  3. Shard     — write cleaned text to numbered .txt shards on disk
  4. Tokenize  — train ByteLevel BPE tokenizer on shards
  5. Encode    — tokenise all shards → binary .bin files (int32 arrays)
  6. Split     — produce train / val / test index files

Output layout (all under data/):
  raw/
    wikipedia/shard_0000.txt … shard_NNNN.txt
    books/shard_0000.txt …
    code/shard_0000.txt …
  tokenizer/
    tokenizer.json  vocab.json  merges.txt
  encoded/
    shard_0000.bin …       ← int32 flat token arrays
  splits/
    train.txt  val.txt  test.txt
"""

from __future__ import annotations

import logging
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
from tqdm import tqdm

from mythos.data.pipelines.clean import TextCleaner, CleanerConfig
from mythos.data.pipelines.tokenize import MythosTokenizer

# ── Python 3.14 / dill / datasets compatibility ─────────────────────────────
# dill is not yet compatible with Python 3.14's pickle changes.
# _use_legacy_cache_dir_if_possible only migrates old cache paths — it's safe
# to skip entirely; streaming datasets don't need it at all.
try:
    from datasets import DatasetBuilder as _DatasetBuilder

    def _noop_legacy_cache(self, dataset_module):
        return None

    _DatasetBuilder._use_legacy_cache_dir_if_possible = _noop_legacy_cache
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data source definition
# ---------------------------------------------------------------------------

@dataclass
class DataSource:
    name:          str
    hf_path:       str
    hf_name:       Optional[str] = None
    split:         str = "train"
    text_column:   str = "text"
    max_docs:      Optional[int] = None
    cleaner_cfg:   CleanerConfig = field(default_factory=CleanerConfig)


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    data_dir:       Path  = Path("data")
    shard_size:     int   = 100_000
    vocab_size:     int   = 32_000
    min_tok_freq:   int   = 2
    tokenizer_docs: int   = 1_000_000
    train_ratio:    float = 0.80
    val_ratio:      float = 0.10
    num_proc:       int   = 2  # type: ignore[operator]
    seed:           int   = 42


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    def __init__(self, sources: List[DataSource], config: Optional[PipelineConfig] = None):
        self.sources = sources
        self.cfg     = config or PipelineConfig()
        self._setup_dirs()

    def _setup_dirs(self):
        for name in ["raw", "tokenizer", "encoded", "splits"]:
            (self.cfg.data_dir / name).mkdir(parents=True, exist_ok=True)
        for src in self.sources:
            (self.cfg.data_dir / "raw" / src.name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 + 2 + 3: Download → Clean → Shard
    # ------------------------------------------------------------------

    def download_and_clean(self) -> Dict[str, List[Path]]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets  to download data")

        all_shards: Dict[str, List[Path]] = {}

        for src in self.sources:
            logger.info("=== Processing source: %s ===", src.name)
            cleaner = TextCleaner(src.cleaner_cfg)
            shards  = self._stream_source(src, cleaner, load_dataset)
            all_shards[src.name] = shards
            logger.info("  %s → %d shards written", src.name, len(shards))

        return all_shards

    def _stream_source(self, src: DataSource, cleaner: TextCleaner, load_dataset) -> List[Path]:
        out_dir = self.cfg.data_dir / "raw" / src.name

        ds = load_dataset(
            src.hf_path,
            name=src.hf_name,
            split=src.split,
            streaming=True,
            trust_remote_code=True,
        )

        shard_paths: List[Path] = []
        shard_idx   = 0
        buf: List[str] = []
        n_seen = 0
        n_kept = 0
        t0 = time.time()

        pbar = tqdm(desc=f"  {src.name}", unit=" docs", dynamic_ncols=True)

        for example in ds:
            raw_text = example.get(src.text_column, "")
            if not isinstance(raw_text, str):
                continue

            cleaned = cleaner.clean(raw_text)
            n_seen += 1
            pbar.update(1)

            if cleaned:
                buf.append(cleaned)
                n_kept += 1

            if len(buf) >= self.cfg.shard_size:
                path = self._write_shard(buf, out_dir, shard_idx)
                shard_paths.append(path)
                buf = []
                shard_idx += 1

            if src.max_docs and n_seen >= src.max_docs:
                break

        if buf:
            path = self._write_shard(buf, out_dir, shard_idx)
            shard_paths.append(path)

        pbar.close()
        elapsed = time.time() - t0
        keep_rate = n_kept / max(n_seen, 1) * 100
        logger.info(
            "  done  docs_seen=%d  docs_kept=%d  keep_rate=%.1f%%  time=%.1fs",
            n_seen, n_kept, keep_rate, elapsed,
        )
        return shard_paths

    @staticmethod
    def _write_shard(docs: List[str], directory: Path, idx: int) -> Path:
        path = directory / f"shard_{idx:04d}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(docs))
        return path

    # ------------------------------------------------------------------
    # Stage 4: Train tokenizer
    # ------------------------------------------------------------------

    def train_tokenizer(self, shards: Dict[str, List[Path]]) -> MythosTokenizer:
        tok_dir = self.cfg.data_dir / "tokenizer"

        if (tok_dir / "tokenizer.json").exists():
            logger.info("Tokenizer already exists at %s — loading.", tok_dir)
            return MythosTokenizer.load(tok_dir)

        logger.info("Training tokenizer  vocab_size=%d  sample_docs=%d",
                    self.cfg.vocab_size, self.cfg.tokenizer_docs)

        tokenizer = MythosTokenizer.train(
            text_iterator=self._tokenizer_iterator(shards),
            vocab_size=self.cfg.vocab_size,
            min_frequency=self.cfg.min_tok_freq,
        )
        tokenizer.save(tok_dir)
        return tokenizer

    def _tokenizer_iterator(self, shards: Dict[str, List[Path]]) -> Iterator[str]:
        all_paths: List[Path] = []
        for paths in shards.values():
            all_paths.extend(paths)

        rng = np.random.default_rng(self.cfg.seed)
        rng.shuffle(all_paths)  # type: ignore[arg-type]

        yielded = 0
        for path in all_paths:
            if yielded >= self.cfg.tokenizer_docs:
                break
            with open(path, encoding="utf-8") as f:
                content = f.read()
            for doc in content.split("\n\n"):
                if doc.strip():
                    yield doc
                    yielded += 1
                    if yielded >= self.cfg.tokenizer_docs:
                        return

    # ------------------------------------------------------------------
    # Stage 5: Encode shards to binary
    # ------------------------------------------------------------------

    def encode_shards(self, shards: Dict[str, List[Path]], tokenizer: MythosTokenizer) -> List[Path]:
        """
        Tokenise each shard → flat int32 binary files.

        Binary format: [n_tokens: uint64][token_0: int32][token_1: int32]...
        """
        enc_dir = self.cfg.data_dir / "encoded"
        out_paths: List[Path] = []

        all_shards: List[Path] = [p for paths in shards.values() for p in paths]

        for shard_path in tqdm(all_shards, desc="Encoding shards", unit="shard"):
            out_path = enc_dir / (shard_path.stem + ".bin")
            if out_path.exists():
                out_paths.append(out_path)
                continue

            with open(shard_path, encoding="utf-8") as f:
                docs = [d for d in f.read().split("\n\n") if d.strip()]

            token_arrays = tokenizer.encode_batch(docs)
            flat = [tok for arr in token_arrays for tok in arr]

            with open(out_path, "wb") as f:
                f.write(struct.pack("<Q", len(flat)))
                f.write(np.array(flat, dtype=np.int32).tobytes())

            out_paths.append(out_path)

        total_tokens = sum(self._count_tokens(p) for p in out_paths)
        logger.info("Encoded %d shards  total_tokens=%.2fB",
                    len(out_paths), total_tokens / 1e9)
        return out_paths

    @staticmethod
    def _count_tokens(bin_path: Path) -> int:
        with open(bin_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        return n

    # ------------------------------------------------------------------
    # Stage 6: Train / Val / Test split
    # ------------------------------------------------------------------

    def split(self, encoded_paths: List[Path]) -> Dict[str, List[Path]]:
        paths = list(encoded_paths)
        rng = np.random.default_rng(self.cfg.seed)
        rng.shuffle(paths)  # type: ignore[arg-type]

        n = len(paths)
        # Guarantee at least 1 shard in train
        n_train = max(1, int(n * self.cfg.train_ratio))
        n_val   = max(0, int(n * self.cfg.val_ratio))
        # Don't exceed available shards
        n_train = min(n_train, n)
        n_val   = min(n_val,   n - n_train)

        splits = {
            "train": paths[:n_train],
            "val":   paths[n_train : n_train + n_val],
            "test":  paths[n_train + n_val :],
        }

        splits_dir = self.cfg.data_dir / "splits"
        for split_name, split_paths in splits.items():
            out = splits_dir / f"{split_name}.txt"
            with open(out, "w") as f:
                for p in split_paths:
                    f.write(str(p.resolve()) + "\n")
            logger.info("  %-6s %d shards", split_name, len(split_paths))

        return splits

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, List[Path]]:
        """Execute all stages end-to-end. Returns the train/val/test split."""
        logger.info("DataPipeline starting  sources=%d", len(self.sources))
        shards    = self.download_and_clean()
        tokenizer = self.train_tokenizer(shards)
        encoded   = self.encode_shards(shards, tokenizer)
        return self.split(encoded)


# ---------------------------------------------------------------------------
# Shard reader (used during training)
# ---------------------------------------------------------------------------

class ShardDataset:
    """
    Memory-mapped reader for encoded binary shards.
    Yields non-overlapping chunks of `seq_len` tokens.
    """

    def __init__(self, shard_paths: List[Path], seq_len: int):
        self.seq_len = seq_len
        self._load(shard_paths)

    def _load(self, paths: List[Path]):
        arrays = []
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                arr = np.frombuffer(f.read(n * 4), dtype=np.int32).copy()
            arrays.append(arr)
        self._data = np.concatenate(arrays)

    def __len__(self) -> int:
        return (len(self._data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        import torch
        start = idx * self.seq_len
        chunk = self._data[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
