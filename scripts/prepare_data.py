#!/usr/bin/env python3
"""
prepare_data.py — Mythos data preparation pipeline.

Stages (in order):
  download   → stream from HuggingFace, clean, shard to .txt files
  tokenize   → train ByteLevel BPE tokenizer (vocab_size=32000)
  encode     → tokenise all shards → binary .bin files
  split      → train / val / test index files

Usage:
  python scripts/prepare_data.py --debug           # quick test (5K docs Wikipedia)
  python scripts/prepare_data.py                   # full run (FineWeb + Code + Books)
  python scripts/prepare_data.py --stages download tokenize
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

from mythos.data.dataset import DataPipeline, DataSource, PipelineConfig
from mythos.data.pipelines.clean import CleanerConfig


def setup_logging(debug: bool = False):
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    except ImportError:
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )


logger = logging.getLogger("prepare_data")


# ── Data sources ──────────────────────────────────────────────────────────────

FULL_SOURCES = [
    # English Wikipedia — 6.4M articles
    # min_doc_words=50 to keep more articles (many Wikipedia stubs are short but valid)
    DataSource(
        name="wikipedia",
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        cleaner_cfg=CleanerConfig(
            require_language=None,
            min_doc_words=50,
            max_symbol_word_ratio=0.5,
            min_stop_word_ratio=0.01,
        ),
    ),
]

DEBUG_SOURCES = [
    DataSource(
        name="wikipedia",
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        max_docs=5_000,
        cleaner_cfg=CleanerConfig(require_language=None, min_doc_words=50),
    ),
]

# 100K Wikipedia articles — enough for coherent 150M output (~1B tokens)
MEDIUM_SOURCES = [
    DataSource(
        name="wikipedia",
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        max_docs=100_000,
        cleaner_cfg=CleanerConfig(require_language="en", min_doc_words=100),
    ),
]


# ── Stages ────────────────────────────────────────────────────────────────────

ALL_STAGES = ["download", "tokenize", "encode", "split"]


def run_pipeline(sources, cfg: PipelineConfig, stages: list[str]):
    pipeline = DataPipeline(sources=sources, config=cfg)
    state: dict = {}

    for stage in stages:
        logger.info("─── Stage: %s ───", stage)

        if stage == "download":
            state["shards"] = pipeline.download_and_clean()
            total = sum(len(v) for v in state["shards"].values())
            logger.info("Download done. Total shards: %d", total)

        elif stage == "tokenize":
            shards = state.get("shards") or _load_shards(pipeline)
            state["tokenizer"] = pipeline.train_tokenizer(shards)
            logger.info("Tokenizer ready. Vocab: %d", state["tokenizer"].vocab_size)

        elif stage == "encode":
            shards = state.get("shards") or _load_shards(pipeline)
            tokenizer = state.get("tokenizer") or _load_tokenizer(pipeline)
            state["encoded"] = pipeline.encode_shards(shards, tokenizer)

        elif stage == "split":
            encoded = state.get("encoded") or _load_encoded(pipeline)
            splits = pipeline.split(encoded)
            for name, paths in splits.items():
                logger.info("  %-6s %d shards", name, len(paths))


def _load_shards(pipeline):
    raw_dir = pipeline.cfg.data_dir / "raw"
    shards = {}
    for src_dir in sorted(raw_dir.iterdir()):
        if src_dir.is_dir():
            shards[src_dir.name] = sorted(src_dir.glob("shard_*.txt"))
    if not shards:
        raise RuntimeError(f"No shards in {raw_dir}. Run 'download' stage first.")
    return shards


def _load_tokenizer(pipeline):
    from mythos.data.pipelines.tokenize import MythosTokenizer
    tok_dir = pipeline.cfg.data_dir / "tokenizer"
    if not (tok_dir / "tokenizer.json").exists():
        raise RuntimeError(f"No tokenizer in {tok_dir}. Run 'tokenize' stage first.")
    return MythosTokenizer.load(tok_dir)


def _load_encoded(pipeline):
    enc_dir = pipeline.cfg.data_dir / "encoded"
    paths = sorted(enc_dir.glob("shard_*.bin"))
    if not paths:
        raise RuntimeError(f"No encoded shards in {enc_dir}. Run 'encode' stage first.")
    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mythos data pipeline")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES)
    parser.add_argument("--debug",  action="store_true", help="5K Wikipedia docs (sanity check)")
    parser.add_argument("--medium", action="store_true", help="100K Wikipedia docs (~1B tokens, good for 150M)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    if args.debug:
        sources = DEBUG_SOURCES
        out_dir = args.out or ROOT / "data" / "debug"
        shard_size, tokenizer_docs = 500, 5_000
    elif args.medium:
        sources = MEDIUM_SOURCES
        out_dir = args.out or ROOT / "data" / "medium"
        shard_size, tokenizer_docs = 10_000, 50_000
    else:
        sources = FULL_SOURCES
        out_dir = args.out or ROOT / "data"
        shard_size, tokenizer_docs = 100_000, 1_000_000

    cfg = PipelineConfig(
        data_dir=out_dir,
        vocab_size=args.vocab_size,
        shard_size=shard_size,
        tokenizer_docs=tokenizer_docs,
    )

    mode = "debug" if args.debug else ("medium" if args.medium else "full")
    logger.info("Mythos Data Pipeline | mode=%s | stages=%s | vocab=%d | out=%s",
                mode, args.stages, args.vocab_size, out_dir)

    run_pipeline(sources, cfg, args.stages)

    logger.info("Done. Data ready at: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
