"""
MythosTokenizer — ByteLevel BPE tokenizer wrapper.

Built on HuggingFace `tokenizers` (Rust backend, very fast).
Trains from scratch on your corpus or loads a saved tokenizer.

Vocab: 32k–50k tokens
Algorithm: ByteLevelBPE
  - Byte-level means every possible byte is representable → zero UNK tokens
  - Works for any language and code without special pre-tokenisation
  - Same approach used by GPT-2, RoBERTa, and many modern LLMs

Special tokens:
  <unk>  — unknown (should never appear with byte-level BPE, kept for compat)
  <bos>  — beginning of sequence  (id=1)
  <eos>  — end of sequence        (id=2)
  <pad>  — padding                (id=3)
  <sep>  — separator for multi-turn / instruction data  (id=4)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special tokens — order matters (their IDs are assigned in this sequence)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>", "<pad>", "<sep>"]
UNK_ID, BOS_ID, EOS_ID, PAD_ID, SEP_ID = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# MythosTokenizer
# ---------------------------------------------------------------------------

class MythosTokenizer:
    """
    Thin wrapper around `tokenizers.Tokenizer` that adds:
      - opinionated defaults for LLM training
      - train_from_iterator / train_from_files helpers
      - save / load in standard HuggingFace format
      - encode / decode / batch_encode convenience API
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        text_iterator: Iterator[str],
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "MythosTokenizer":
        """
        Train a new ByteLevel BPE tokenizer from a text iterator.

        text_iterator  : yields strings (one document / line per call)
        vocab_size     : target vocab size including special tokens
        min_frequency  : minimum pair frequency to merge
        """
        logger.info("Starting BPE training  vocab_size=%d  min_freq=%d", vocab_size, min_frequency)

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))

        # Byte-level pre-tokenisation — splits on whitespace then maps to bytes
        tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tokenizer.decoder       = ByteLevelDecoder()

        # NFKC normalisation (same as cleaner — redundant but cheap safety net)
        tokenizer.normalizer = NFKC()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=show_progress,
            initial_alphabet=ByteLevelPre.alphabet(),
        )

        tokenizer.train_from_iterator(text_iterator, trainer=trainer)

        # Post-processor: automatically prepend <bos> and append <eos>
        # when encoding single sequences.
        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")
        tokenizer.post_processor = TemplateProcessing(
            single=f"<bos>:0 $A:0 <eos>:0",
            pair=f"<bos>:0 $A:0 <sep>:0 $B:0 <eos>:0",
            special_tokens=[
                ("<bos>", bos),
                ("<eos>", eos),
                ("<sep>", tokenizer.token_to_id("<sep>")),
            ],
        )

        logger.info("Training complete. Effective vocab size: %d", tokenizer.get_vocab_size())
        return cls(tokenizer)

    @classmethod
    def train_from_files(
        cls,
        files: List[Union[str, Path]],
        vocab_size: int = 32_000,
        min_frequency: int = 2,
    ) -> "MythosTokenizer":
        """Train directly from a list of text file paths (memory-efficient)."""
        files = [str(p) for p in files]
        logger.info("Training from %d files", len(files))

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tokenizer.decoder       = ByteLevelDecoder()
        tokenizer.normalizer    = NFKC()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=ByteLevelPre.alphabet(),
        )

        tokenizer.train(files, trainer=trainer)

        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")
        tokenizer.post_processor = TemplateProcessing(
            single="<bos>:0 $A:0 <eos>:0",
            pair="<bos>:0 $A:0 <sep>:0 $B:0 <eos>:0",
            special_tokens=[
                ("<bos>", bos),
                ("<eos>", eos),
                ("<sep>", tokenizer.token_to_id("<sep>")),
            ],
        )

        return cls(tokenizer)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save tokenizer in modern HuggingFace format.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # ✅ основной файл (в нём ВСЁ)
        self._tok.save(str(directory / "tokenizer.json"))

        # ✅ опционально: сохранить vocab/merges через встроенный метод
        try:
            self._tok.model.save(str(directory))
        except Exception:
            pass

        logger.info("Tokenizer saved to %s", directory)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "MythosTokenizer":
        path = Path(directory) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {directory}")
        tok = Tokenizer.from_file(str(path))
        logger.info("Tokenizer loaded from %s  vocab_size=%d", directory, tok.get_vocab_size())
        return cls(tok)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        enc = self._tok.encode(text)
        if not add_special_tokens:
            # strip <bos> and <eos> added by post-processor
            ids = enc.ids
            if ids and ids[0] == BOS_ID:
                ids = ids[1:]
            if ids and ids[-1] == EOS_ID:
                ids = ids[:-1]
            return ids
        return enc.ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        return self._tok.id_to_token(id)

    def __repr__(self) -> str:
        return f"MythosTokenizer(vocab_size={self.vocab_size})"
