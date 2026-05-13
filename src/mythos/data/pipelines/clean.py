"""
TextCleaner — production-grade text cleaning pipeline.

Design follows C4 (Raffel et al., 2020) quality heuristics extended with:
  - HTML / markup stripping
  - Unicode normalisation (NFKC)
  - Deduplication at line level
  - Heuristic quality gates (line length, word ratio, punctuation, etc.)
  - Optional ftfy for encoding repair

All filters are configurable so you can tune precision/recall per data source.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# Optional dependency — graceful fallback if not installed
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# Optional dependency — graceful fallback if not installed
try:
    from langdetect import LangDetectException, detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CleanerConfig:
    # Encoding
    fix_encoding: bool = True          # use ftfy when available

    # Language gating
    require_language: Optional[str] = "en"   # None = accept all

    # C4-style heuristics
    min_words_per_line:  int   = 3      # drop lines with fewer words
    max_line_length:     int   = 1000   # drop absurdly long lines
    min_doc_words:       int   = 50     # drop very short documents
    max_doc_words:       int   = 100_000

    # Quality ratios
    max_symbol_word_ratio:  float = 0.1   # fraction of tokens that are symbols
    max_bullet_line_ratio:  float = 0.9   # fraction of lines starting with bullet
    min_stop_word_ratio:    float = 0.02  # must have some common words

    # Line-level deduplication within a document
    dedup_lines: bool = True

    # Specific pattern removal
    remove_html:     bool = True
    remove_urls:     bool = False   # keep URLs for code/web data
    remove_emails:   bool = True
    normalize_whitespace: bool = True


# ---------------------------------------------------------------------------
# Precompiled regexes
# ---------------------------------------------------------------------------

_RE_HTML        = re.compile(r"<[^>]+>")
_RE_URL         = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL       = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_RE_MULTI_SPACE = re.compile(r" {2,}")
_RE_MULTI_NL    = re.compile(r"\n{3,}")
_RE_BULLET      = re.compile(r"^\s*[•·▪▸◦‣\-\*]\s")

# Common English stop words used for quality filtering
_STOP_WORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
})


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """
    Stateless text cleaning pipeline.
    Each method returns None if the document should be discarded.
    """

    def __init__(self, config: Optional[CleanerConfig] = None):
        self.cfg = config or CleanerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> Optional[str]:
        """
        Full pipeline. Returns cleaned text or None if it fails quality gates.
        Designed to be used in multiprocessing pools (no shared state).
        """
        if not text or not text.strip():
            return None

        # 1. Fix encoding artifacts
        if self.cfg.fix_encoding and HAS_FTFY:
            text = ftfy.fix_text(text)

        # 2. Unicode normalisation — NFKC collapses compatibility variants
        text = unicodedata.normalize("NFKC", text)

        # 3. Remove markup / patterns
        if self.cfg.remove_html:
            text = _RE_HTML.sub(" ", text)
        if self.cfg.remove_urls:
            text = _RE_URL.sub(" ", text)
        if self.cfg.remove_emails:
            text = _RE_EMAIL.sub(" ", text)

        # 4. Whitespace normalisation
        if self.cfg.normalize_whitespace:
            text = _RE_MULTI_SPACE.sub(" ", text)
            text = _RE_MULTI_NL.sub("\n\n", text)
            text = text.strip()

        # 5. Quality filters
        text = self._filter_lines(text)
        if text is None:
            return None

        if not self._passes_doc_filters(text):
            return None

        # 6. Language gate
        if self.cfg.require_language is not None:
            if not self._is_language(text, self.cfg.require_language):
                return None

        return text

    # ------------------------------------------------------------------
    # Line-level filtering
    # ------------------------------------------------------------------

    def _filter_lines(self, text: str) -> Optional[str]:
        lines = text.splitlines()
        seen = set()
        kept = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue

            # Line deduplication
            if self.cfg.dedup_lines:
                key = stripped.lower()
                if key in seen:
                    continue
                seen.add(key)

            words = stripped.split()

            # Minimum words per line
            if len(words) < self.cfg.min_words_per_line:
                continue

            # Maximum line length (char)
            if len(stripped) > self.cfg.max_line_length:
                continue

            kept.append(line)

        if not kept:
            return None

        result = "\n".join(kept).strip()
        return result if result else None

    # ------------------------------------------------------------------
    # Document-level filters (C4 heuristics)
    # ------------------------------------------------------------------

    def _passes_doc_filters(self, text: str) -> bool:
        words = text.split()
        n_words = len(words)

        # Length gates
        if n_words < self.cfg.min_doc_words:
            return False
        if n_words > self.cfg.max_doc_words:
            return False

        # Symbol/word ratio — high ratio indicates code/binary garbage
        n_symbols = sum(1 for w in words if not w.isalpha())
        if n_symbols / max(n_words, 1) > self.cfg.max_symbol_word_ratio:
            return False

        # Bullet-dominated documents (navigation menus, structured data)
        lines = [l for l in text.splitlines() if l.strip()]
        if lines:
            n_bullets = sum(1 for l in lines if _RE_BULLET.match(l))
            if n_bullets / len(lines) > self.cfg.max_bullet_line_ratio:
                return False

        # Stop-word ratio — very low means it's likely not natural language
        words_lower = [w.lower() for w in words]
        n_stop = sum(1 for w in words_lower if w in _STOP_WORDS)
        if n_stop / max(n_words, 1) < self.cfg.min_stop_word_ratio:
            return False

        return True

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def _is_language(self, text: str, lang: str) -> bool:
        if not HAS_LANGDETECT:
            return True  # skip if library not available
        try:
            # Sample first 500 chars — sufficient for detection, faster
            return detect(text[:500]) == lang
        except LangDetectException:
            return False
