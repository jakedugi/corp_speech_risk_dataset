"""
Revised first-pass extractor that keeps the net wide – so we don't miss
important quotes – but adds just-enough hygiene to reduce obvious noise.
"""

import re
import hashlib
from collections import deque
from typing import List, Iterator, Set

import nltk
from loguru import logger
from ..types.schemas.models import QuoteCandidate
from ..domain.ports import QuoteExtractor

# optional spaCy sentence splitter
try:
    import spacy
    from spacy.lang.en import English

    _spacy_nlp = English()
    _spacy_nlp.add_pipe("sentencizer")
except ImportError:
    _spacy_nlp = None


class FirstPassExtractor(QuoteExtractor):
    # ───────────── configurable constants ─────────
    MAX_SENT_WINDOW = 10  # window = ± this many sentences (increased for test)
    MIN_QUOTE_CHARS = 20  # ignore ultra-short matches ("hi" etc.)
    MAX_QUOTE_CHARS = 600  # ignore page-long extracts
    MAX_DUP_HAMMING = 3  # SimHash distance threshold for near-dupes
    BULLET = re.compile(r"^\s*(?:[\*\-–‣•]|[0-9]+[.)])\s+")
    BLOCK_QUOTE = re.compile(r"^\s*[>│▏▕▍▌▊]+")
    # DOTALL makes "..." match newlines
    QUOTE = re.compile(
        r"""
        (?s)                                # DOTALL: allow multiline
        (?:
          “(.*?)”                          # curly quotes
         |"(.*?)"                          # straight quotes
        )
    """,
        re.X,
    )
    ANC = re.compile(
        r"\b(?:said|testif(?:y|ied)|deposed|swor(?:e|n)|submitted|"
        r"annonce(?:d|ment)|blog(?:ged)?|posted|wrote|quoted|"
        r"according to|stated?|notes?|privacy policy|exhibit(?:s)?|"
        r"factual background|public statements?)\b",
        re.I,
    )
    URL = re.compile(r"https?://\S+")

    # stop-phrase blacklist (boiler-plate)
    STOP_PHRASES = [
        # Uncomment or add phrases as needed:
        # r'this email (and any attachments)? is (confidential|private)',
        # r'click here to (unsubscribe|opt out)',
        # r'please (do not )?rely solely on this email',
        # r'for internal use only',
        # r'no(?:t)? intended as legal advice',
    ]
    STOP_RE = (
        re.compile("|".join(STOP_PHRASES), re.I)
        if STOP_PHRASES
        else re.compile(r"^\b$")
    )  # dummy fallback

    def __init__(self, keywords: List[str], cleaner):
        """
        Initializes the extractor with keywords and a text cleaner.

        Args:
            keywords: List of keywords to filter for relevant content
            cleaner: Text cleaning utility instance
        """
        self.keywords = set(k.lower() for k in keywords)
        self.cleaner = cleaner
        self.seen_hashes: deque = deque(maxlen=10_000)

    def _is_near_duplicate(self, quote: str) -> bool:
        """Simple SimHash-based deduplication."""
        h = self._simhash(quote)
        for seen in self.seen_hashes:
            if self._hamming_distance(h, seen) <= self.MAX_DUP_HAMMING:
                return True
        self.seen_hashes.append(h)
        return False

    def _simhash(self, text: str, bits: int = 64) -> int:
        """Simplified SimHash implementation."""
        words = text.lower().split()
        if not words:
            return 0
        vec = [0] * bits
        for word in words:
            h = hash(word) % (2**32)
            for i in range(bits):
                if h & (1 << i):
                    vec[i] += 1
                else:
                    vec[i] -= 1
        return sum(1 << i for i, v in enumerate(vec) if v > 0)

    def _hamming_distance(self, h1: int, h2: int) -> int:
        """Count differing bits."""
        return bin(h1 ^ h2).count("1")

    def extract(self, doc_text: str) -> Iterator[QuoteCandidate]:
        # Scan per-paragraph to avoid page headers & stray lines
        for para in doc_text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            for m in self.QUOTE.finditer(para):
                quote = m.group(1) or m.group(2)
                raw_quote = quote.strip()
                # 1) length sanity check
                if not (self.MIN_QUOTE_CHARS <= len(raw_quote) <= self.MAX_QUOTE_CHARS):
                    continue
                # 2) skip if it's just numbers or starts with a number
                if re.match(r"^[\d\W]+$", raw_quote) or re.match(r"^\d", raw_quote):
                    continue
                # 3) near-duplicate, stop-phrases, etc.
                if self._is_near_duplicate(raw_quote) or self.STOP_RE.search(raw_quote):
                    continue
                # Clean context using the provided cleaner (already cleaned doc, but context may need normalization)
                context = self.cleaner.clean(" ".join(para.split()))
                urls = self.URL.findall(context)
                quote_text = self.URL.sub("", raw_quote).strip()
                yield QuoteCandidate(quote=quote_text, context=context, urls=urls)
