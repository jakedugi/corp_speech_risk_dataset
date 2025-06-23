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
from ..models.quote_candidate import QuoteCandidate

# optional spaCy sentence splitter
try:
    import spacy
    from spacy.lang.en import English
    _spacy_nlp = English()
    _spacy_nlp.add_pipe('sentencizer')
except ImportError:
    _spacy_nlp = None

class FirstPassExtractor:
    # ───────────── configurable constants ─────────
    MAX_SENT_WINDOW   = 10           # window = ± this many sentences (increased for test)
    MIN_QUOTE_CHARS   = 20          # ignore ultra-short matches ("hi" etc.)
    MAX_QUOTE_CHARS   = 600         # ignore page-long extracts
    MAX_DUP_HAMMING   = 3           # SimHash distance threshold for near-dupes
    BULLET            = re.compile(r'^\s*(?:[\*\-–‣•]|[0-9]+[.)])\s+')
    BLOCK_QUOTE       = re.compile(r'^\s*[>│▏▕▍▌▊]+')
    # Match curly or straight double quotes only
    QUOTE = re.compile(r'''
        (?:
        “([^”]+)”         # curly double
        | "([^"]+)"         # straight double
        )
    ''', re.S | re.X)
    ANC = re.compile(
        r'\b(?:said|testif(?:y|ied)|deposed|swor(?:e|n)|submitted|'
        r'annonce(?:d|ment)|blog(?:ged)?|posted|wrote|quoted|'
        r'according to|stated?|notes?|privacy policy|exhibit(?:s)?|'
        r'factual background|public statements?)\b', re.I
    )
    URL               = re.compile(r'https?://\S+')

    # stop-phrase blacklist (boiler-plate)
    STOP_PHRASES = [
        # Uncomment or add phrases as needed:
        # r'this email (and any attachments)? is (confidential|private)',
        # r'click here to (unsubscribe|opt out)',
        # r'please (do not )?rely solely on this email',
        # r'for internal use only',
        # r'no(?:t)? intended as legal advice',
    ]
    STOP_RE = re.compile("|".join(STOP_PHRASES), re.I) if STOP_PHRASES else re.compile(r'^\b$')  # dummy fallback

    def __init__(self, keywords: List[str]):
            self.keywords: Set[str] = {k.lower() for k in keywords}
            self._seen_hashes: deque[int] = deque(maxlen=8000)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

    @staticmethod
    def _simhash(text: str, bits: int = 64) -> int:
        """
        SimHash implementation for approximate duplicate detection.
        """
        v = [0] * bits
        for token in re.findall(r'\w+', text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(bits):
                v[i] += 1 if h >> i & 1 else -1
        return sum(1 << i for i, x in enumerate(v) if x > 0)

    def _is_near_duplicate(self, text: str) -> bool:
        h = self._simhash(text)
        for prev in self._seen_hashes:
            if bin(h ^ prev).count("1") <= self.MAX_DUP_HAMMING:
                return True
        self._seen_hashes.append(h)
        return False

    def extract(self, doc_text: str) -> Iterator[QuoteCandidate]:
        # sentence splitting via spaCy or NLTK
        if _spacy_nlp:
            doc = _spacy_nlp(doc_text)
            sentences = [s.text for s in doc.sents]
        else:
            sentences = nltk.sent_tokenize(doc_text)

        for i, sent in enumerate(sentences):
            window = " ".join(
                sentences[max(0, i - self.MAX_SENT_WINDOW):
                          i + self.MAX_SENT_WINDOW + 1]
            )

            # quick heuristic filter
            if not (self.QUOTE.search(sent) or self.ANC.search(sent)
                    or self.BULLET.match(sent) or self.BLOCK_QUOTE.match(sent)):
                continue
            if self.keywords and not any(k in window.lower() for k in self.keywords):
                continue

            # print(f"Window: {window}")
            #print(f"Matches: {[m.group(0) for m in self.QUOTE.finditer(window)]}")

            # extract and clean quote candidates
            for m in self.QUOTE.finditer(window):
                raw_quote = m.group(0).strip()
                if not (self.MIN_QUOTE_CHARS <= len(raw_quote) <= self.MAX_QUOTE_CHARS):
                    continue
                if self._is_near_duplicate(raw_quote):
                    continue
                # stop-phrase blacklist
                if self.STOP_RE.search(raw_quote):
                    logger.debug(f"Dropped by stop-phrase filter: {raw_quote}")
                    continue

                context = window
                urls = self.URL.findall(context)
                # lazy URL stripping from quote text
                quote_text = self.URL.sub("", raw_quote).strip()

                yield QuoteCandidate(quote=quote_text,
                                     context=context,
                                     urls=urls) 