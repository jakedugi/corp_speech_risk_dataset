"""
The first-pass extractor, responsible for generating quote candidates using
regex and keyword heuristics.
"""
import re
import nltk
from loguru import logger
from typing import List, Iterator
from ..models.quote_candidate import QuoteCandidate

class FirstPassExtractor:
    """
    Uses regex rules to quickly identify potential quotes, creating a candidate
    set for downstream processing.
    """
    BULLET = re.compile(r'^(?:\s{4,}|\d+\.)|^[\u2022\*-]\s')
    QUOTE  = re.compile(r'[“"\'‘].+?[”"\'’]', re.S)
    ANC    = re.compile(r'\b(?:said|testif(?:y|ied)|posted|according to|quoted)\b', re.I)
    URL    = re.compile(r'https?://\S+')

    def __init__(self, keywords: List[str]):
        """Initializes the extractor with keywords for filtering."""
        self.keywords = set(k.lower() for k in keywords)
        try:
            nltk.data.find("tokenizers/punkt")
        except nltk.downloader.DownloadError:
            logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
            nltk.download("punkt")

    def extract(self, doc_text: str) -> Iterator[QuoteCandidate]:
        """
        Extracts quote candidates from a document's text.

        Args:
            doc_text: The full text of the document.

        Yields:
            QuoteCandidate objects for each potential quote found.
        """
        sents = nltk.sent_tokenize(doc_text)
        for i, s in enumerate(sents):
            window = " ".join(sents[max(0,i-2):i+3])
            # Check for quote-like patterns and keyword presence
            if (self.QUOTE.search(s) or self.ANC.search(s) or self.BULLET.match(s)) \
               and any(k in window.lower() for k in self.keywords):
                logger.debug(f"FirstPassExtractor candidate: {s!r}")
                urls = self.URL.findall(window)
                yield QuoteCandidate(quote=s.strip(), context=window, urls=urls)
