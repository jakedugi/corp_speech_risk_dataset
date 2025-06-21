import re
import nltk
import logging
from typing import List, Iterator
from .models import QuoteCandidate
import textacy.extract

# —————— debug logger ——————
logging.basicConfig(level=logging.INFO)
# ——————————————————————————

class FirstPassExtractor:
    BULLET = re.compile(r'^(?:\s{4,}|\d+\.)|^[\u2022\*-]\s')
    QUOTE  = re.compile(r'[“"\'‘].+?[”"\'’]', re.S)
    ANC    = re.compile(r'\b(?:said|testif(?:y|ied)|posted|according to|quoted)\b', re.I)
    URL    = re.compile(r'https?://\S+')

    def __init__(self, keywords: List[str]):
        self.keywords = set(k.lower() for k in keywords)
        try:
            nltk.data.find("tokenizers/punkt")
        except nltk.downloader.DownloadError:
            nltk.download("punkt")

    def extract(self, doc_text: str) -> Iterator[QuoteCandidate]:
        sents = nltk.sent_tokenize(doc_text)
        for i, s in enumerate(sents):
            window = " ".join(sents[max(0,i-2):i+3])
            if (self.QUOTE.search(s) or self.ANC.search(s) or self.BULLET.match(s)) \
               and any(k in window.lower() for k in self.keywords):
                print(f"[PRINT FirstPassExtractor] candidate sentence = {s!r}")
                urls = self.URL.findall(window)
                yield QuoteCandidate(quote=s.strip(), context=window, urls=urls) 