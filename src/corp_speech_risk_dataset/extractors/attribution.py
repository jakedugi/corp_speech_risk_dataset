"""
The Attributor, responsible for identifying the speaker of a quote and
filtering candidates based on company aliases.
"""
from typing import Iterable, List
from loguru import logger
import textacy.extract
import csv
from pathlib import Path
import re

from ..utils.nlp import get_nlp
from ..models.quote_candidate import QuoteCandidate
from ..orchestrators.quote_extraction_config import ROLE_KEYWORDS, SP500_CSV


class Attributor:
    """
    Uses NLP (spaCy, textacy) to perform quote attribution and filters out
    quotes that are not attributed to a known company alias.
    """
    def __init__(self, company_aliases: List[str]):
        """Initializes the attributor with company aliases."""
        self.nlp = get_nlp()
        self.aliases = set(a.lower() for a in company_aliases)

    def _doc_aliases(self, text: str) -> set[str]:
        """
        Scan for any official_name or ticker in the S&P CSV;
        if found, pull that row's official_name + ticker + all exec/board names.
        """
        aliases = set()
        low = text.lower()
        with SP500_CSV.open(encoding="utf8", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name, tkr = row["official_name"].lower(), row["ticker"].lower()
                if name in low or tkr in low:
                    # always include ticker & official name
                    aliases.update({name, tkr})
                    # pull every exec*/board* column
                    for col, val in row.items():
                        if (col.startswith("exec") or col.startswith("board")) and val.strip():
                            aliases.add(val.strip().lower())
        return aliases

    def filter(self, candidates: List[QuoteCandidate]) -> Iterable[QuoteCandidate]:
        """
        Filters a list of candidates, keeping only those attributed to
        a known alias.

        Args:
            candidates: A list of QuoteCandidate objects from the first pass.

        Yields:
            Enriched QuoteCandidate objects with the speaker identified.
        """
        for qc in candidates:
            # 0.5) sanitize unbalanced straight-quotes in context
            ctx = qc.context
            if ctx.count('"') % 2 == 1:
                # append a closing quote to balance
                ctx = ctx + '"'
            # rebuild doc on sanitized text
            doc = self.nlp(ctx)

            # reuse sanitized low-ctx and quote text
            dyn             = self._doc_aliases(ctx)
            aliases         = self.aliases | dyn | set(ROLE_KEYWORDS)
            speaker_assigned = None
            low_ctx   = ctx.lower()
            low_quote = qc.quote.lower() if qc.quote else ""

            # 1) sentence-level Textacy: only the sentence containing the quote
            try:
                for sent in doc.sents:
                    if low_quote in sent.text.lower():
                        mini = self.nlp(sent.text)
                        # wrap Textacy call in try/except
                        try:
                            for speaker_span, _, content in textacy.extract.triples.direct_quotations(mini):
                                # resolve speaker text
                                if isinstance(speaker_span, list):
                                    spk = " ".join(s.text for s in speaker_span)
                                else:
                                    spk = speaker_span.text or ""
                                if spk:
                                    speaker_assigned = spk
                                    break
                        except ValueError:
                            # fallback: simple regex within this sentence
                            m = re.search(r'"([^"]+)"', sent.text)
                            if m:
                                speaker_assigned = "Unknown"
                        break
            except Exception:
                # if sent iteration fails, skip to next step
                pass

            # 2) full-context Textacy: any speaker_span in the whole window
            if not speaker_assigned:
                try:
                    for speaker_span, _, content in textacy.extract.triples.direct_quotations(doc):
                        if isinstance(speaker_span, list):
                            spk = " ".join(s.text for s in speaker_span)
                        else:
                            spk = speaker_span.text or ""
                        if spk:
                            speaker_assigned = spk
                            break
                except ValueError:
                    # still unbalanced somewhere? regex fallback across full context
                    m = re.search(r'"([^"]+)"', ctx)
                    if m:
                        speaker_assigned = "Unknown"

            # 3) alias fallback: company or exec name in quote or context
            if not speaker_assigned:
                for alias in aliases:
                    if alias in low_quote or alias in low_ctx:
                        speaker_assigned = alias.title()
                        break

            # 4) pure-role fallback: any role keyword in quote or context
            if not speaker_assigned:
                for role in ROLE_KEYWORDS:
                    if role in low_quote or role in low_ctx:
                        speaker_assigned = role.title()
                        break

            # 5) default to "Unknown"
            if not speaker_assigned:
                speaker_assigned = "Unknown"

            # assign & yield once, preserving qc.quote
            qc.speaker = speaker_assigned
            yield qc 