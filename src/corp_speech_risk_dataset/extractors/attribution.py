"""
The Attributor, responsible for identifying the speaker of a quote and
filtering candidates based on company aliases.
"""
from typing import Iterable, List
from loguru import logger
import textacy.extract

from ..utils.nlp import get_nlp
from ..models.quote_candidate import QuoteCandidate


class Attributor:
    """
    Uses NLP (spaCy, textacy) to perform quote attribution and filters out
    quotes that are not attributed to a known company alias.
    """
    def __init__(self, company_aliases: List[str]):
        """Initializes the attributor with company aliases."""
        self.nlp = get_nlp()
        self.aliases = set(a.lower() for a in company_aliases)

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
            doc = self.nlp(qc.context)
            try:
                for speaker_span, cue_span, content in textacy.extract.triples.direct_quotations(doc):
                    logger.debug(f"textacy spans: speaker={speaker_span}, cue={cue_span}, content={content!r}")
                    
                    # 1) raw speaker
                    if isinstance(speaker_span, list):
                        resolved_speaker = " ".join(span.text for span in speaker_span)
                    else:
                        resolved_speaker = speaker_span.text if speaker_span else ""
                    
                    # 2) coreference resolution using resolve() method for robustness
                    if speaker_span and doc._.has('coref_chains'):
                        spans_to_resolve = speaker_span if isinstance(speaker_span, list) else [speaker_span]
                        for span_to_resolve in spans_to_resolve:
                            resolved_spans = doc._.coref_chains.resolve(span_to_resolve)
                            if resolved_spans:
                                # Use the text from the most representative span in the cluster
                                resolved_speaker = resolved_spans[0].text
                                break # Found a resolution, no need to check other spans

                    low_speaker = resolved_speaker.lower()
                    low_content = content.text.lower()
    
                    if any(alias in low_speaker for alias in self.aliases) \
                       or any(alias in low_content for alias in self.aliases):
                        qc.quote = content.text.strip()
                        qc.speaker = resolved_speaker
                        yield qc
                        break # Move to the next candidate once a valid quote is found in the context
            except ValueError as e:
                # textacy can fail on unbalanced quotes, which is common.
                logger.warning(f"textacy failed to extract quotes from context: {qc.context!r}. Error: {e}")
                continue 