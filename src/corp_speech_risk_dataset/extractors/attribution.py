from typing import Iterable, List
from ..utils.nlp import get_nlp
from .models import QuoteCandidate
import textacy.extract

class Attributor:
    def __init__(self, company_aliases: List[str]):
        self.nlp = get_nlp()
        self.aliases = set(a.lower() for a in company_aliases)

    def filter(self, candidates: List[QuoteCandidate]) -> Iterable[QuoteCandidate]:
        for qc in candidates:
            text = qc.context
            doc = self.nlp(text)
            for speaker_span, cue_span, content in textacy.extract.triples.direct_quotations(doc):
                # ── DEBUG ──
                print(f"[DEBUG textacy] spans: speaker={speaker_span}, cue={cue_span}, content={content!r}")
                # 1) raw speaker
                resolved = speaker_span.text if speaker_span else ""
                # 2) coreference resolution
                clusters = doc._.coref_chains
                for chain in clusters:
                    if speaker_span is not None and any(t.i in [t.i for t in speaker_span] for t in chain):
                        resolved = chain.main.text
                        break
                low_sp = resolved.lower()
                low_qt = content.lower()
                if any(alias in low_sp for alias in self.aliases) \
                   or any(alias in low_qt for alias in self.aliases):
                    yield QuoteCandidate(
                        quote=content.strip(),
                        context=qc.context,
                        urls=qc.urls,
                        speaker=resolved
                    ) 