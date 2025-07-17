"""
The Attributor, responsible for identifying the speaker of a quote and
filtering candidates based on company aliases.
"""

from typing import Iterable, List, Set
from loguru import logger
import textacy.extract
import csv
from pathlib import Path
import re
from spacy.pipeline import EntityRuler

from ..infrastructure.nlp import get_nlp
from ..models.quote_candidate import QuoteCandidate
from ..domain.ports import QuoteAttributor
from ..orchestrators.quote_extraction_config import ROLE_KEYWORDS

# Optional transformer imports (commented out)
# import onnxruntime as ort
# from transformers import DistilBertTokenizerFast


class Attributor(QuoteAttributor):
    """
    Multi-sieve quote attribution:
      1) Rule-based cue regex
      2) Dependency-pattern fallback
      3) Textacy direct-quotation triples (original logic)
      4) Alias-enhanced NER via EntityRuler
      5) Optional quantized DistilBERT (commented)
      6) (Commented) Coreference and LLM fallbacks
      7) Final alias & role fallbacks
    """

    ANC_PATTERN = re.compile(
        r"\b(?:said|stated|noted|blogged|posted|wrote|quoted|"
        r"according to|testif(?:y|ied)|deposed|swor(?:e|n)|submitted|"
        r"annonce(?:d|ment)|privacy policy|public statements?)\b",
        re.I,
    )

    def __init__(self, company_aliases: Set[str]):
        """Initializes the attributor with company + officer aliases."""
        self.nlp = get_nlp()  # spaCy with AppleOps if available
        self.aliases = company_aliases  # already lowered in config
        self._add_alias_ruler()

        # Optional: load quantized DistilBERT (commented)
        # self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        # self.ort_sess = ort.InferenceSession("distilbert_cpu_quant.onnx")

    def _add_alias_ruler(self):
        """Add custom entity ruler with our aliases (company + role keywords)."""
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = []
            for alias in self.aliases:
                patterns.append({"label": "CUSTOM_ENTITY", "pattern": alias})
            for keyword in ROLE_KEYWORDS:
                patterns.append({"label": "CUSTOM_ENTITY", "pattern": keyword})
            ruler.add_patterns(patterns)

    def filter(self, candidates: List[QuoteCandidate]) -> Iterable[QuoteCandidate]:
        """
        Applies all sieves in order. If any sieve finds a valid attribution,
        we break and yield that result. Otherwise, discard the candidate.
        """

        for qc in candidates:
            # 0) Sanitize unbalanced quotes
            ctx = qc.context
            if ctx.count('"') % 2 == 1:
                ctx += '"'
            doc = self.nlp(ctx)
            low_ctx = ctx.lower()
            low_quote = qc.quote.lower() if qc.quote else ""
            aliases = self.aliases | set(ROLE_KEYWORDS)

            # 1) Rule-based cue regex
            for sent in doc.sents:
                m = re.search(
                    rf'([A-Z][a-z]+)\s+{self.ANC_PATTERN.pattern}\s*[:,-]?\s*[""](.+?)[""]',
                    sent.text,
                )
                if m:
                    qc.speaker = m.group(1)
                    yield qc
                    break
            if getattr(qc, "speaker", None):
                continue

            # 2) Dependency-pattern fallback
            for sent in doc.sents:
                for token in sent:
                    if token.lemma_ in {
                        "say",
                        "state",
                        "note",
                        "post",
                        "blog",
                        "quote",
                        "write",
                    }:
                        subs = [
                            c
                            for c in token.children
                            if c.dep_ in {"nsubj", "nsubjpass"}
                        ]
                        if subs and subs[0].ent_type_ in {"PERSON", "ORG"}:
                            qc.speaker = subs[0].text
                            yield qc
                            break
                if getattr(qc, "speaker", None):
                    break
            if getattr(qc, "speaker", None):
                continue

            # 3) Original Textacy direct-quotation triples
            # try:
            #     for sent in doc.sents:
            #         if low_quote in sent.text.lower():
            #             mini = self.nlp(sent.text)
            #             try:
            #                 for sp, _, _ in textacy.extract.triples.direct_quotations(mini):
            #                     spk = (" ".join([t.text for t in sp]) if isinstance(sp, list) else sp.text)
            #                     if spk:
            #                         qc.speaker = spk
            #                         break
            #             except ValueError:
            #                 pass
            #             break
            # except Exception:
            #     pass
            if getattr(qc, "speaker", None):
                yield qc
                continue

            # 4) Alias-enhanced NER via EntityRuler
            entities = [
                ent
                for ent in doc.ents
                if ent.label_ in {"PERSON", "ORG", "CUSTOM_ENTITY"}
            ]
            for ent in entities:
                if ent.text.lower() in aliases:
                    qc.speaker = ent.text
                    yield qc
                    break
            if getattr(qc, "speaker", None):
                continue

            # 5) Optional quantized DistilBERT pipeline (disabled)
            # if hasattr(self, 'ort_sess'):
            #     inputs = self.tokenizer(ctx, return_tensors="np", truncation=True, max_length=512)
            #     outputs = self.ort_sess.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
            #     ...

            # 6) Final alias & role fallbacks
            for alias in aliases:
                if alias in low_ctx:
                    qc.speaker = alias.title()
                    yield qc
                    break
            if getattr(qc, "speaker", None):
                continue

            # If no attribution found, discard this candidate
            # (do not yield)
