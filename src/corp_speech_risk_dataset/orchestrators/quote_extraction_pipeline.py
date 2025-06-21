from . import quote_extraction_config as config
from ..extractors.loader import DocumentLoader
from ..extractors.first_pass import FirstPassExtractor
from ..extractors.attribution import Attributor
from ..extractors.rerank import SemanticReranker

class QuoteExtractionPipeline:
    def __init__(self):
        self.loader     = DocumentLoader()
        self.first_pass = FirstPassExtractor(config.KEYWORDS)
        self.attributor = Attributor(config.COMPANY_ALIASES)
        self.reranker   = SemanticReranker(config.SEED_QUOTES, config.THRESHOLD)

    def run(self):
        for doc in self.loader:
            print(f"[PRINT Pipeline] doc_id={doc.doc_id!r}, raw_text={doc.text!r}")
            candidates = list(self.first_pass.extract(doc.text))
            vetted     = list(self.attributor.filter(candidates))
            final      = list(self.reranker.rerank(vetted))
            yield doc.doc_id, final 