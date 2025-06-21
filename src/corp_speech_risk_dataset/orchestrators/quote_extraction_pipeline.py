"""
The main quote extraction pipeline, which orchestrates the three-stage
extraction process:
1. First Pass: Regex and keyword-based candidate generation.
2. Attribution: Speaker identification and filtering.
3. Reranking: Semantic scoring and thresholding.
"""
import json
from loguru import logger
from typing import Iterator

from . import quote_extraction_config as config
from ..extractors.loader import DocumentLoader
from ..extractors.first_pass import FirstPassExtractor
from ..extractors.attribution import Attributor
from ..extractors.rerank import SemanticReranker
from ..models.quote_candidate import QuoteCandidate

class QuoteExtractionPipeline:
    """Orchestrates the quote extraction workflow."""
    def __init__(self):
        """Initializes all components of the pipeline."""
        logger.info("Initializing Quote Extraction Pipeline...")
        self.loader = DocumentLoader()
        self.first_pass = FirstPassExtractor(config.KEYWORDS)
        self.attributor = Attributor(config.COMPANY_ALIASES)
        self.reranker = SemanticReranker(config.SEED_QUOTES,
                                           config.THRESHOLD)
        logger.info("Pipeline initialized.")

    def run(self) -> Iterator[tuple[str, list[QuoteCandidate]]]:
        """
        Runs the full pipeline and yields the final results.
        
        Yields:
            A tuple of (doc_id, list_of_quotes) for each document processed.
        """
        logger.debug("Starting pipeline run...")
        for doc in self.loader:
            logger.debug(f"Processing document: {doc.doc_id}")
            
            # Stage 1: Get candidates
            candidates = list(self.first_pass.extract(doc.text))
            if not candidates:
                continue
            logger.debug(f"Found {len(candidates)} candidates in {doc.doc_id}")

            # Stage 2: Attribute and filter
            vetted = list(self.attributor.filter(candidates))
            if not vetted:
                continue
            logger.debug(f"Attributed {len(vetted)} quotes in {doc.doc_id}")

            # Stage 3: Rerank and yield
            final_quotes = list(self.reranker.rerank(vetted))
            if final_quotes:
                logger.debug(f"Yielding {len(final_quotes)} final quotes for {doc.doc_id}")
                yield doc.doc_id, final_quotes
        logger.debug("Pipeline run finished.")

    def save_results(self, results, output_file="extracted_quotes.jsonl"):
        """
        Writes the extracted results to a JSONL file.

        Args:
            results: The iterator of (doc_id, quotes) tuples from the run() method.
            output_file: The path to the output JSONL file.
        """
        logger.info(f"Saving results to {output_file}...")
        count = 0
        with open(output_file, "w", encoding="utf8") as out:
            for doc_id, quotes in results:
                count += 1
                rec = {
                    "doc_id": doc_id,
                    "quotes": [q.to_dict() for q in quotes]
                }
                out.write(json.dumps(rec) + "\n")
        logger.info(f"Saved {count} documents with quotes.") 