"""
The main quote extraction pipeline, orchestrating extraction, attribution, and reranking.
"""
import json
from loguru import logger
from typing import Iterator, Optional
from . import quote_extraction_config as config
from ..extractors.loader import DocumentLoader
from ..extractors.first_pass import FirstPassExtractor
from ..extractors.attribution import Attributor
from ..extractors.rerank import SemanticReranker
from ..models.quote_candidate import QuoteCandidate
from ..extractors.cleaner import TextCleaner

class QuoteExtractionPipeline:
    """
    Orchestrates the quote extraction workflow.
    Use `run()` to yield results and `save_results()` to write them to disk.
    """
    def __init__(self, visualization_mode: bool = False, output_dir: Optional[str] = None):
        logger.info("Initializing Quote Extraction Pipeline...")
        self.visualization_mode = visualization_mode
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.first_pass = FirstPassExtractor(config.KEYWORDS, self.cleaner)
        self.attributor = Attributor(config.COMPANY_ALIASES)
        self.reranker = SemanticReranker(config.SEED_QUOTES, config.THRESHOLD)
        self.output_dir = output_dir or (config.ROOT / "data")
        if self.visualization_mode:
            self.DATA_DIR = self.output_dir
            self.DATA_DIR.mkdir(exist_ok=True, parents=True)
            self.file_paths = {
                0: self.DATA_DIR / "stage0_raw.jsonl",
                1: self.DATA_DIR / "stage1_cleaner.jsonl",
                2: self.DATA_DIR / "stage2_extractor.jsonl",
                3: self.DATA_DIR / "stage3_attributor.jsonl",
                4: self.DATA_DIR / "stage4_reranker.jsonl",
            }
            logger.info(f"Visualization mode is ON. Output will be saved to {self.DATA_DIR}")
            for path in self.file_paths.values():
                path.open("w").close()
        logger.info("Pipeline initialized.")

    @classmethod
    def from_config(cls, visualization_mode: bool = False, output_dir: Optional[str] = None):
        return cls(visualization_mode=visualization_mode, output_dir=output_dir)

    @staticmethod
    def merge_adjacent(candidates):
        merged = []
        prev = None
        for qc in candidates:
            if prev and qc.context == prev.context and \
               qc.quote.split() and qc.quote.startswith(prev.quote.split()[-1]):
                prev.quote += ' ' + qc.quote
            else:
                if prev:
                    merged.append(prev)
                prev = qc
        if prev:
            merged.append(prev)
        return merged

    def run(self) -> Iterator[tuple[str, list[QuoteCandidate]]]:
        """
        Runs the full pipeline and yields (doc_id, list_of_quotes) for each document.
        """
        logger.debug("Starting pipeline run...")
        docs = list(self.loader)
        if self.visualization_mode:
            with self.file_paths[0].open("a", encoding="utf8") as f0:
                for doc in docs:
                    rec = {
                        "doc_id": doc.doc_id,
                        "stage": 0,
                        "text": doc.text,
                        "context": None,
                        "speaker": None,
                        "score": None,
                        "urls": []
                    }
                    f0.write(json.dumps(rec) + "\n")
        for doc in docs:
            logger.debug(f"Processing document: {doc.doc_id}")
            raw_text = doc.text
            # Stage 1: CLEAN
            cleaned_text = self.cleaner.clean(raw_text)
            if self.visualization_mode:
                with self.file_paths[1].open("a", encoding="utf8") as f1:
                    rec = {
                        "doc_id": doc.doc_id,
                        "stage": 1,
                        "text": cleaned_text,
                        "context": None,
                        "speaker": None,
                        "score": None,
                        "urls": []
                    }
                    f1.write(json.dumps(rec) + "\n")
            # Stage 2: FIRST PASS EXTRACTION
            candidates = list(self.first_pass.extract(cleaned_text))
            candidates = self.merge_adjacent(candidates)
            if self.visualization_mode and candidates:
                with self.file_paths[2].open("a", encoding="utf8") as f2:
                    for qc in candidates:
                        rec = {"doc_id": doc.doc_id, "stage": 2, **qc.to_dict()}
                        f2.write(json.dumps(rec) + "\n")
            if not candidates:
                continue
            logger.debug(f"Found {len(candidates)} candidates in {doc.doc_id}")
            # Stage 3: ATTRIBUTION
            vetted = list(self.attributor.filter(candidates))
            if self.visualization_mode and vetted:
                with self.file_paths[3].open("a", encoding="utf8") as f3:
                    for qc in vetted:
                        rec = {"doc_id": doc.doc_id, "stage": 3, **qc.to_dict()}
                        f3.write(json.dumps(rec) + "\n")
            if not vetted:
                continue
            logger.debug(f"Attributed {len(vetted)} quotes in {doc.doc_id}")
            # Stage 4: RERANKING
            final_quotes = list(self.reranker.rerank(vetted))
            if self.visualization_mode and final_quotes:
                with self.file_paths[4].open("a", encoding="utf8") as f4:
                    for qc in final_quotes:
                        rec = {"doc_id": doc.doc_id, "stage": 4, **qc.to_dict()}
                        f4.write(json.dumps(rec) + "\n")
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