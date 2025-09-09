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
from ..types.schemas.models import QuoteCandidate
from ..extractors.cleaner import TextCleaner
from enum import Enum
from ..shared.stage_writer import StageWriter
from pathlib import Path


class Stage(Enum):
    RAW = 0
    CLEAN = 1
    EXTRACT = 2
    ATTRIB = 3
    RERANK = 4


class QuoteExtractionPipeline:
    """
    Orchestrates the quote extraction workflow.
    Use `run()` to yield results and `save_results()` to write them to disk.
    """

    def __init__(
        self,
        visualization_mode: bool = False,
        output_dir: Optional[str] = None,
        mirror_mode: bool = True,
        db_root: Path | None = None,
    ):
        logger.info("Initializing Quote Extraction Pipeline...")
        self.visualization_mode = visualization_mode
        self.mirror_mode = mirror_mode
        self.loader = DocumentLoader(db_root or config.DB_DIR)
        self.cleaner = TextCleaner()
        self.first_pass = FirstPassExtractor(config.KEYWORDS, self.cleaner)
        self.attributor = Attributor(config.COMPANY_ALIASES)
        self.reranker = SemanticReranker(config.SEED_QUOTES, config.THRESHOLD)
        self.output_dir = output_dir or (config.ROOT / "data")
        if self.mirror_mode:
            self.stage_writer = StageWriter(
                self.loader.source_root, config.MIRROR_OUT_DIR
            )
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
            logger.info(
                f"Visualization mode is ON. Output will be saved to {self.DATA_DIR}"
            )
            for path in self.file_paths.values():
                path.open("w").close()
        logger.info("Pipeline initialized.")

    @classmethod
    def from_config(
        cls, visualization_mode: bool = False, output_dir: Optional[str] = None
    ):
        return cls(visualization_mode=visualization_mode, output_dir=output_dir)

    @staticmethod
    def merge_adjacent(candidates):
        merged = []
        prev = None
        for qc in candidates:
            if (
                prev
                and qc.context == prev.context
                and qc.quote.split()
                and qc.quote.startswith(prev.quote.split()[-1])
            ):
                prev.quote += " " + qc.quote
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
        for doc in docs:
            logger.debug(f"Processing document: {doc.doc_id}")
            raw_text = doc.text
            rec = {
                "doc_id": doc.doc_id,
                "stage": Stage.RAW.value,
                "text": raw_text,
                "context": None,
                "speaker": None,
                "score": None,
                "urls": [],
                "_src": str(doc.path),
            }
            if self.mirror_mode:
                self.stage_writer.write(doc.path, Stage.RAW.value, rec)
            if self.visualization_mode:
                with self.file_paths[0].open("a", encoding="utf8") as f0:
                    f0.write(json.dumps(rec) + "\n")
            # Stage 1: CLEAN
            cleaned_text = self.cleaner.clean(raw_text)
            rec_clean = {**rec, "stage": Stage.CLEAN.value, "text": cleaned_text}
            if self.mirror_mode:
                self.stage_writer.write(doc.path, Stage.CLEAN.value, rec_clean)
            if self.visualization_mode:
                with self.file_paths[1].open("a", encoding="utf8") as f1:
                    f1.write(json.dumps(rec_clean) + "\n")
            # Stage 2: FIRST PASS EXTRACTION
            raw_cands = list(self.first_pass.extract(cleaned_text))
            # drop any empty‐string quotes (avoids IndexError in merge_adjacent)
            candidates = [qc for qc in raw_cands if qc.quote and qc.quote.strip()]
            candidates = self.merge_adjacent(candidates)
            if self.mirror_mode and candidates:
                for qc in candidates:
                    self.stage_writer.write(
                        doc.path,
                        Stage.EXTRACT.value,
                        {
                            "doc_id": doc.doc_id,
                            "stage": Stage.EXTRACT.value,
                            **qc.to_dict(),
                            "_src": str(doc.path),
                        },
                    )
            if self.visualization_mode and candidates:
                with self.file_paths[2].open("a", encoding="utf8") as f2:
                    for qc in candidates:
                        rec2 = {
                            "doc_id": doc.doc_id,
                            "stage": Stage.EXTRACT.value,
                            **qc.to_dict(),
                            "_src": str(doc.path),
                        }
                        f2.write(json.dumps(rec2) + "\n")
            if not candidates:
                continue
            logger.debug(f"Found {len(candidates)} candidates in {doc.doc_id}")
            # Stage 3: ATTRIBUTION
            vetted = list(self.attributor.filter(candidates))
            if self.mirror_mode and vetted:
                for qc in vetted:
                    self.stage_writer.write(
                        doc.path,
                        Stage.ATTRIB.value,
                        {
                            "doc_id": doc.doc_id,
                            "stage": Stage.ATTRIB.value,
                            **qc.to_dict(),
                            "_src": str(doc.path),
                        },
                    )
            if self.visualization_mode and vetted:
                with self.file_paths[3].open("a", encoding="utf8") as f3:
                    for qc in vetted:
                        rec3 = {
                            "doc_id": doc.doc_id,
                            "stage": Stage.ATTRIB.value,
                            **qc.to_dict(),
                            "_src": str(doc.path),
                        }
                        f3.write(json.dumps(rec3) + "\n")
            if not vetted:
                continue
            logger.debug(f"Attributed {len(vetted)} quotes in {doc.doc_id}")
            # Stage 4: RERANKING
            final_quotes = list(self.reranker.rerank(vetted))

            # If this was an RSS record and we got *no* reranked quotes,
            # fall back to “whole‐doc” as a single QuoteCandidate:
            # if not final_quotes and hasattr(doc, "_rss_parts"):
            #     title, summary, content = doc._rss_parts
            #     parts = []
            #     for s in (title, summary, content):
            #         if s and s not in parts:
            #             parts.append(s)
            #     whole = " ".join(parts).strip()
            #     if whole:
            #         from ..types.schemas.models import QuoteCandidate
            #         # build a dummy QuoteCandidate
            #         qc = QuoteCandidate(
            #             doc_id=doc.doc_id,
            #             quote=whole,
            #             speaker=doc.speaker,
            #             score=0.0,
            #             urls=doc.urls,
            #             context=whole,
            #         )
            #         final_quotes = [qc]

            if self.mirror_mode and final_quotes:
                for qc in final_quotes:
                    self.stage_writer.write(
                        doc.path,
                        Stage.RERANK.value,
                        {
                            "doc_id": doc.doc_id,
                            "stage": Stage.RERANK.value,
                            **qc.to_dict(),
                            "_src": str(doc.path),
                        },
                    )
            if self.visualization_mode and final_quotes:
                with self.file_paths[4].open("a", encoding="utf8") as f4:
                    for qc in final_quotes:
                        rec4 = {
                            "doc_id": doc.doc_id,
                            "stage": Stage.RERANK.value,
                            **qc.to_dict(),
                            "_src": str(doc.path),
                        }
                        f4.write(json.dumps(rec4) + "\n")
            if final_quotes:
                logger.debug(
                    f"Yielding {len(final_quotes)} final quotes for {doc.doc_id}"
                )
            # (this yield now covers either the “real” quotes or our one‐item fallback)
            if final_quotes:
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
                rec = {"doc_id": doc_id, "quotes": [q.to_dict() for q in quotes]}
                out.write(json.dumps(rec) + "\n")
        logger.info(f"Saved {count} documents with quotes.")
