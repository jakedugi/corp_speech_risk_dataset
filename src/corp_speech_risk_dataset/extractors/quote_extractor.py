"""
Main QuoteExtractor implementation that orchestrates the full extraction pipeline.
This consolidates functionality from multiple scattered extractor classes.
"""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Set
from loguru import logger

from ..domain.ports import QuoteExtractor as QuoteExtractorPort
from ..types.schemas.models import QuoteCandidate
from .first_pass import FirstPassExtractor
from .attribution import Attributor
from .rerank import SemanticReranker
from .cleaner import TextCleaner


class QuoteExtractor(QuoteExtractorPort):
    """
    Main quote extractor that orchestrates the full pipeline:
    1. First pass extraction (regex + keyword filtering)
    2. Attribution (speaker identification)
    3. Semantic reranking
    """

    def __init__(
        self,
        keywords: List[str] = None,
        company_aliases: Set[str] = None,
        seed_quotes: List[str] = None,
        threshold: float = 0.55,
    ):
        """
        Initialize the quote extractor with configuration.

        Args:
            keywords: Keywords to filter for relevant content
            company_aliases: Company/person aliases for attribution
            seed_quotes: Example quotes for semantic similarity
            threshold: Minimum similarity threshold for reranking
        """
        # Use reasonable defaults if not provided
        if keywords is None:
            keywords = ["regulation", "policy", "statement", "violation", "compliance"]
        if company_aliases is None:
            company_aliases = {"company", "corporation", "inc", "llc"}
        if seed_quotes is None:
            seed_quotes = ["The company stated that", "According to the policy"]

        self.cleaner = TextCleaner()
        self.first_pass = FirstPassExtractor(keywords, self.cleaner)
        self.attributor = Attributor(company_aliases)
        self.reranker = SemanticReranker(seed_quotes, threshold)

    def extract_quotes(self, doc_text: str) -> List[QuoteCandidate]:
        """
        Extract quotes method for test compatibility.
        Returns a list instead of iterator.
        """
        return list(self.extract(doc_text))

    def extract(self, doc_text: str) -> Iterator[QuoteCandidate]:
        """
        Extract and process quotes through the full pipeline.

        Args:
            doc_text: The full text of the document

        Yields:
            Fully processed QuoteCandidate objects
        """
        # Clean the document text
        cleaned_text = self.cleaner.clean(doc_text)

        # First pass: extract potential quotes
        candidates = list(self.first_pass.extract(cleaned_text))
        if not candidates:
            return

        logger.debug(f"First pass found {len(candidates)} candidates")

        # Attribution: identify speakers
        attributed = list(self.attributor.filter(candidates))
        if not attributed:
            return

        logger.debug(f"Attribution found {len(attributed)} attributed quotes")

        # Semantic reranking: filter by similarity to seed quotes
        final_quotes = list(self.reranker.rerank(attributed))
        logger.debug(f"Reranking found {len(final_quotes)} final quotes")

        yield from final_quotes

    def process_file(self, input_file: Path, output_file: Path) -> None:
        """
        Process a single file and save extracted quotes to output file.

        Args:
            input_file: Path to the input file to process
            output_file: Path to save the results
        """
        try:
            # For compatibility with test, assume input is JSON with text field
            import json

            if input_file.suffix == ".json":
                data = json.loads(input_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    all_quotes = []
                    for item in data:
                        text = item.get("opinion_text", "")
                        quotes = list(self.extract(text))
                        all_quotes.extend([q.to_dict() for q in quotes])
                    output_file.write_text(json.dumps(all_quotes, indent=2))
                else:
                    text = data.get("opinion_text", "")
                    quotes = list(self.extract(text))
                    output_file.write_text(
                        json.dumps([q.to_dict() for q in quotes], indent=2)
                    )
            else:
                # For text files, process directly
                text = input_file.read_text(encoding="utf-8")
                quotes = list(self.extract(text))
                import json

                output_file.write_text(
                    json.dumps([q.to_dict() for q in quotes], indent=2)
                )
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")

    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """
        Process all files in input directory and save results to output directory.

        Args:
            input_dir: Directory containing files to process
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_file in input_dir.glob("*.json"):
            if input_file.is_file():
                logger.info(f"Processing {input_file}")
                output_file = output_dir / f"processed_{input_file.name}"
                self.process_file(input_file, output_file)
