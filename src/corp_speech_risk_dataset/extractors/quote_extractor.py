"""Quote extraction from legal documents."""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from corp_speech_risk_dataset.utils.file_io import load_json, save_json, list_json_files
from corp_speech_risk_dataset.models.quote import Quote  # assume a Quote dataclass

class QuoteExtractor:
    """Extracts relevant quotes from legal documents."""
    def __init__(
        self,
        json_path: Path,
        txt_path: Path,
        keywords: List[str],
        aliases: List[str],
        seed_quotes: List[str],
        threshold: float
    ):
        """Initialize the extractor with file paths and configuration."""
        self.json_path = json_path
        self.txt_path = txt_path
        self.keywords = keywords
        self.aliases = aliases
        self.seed_quotes = seed_quotes
        self.threshold = threshold

        # Load raw inputs
        raw = load_json(self.json_path)
        if isinstance(raw, dict):
            self.data = [raw]
        else:
            self.data = raw
        self.text = self.txt_path.read_text(encoding="utf8")
        # TODO: Initialize NLP models or other resources here (e.g., spaCy)

    def extract(self) -> List[Quote]:
        """Run extraction on the initialized files and return a list of Quote objects."""
        results: List[Quote] = []
        docs = self.data if isinstance(self.data, list) else [self.data]
        for quote_text in self.seed_quotes:
            for doc in docs:
                text_block = doc.get("plain_text", "")
                if quote_text in text_block or quote_text in self.text:
                    # Speaker resolution and score logic would go here
                    results.append(Quote(quote=quote_text, speaker=None, score=1.0, urls=[]))
        return results

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Deprecated: use pipeline.run() instead"""
        # Deprecated method retained for backward compatibility
        save_json([q.dict() for q in self.extract()], output_path)
        logger.info(f"Extracted {len(self.extract())} quotes from {input_path}")

    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """Deprecated: use pipeline.run() instead"""
        output_dir.mkdir(parents=True, exist_ok=True)
        for input_file in list_json_files(input_dir):
            output_file = output_dir / input_file.name
            self.process_file(input_file, output_file)