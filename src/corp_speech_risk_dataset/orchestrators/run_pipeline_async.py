"""Async pipeline orchestration for the corporate speech risk dataset.

This module coordinates the execution of multiple data collection and processing steps
across different legal sources and statutes using async operations.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio

from loguru import logger

from corp_speech_risk_dataset.api.courtlistener import CourtListenerClient
from corp_speech_risk_dataset.config import load_config
from corp_speech_risk_dataset.extractors.quote_extractor import QuoteExtractor
from corp_speech_risk_dataset.extractors.law_labeler import LawLabeler
from corp_speech_risk_dataset.shared.logging_utils import setup_logging
from corp_speech_risk_dataset.infrastructure.file_io import ensure_dir
from corp_speech_risk_dataset.api.client.base_api_client import BaseAPIClient
from corp_speech_risk_dataset.extractors.base_extractor import BaseExtractor


class PipelineOrchestrator:
    def __init__(self, api_clients=None, extractors=None, output_path=None):
        self.api_clients = api_clients or []
        self.extractors = extractors or []
        self.output_path = output_path

    async def run(self):
        pass


def run_full_pipeline(
    statutes: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    save_opinions: bool = True,
) -> None:
    """Run the complete data collection and processing pipeline.

    Args:
        statutes: Optional list of specific statutes to process
        output_dir: Optional custom output directory
        save_opinions: Whether to fetch and save full opinion texts
    """
    # Setup logging
    setup_logging()
    logger.info("Starting full pipeline run")

    # Load configuration
    config = load_config()

    # Ensure output directories exist
    raw_dir = output_dir / "raw" if output_dir else config.output_dir
    processed_dir = output_dir / "processed" if output_dir else Path("data/processed")
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    try:
        # Initialize clients and processors
        courtlistener = CourtListenerClient(config.api_token)
        quote_extractor = QuoteExtractor()
        law_labeler = LawLabeler()

        # Run CourtListener collection
        logger.info("Starting CourtListener data collection")
        courtlistener.process_statutes(
            statutes=statutes,
            pages=config.default_pages,
            page_size=config.default_page_size,
            date_filed_min=config.default_date_min,
            save_opinions=save_opinions,
            output_root=raw_dir,
            chunk_size=10,
        )

        # Process collected data
        logger.info("Starting quote extraction and law labeling")
        quote_extractor.process_directory(raw_dir, processed_dir)
        law_labeler.process_directory(processed_dir)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    run_full_pipeline()
