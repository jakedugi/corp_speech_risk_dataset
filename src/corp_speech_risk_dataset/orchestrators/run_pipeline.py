"""Main pipeline orchestration for the corporate speech risk dataset.

This module coordinates the execution of multiple data collection and processing steps
across different legal sources and statutes.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from corp_speech_risk_dataset.api.courtlistener import CourtListenerClient
from corp_speech_risk_dataset.config import load_config
from corp_speech_risk_dataset.extractors.quote_extractor import QuoteExtractor
from corp_speech_risk_dataset.extractors.law_labeler import LawLabeler
from corp_speech_risk_dataset.shared.logging_utils import setup_logging
from corp_speech_risk_dataset.infrastructure.file_io import ensure_dir
from corp_speech_risk_dataset.api.client.base_api_client import BaseAPIClient
from corp_speech_risk_dataset.extractors.base_extractor import BaseExtractor


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


"""
Pipeline orchestrator for coordinating API clients and extractors.
"""
import asyncio


class PipelineOrchestrator:
    """
    Orchestrates the full pipeline by coordinating API clients and extractors.
    Handles async data fetching and extraction processing.
    """

    def __init__(
        self,
        api_clients: List[BaseAPIClient],
        extractors: List[BaseExtractor],
        output_path: Optional[str] = None,
    ):
        """
        Initialize the orchestrator with clients and extractors.

        Args:
            api_clients: List of API clients for data fetching
            extractors: List of extractors for data processing
            output_path: Optional output path for results (for compatibility)
        """
        self.api_clients = api_clients
        self.extractors = extractors
        self.output_path = output_path

    async def initialize(self):
        """Initialize all API clients."""
        for client in self.api_clients:
            await client.initialize()

    async def close(self):
        """Close all API clients."""
        for client in self.api_clients:
            await client.close()

    async def run(self) -> List[Dict[str, Any]]:
        """
        Run method for compatibility with existing tests.
        Uses empty query params as default.
        """
        return await self.run_pipeline({})

    async def run_pipeline(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run the full pipeline: fetch data, extract, and validate.

        Args:
            query_params: Parameters for data fetching

        Returns:
            List of processed results
        """
        try:
            # Initialize clients
            await self.initialize()

            # Fetch data from all clients
            all_data = []
            for client in self.api_clients:
                data = await client.fetch_data(query_params)
                all_data.extend(data)

            logger.info(
                f"Fetched {len(all_data)} records from {len(self.api_clients)} clients"
            )

            # Process with extractors
            results = []
            for extractor in self.extractors:
                for data_item in all_data:
                    extracted = extractor.extract(data_item)
                    validated = extractor.validate(extracted)
                    # Handle both boolean and list returns from validate
                    if validated:
                        if isinstance(validated, bool):
                            # If validate returns True, use the extracted data
                            if isinstance(extracted, list):
                                results.extend(extracted)
                            else:
                                results.append(extracted)
                        else:
                            # If validate returns a list, use that
                            results.extend(validated)

            logger.info(f"Extracted and validated {len(results)} items")
            return results

        finally:
            # Always close clients
            await self.close()

    async def run_extraction_only(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run extraction on provided data without fetching.

        Args:
            data: Pre-fetched data to process

        Returns:
            List of extracted results
        """
        results = []
        for extractor in self.extractors:
            for data_item in data:
                extracted = extractor.extract(data_item)
                validated = extractor.validate(extracted)
                if validated:
                    results.extend(validated)

        return results


if __name__ == "__main__":
    run_full_pipeline()
