"""Single source single law orchestrator for the corporate speech risk dataset.

This module coordinates the execution of data collection and processing for a specific
legal source and statute combination.
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

def run_law_pipeline(
    statute: str,
    output_dir: Optional[Path] = None,
    save_opinions: bool = True
) -> None:
    """Run the pipeline for a single statute.
    
    Args:
        statute: The statute to process
        output_dir: Optional custom output directory
        save_opinions: Whether to fetch and save full opinion texts
    """
    # Setup logging
    setup_logging()
    logger.info(f"Starting pipeline for statute: {statute}")
    
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
        
        # Run CourtListener collection for single statute
        logger.info(f"Collecting data for {statute}")
        courtlistener.process_statutes(
            statutes=[statute],
            pages=config.default_pages,
            page_size=config.default_page_size,
            date_filed_min=config.default_date_min,
            save_opinions=save_opinions,
            output_root=raw_dir,
            chunk_size=10,
        )
        
        # Process collected data
        logger.info("Extracting quotes and labeling laws")
        quote_extractor.process_directory(raw_dir, processed_dir)
        law_labeler.process_directory(processed_dir)
        
        logger.info(f"Pipeline completed for {statute}")
        
    except Exception as e:
        logger.exception(f"Pipeline failed for {statute}")
        raise

if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def main(
        statute: str = typer.Argument(..., help="Statute to process"),
        output_dir: Optional[Path] = typer.Option(None, help="Custom output directory"),
        save_opinions: bool = typer.Option(True, help="Save full opinion texts")
    ):
        run_law_pipeline(statute, output_dir, save_opinions)
    
    app() 