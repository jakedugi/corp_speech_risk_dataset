import asyncio
import os
from pathlib import Path

from src.api.ftc_client import FTCClient
from src.api.sec_client import SECClient
from src.api.courtlistener_client import CourtListenerClient
from src.extractors.quote_extractor import QuoteExtractor
from src.extractors.law_labeler import LawLabeler
from src.orchestrators.run_pipeline import PipelineOrchestrator
from src.utils.logging_utils import setup_logging

logger = setup_logging()

async def main():
    """Main entry point for the application."""
    try:
        # Initialize API clients
        api_clients = [
            FTCClient(api_key=os.getenv("FTC_API_KEY")),
            SECClient(api_key=os.getenv("SEC_API_KEY")),
            CourtListenerClient(api_key=os.getenv("COURTLISTENER_API_KEY"))
        ]
        
        # Initialize extractors
        extractors = [
            QuoteExtractor(),
            LawLabeler()
        ]
        
        # Set output path
        output_path = Path("data/processed")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        await PipelineOrchestrator.run_pipeline(
            api_clients=api_clients,
            extractors=extractors,
            output_path=str(output_path)
        )
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

