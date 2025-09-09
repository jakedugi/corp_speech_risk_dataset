import asyncio
import os
from pathlib import Path

from corp_speech_risk_dataset.api.ftc_client import FTCClient
from corp_speech_risk_dataset.api.sec_client import SECClient
from corp_speech_risk_dataset.api.courtlistener import CourtListenerClient
from corp_speech_risk_dataset.corpus_extractors import QuoteExtractor
from corp_speech_risk_dataset.extractors.law_labeler import LawLabeler
from corp_speech_risk_dataset.orchestrators.run_pipeline import PipelineOrchestrator
from corp_speech_risk_dataset.shared.logging_utils import setup_logging

logger = setup_logging()


async def main():
    """Main entry point for the application."""
    try:
        # Initialize API clients
        api_clients = [
            FTCClient(api_key=os.getenv("FTC_API_KEY")),
            SECClient(api_key=os.getenv("SEC_API_KEY")),
            CourtListenerClient(api_key=os.getenv("COURTLISTENER_API_KEY")),
        ]

        # Initialize extractors
        extractors = [QuoteExtractor(), LawLabeler()]

        # Set output path
        output_path = Path("data/processed")
        output_path.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        await PipelineOrchestrator.run_pipeline(
            api_clients=api_clients, extractors=extractors, output_path=str(output_path)
        )

    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
