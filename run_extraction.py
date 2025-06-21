"""
Main execution script for the quote extraction pipeline.

This script initializes the QuoteExtractionPipeline, runs it on the configured
dataset, and saves the results to a JSONL file.
"""
from loguru import logger
from corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline import QuoteExtractionPipeline

def main():
    """
    Initializes and runs the quote extraction pipeline, saving the results.
    """
    logger.add("pipeline_run.log", rotation="500 MB")
    logger.info("Starting quote extraction process...")

    pipeline = QuoteExtractionPipeline()
    results = pipeline.run()
    pipeline.save_results(results, "extracted_quotes.jsonl")

    logger.info("Quote extraction process finished successfully.")

if __name__ == "__main__":
    main() 