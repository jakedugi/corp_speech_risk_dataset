"""
Entry point for the quote extraction pipeline.
Run with CLI options for output file and visualization mode.
"""

import argparse
from pathlib import Path
from loguru import logger
from corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline import (
    QuoteExtractionPipeline,
)
from corp_speech_risk_dataset.orchestrators import quote_extraction_config as config


def main():
    parser = argparse.ArgumentParser(description="Run the quote extraction pipeline.")
    parser.add_argument(
        "--output",
        type=str,
        default="extracted_quotes.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization mode (stage outputs).",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        default=None,
        help="Source root for .txt files (overrides config.DB_DIR)",
    )
    parser.add_argument(
        "--mirror-out",
        type=Path,
        default=None,
        help="Destination root for mirrored output (overrides config.MIRROR_OUT_DIR)",
    )
    parser.add_argument(
        "--viz-out",
        type=Path,
        default=None,
        help="Directory to write stage JSONL files (overrides default)",
    )
    args = parser.parse_args()

    logger.add("pipeline_run.log", rotation="1 day")
    logger.info("Starting quote extraction process...")

    if args.mirror_out:
        config.MIRROR_OUT_DIR = args.mirror_out

    pipeline = QuoteExtractionPipeline(
        visualization_mode=args.visualize,
        output_dir=args.viz_out,
        mirror_mode=True,
        db_root=args.db_root,
    )
    results = pipeline.run()
    pipeline.save_results(results, args.output)

    logger.info("Quote extraction process finished successfully.")


if __name__ == "__main__":
    main()
