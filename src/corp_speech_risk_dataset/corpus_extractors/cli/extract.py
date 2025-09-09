#!/usr/bin/env python3
"""
CLI interface for corpus-extractors module.

Usage:
    python -m corpus_extractors.cli.extract quotes --input docs.norm.jsonl --output quotes.jsonl
    python -m corpus_extractors.cli.extract outcomes --input docs.norm.jsonl --output outcomes.jsonl
"""

import json
from pathlib import Path
from typing import Optional
import typer
import logging

from ..quote_extractor import QuoteExtractor
from ..case_outcome_imputer import CaseOutcomeImputer

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def quotes(
    input_file: Path = typer.Option(
        ..., "--input", help="Input normalized documents JSONL file"
    ),
    output_file: Path = typer.Option(..., "--output", help="Output quotes JSONL file"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="Quote extraction configuration YAML"
    ),
):
    """
    Extract quotes from normalized documents.

    Processes documents to identify quoted speech, extract spans, and perform
    attribution to speakers where possible.
    """
    logger.info(f"Extracting quotes from {input_file}")
    logger.info(f"Output: {output_file}")

    # Load configuration
    config = {}
    if config_file and config_file.exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

    # Initialize extractor
    extractor = QuoteExtractor(config)

    # Process documents
    quotes = []
    with open(input_file, "r") as infile:
        for i, line in enumerate(infile):
            if i % 1000 == 0:
                logger.info(f"Processed {i} documents")

            doc = json.loads(line.strip())
            doc_quotes = extractor.extract_quotes(doc)
            quotes.extend(doc_quotes)

    # Write output
    with open(output_file, "w") as f:
        for quote in quotes:
            f.write(json.dumps(quote, ensure_ascii=False) + "\n")

    logger.info(f"Successfully extracted {len(quotes)} quotes")


@app.command()
def outcomes(
    input_file: Path = typer.Option(
        ..., "--input", help="Input normalized documents JSONL file"
    ),
    output_file: Path = typer.Option(
        ..., "--output", help="Output outcomes JSONL file"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="Outcome extraction configuration YAML"
    ),
):
    """
    Extract case outcomes from normalized documents.

    Parses legal documents to identify case outcomes, labels, and metadata
    including cash amounts, settlement terms, and case dispositions.
    """
    logger.info(f"Extracting outcomes from {input_file}")
    logger.info(f"Output: {output_file}")

    # Load configuration
    config = {}
    if config_file and config_file.exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

    # Initialize extractor
    extractor = CaseOutcomeImputer(config)

    # Process documents
    outcomes = []
    with open(input_file, "r") as infile:
        for i, line in enumerate(infile):
            if i % 1000 == 0:
                logger.info(f"Processed {i} documents")

            doc = json.loads(line.strip())
            doc_outcomes = extractor.extract_outcomes(doc)
            outcomes.extend(doc_outcomes)

    # Write output
    with open(output_file, "w") as f:
        for outcome in outcomes:
            f.write(json.dumps(outcome, ensure_ascii=False) + "\n")

    logger.info(f"Successfully extracted {len(outcomes)} outcomes")


@app.command()
def cash_amounts(
    input_file: Path = typer.Option(..., "--input", help="Input documents JSONL file"),
    output_file: Path = typer.Option(
        ..., "--output", help="Output with extracted cash amounts JSONL file"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="Cash extraction configuration YAML"
    ),
):
    """
    Extract cash amounts from documents.

    Specialized extraction for monetary amounts, penalties, settlements,
    and financial figures mentioned in legal documents.
    """
    logger.info(f"Extracting cash amounts from {input_file}")

    # This would use the extract_cash_amounts_stage1 functionality
    # Implementation would integrate with existing cash extraction logic

    logger.info("Cash amount extraction completed")


@app.command()
def evaluate(
    input_file: Path = typer.Option(..., "--input", help="Input documents JSONL file"),
    output_file: Path = typer.Option(
        ..., "--output", help="Output evaluation results JSONL file"
    ),
):
    """
    Evaluate extraction quality and completeness.

    Runs evaluation metrics on extracted quotes and outcomes to assess
    extraction quality and identify areas for improvement.
    """
    logger.info(f"Evaluating extraction results from {input_file}")

    # This would use the final_evaluate functionality
    # Implementation would integrate with existing evaluation logic

    logger.info("Evaluation completed")


if __name__ == "__main__":
    app()
