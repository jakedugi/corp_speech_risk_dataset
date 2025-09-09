#!/usr/bin/env python3
"""
CLI interface for corpus-features module.

Usage:
    python -m corpus_features.cli.encode --in quotes.jsonl --out features.jsonl --config config.yaml
"""

import json
from pathlib import Path
from typing import Optional
import typer
import logging

from ..registry import registry

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def encode(
    input_file: Path = typer.Option(..., "--in", help="Input quotes JSONL file"),
    output_file: Path = typer.Option(..., "--out", help="Output features JSONL file"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="Feature configuration YAML"
    ),
    feature_version: str = typer.Option(
        "v1", "--version", help="Feature version to use"
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", help="Batch size for processing"
    ),
):
    """
    Encode quotes into features.

    Takes a JSONL file of quotes and produces a JSONL file of quote features.
    """
    logger.info(f"Encoding features from {input_file} to {output_file}")

    # Load configuration
    config = {}
    if config_file and config_file.exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

    # Get feature extractor from registry
    try:
        extractor = registry.get_extractor(feature_version)
    except ValueError as e:
        logger.error(f"Feature version error: {e}")
        return

    # Process quotes
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for i, line in enumerate(infile):
            if i % batch_size == 0:
                logger.info(f"Processed {i} quotes")

            quote_data = json.loads(line.strip())

            # Extract features based on extractor type
            if isinstance(extractor, list):
                # Function-based extractor (v1)
                features = {}
                for func in extractor:
                    func_name = func.__name__
                    try:
                        result = func(
                            quote_data.get("text", ""),
                            quote_data.get("context", ""),
                        )
                        if isinstance(result, dict):
                            features.update(result)
                        else:
                            features[func_name] = result
                    except Exception as e:
                        logger.warning(f"Error in {func_name}: {e}")
                        continue
            else:
                # Class-based extractor (v2)
                try:
                    features = extractor.extract_features(quote_data)
                except AttributeError:
                    # If no extract_features method, try calling directly
                    features = extractor(quote_data)

            # Add metadata
            features["feature_version"] = feature_version
            features["quote_id"] = quote_data.get("quote_id")

            # Write output
            outfile.write(json.dumps(features, ensure_ascii=False) + "\n")

    logger.info(f"Feature encoding complete. Output: {output_file}")


@app.command()
def validate(
    input_file: Path = typer.Option(..., "--in", help="Input features JSONL file"),
    feature_version: str = typer.Option(
        ..., "--version", help="Expected feature version"
    ),
):
    """
    Validate feature file against expected schema and version.
    """
    logger.info(f"Validating features in {input_file}")

    # Basic validation - could be expanded with schema validation
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            feature_data = json.loads(line.strip())

            # Check feature version
            if feature_data.get("feature_version") != feature_version:
                logger.error(f"Line {i}: Feature version mismatch")
                return False

            # Check required fields
            required_fields = ["quote_id", "vector"]
            for field in required_fields:
                if field not in feature_data:
                    logger.error(f"Line {i}: Missing required field '{field}'")
                    return False

    logger.info("Feature validation passed")
    return True


if __name__ == "__main__":
    app()
