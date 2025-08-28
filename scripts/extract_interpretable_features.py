#!/usr/bin/env python3
"""
Extract all interpretable features from raw data and save enhanced dataset.

This script applies the comprehensive InterpretableFeatureExtractor to raw data
and saves the feature-enhanced dataset for model training.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    records = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise

    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def save_jsonl_data(records: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"Saved {len(records)} records to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise


def extract_features_batch(
    records: List[Dict[str, Any]],
    extractor: InterpretableFeatureExtractor,
    text_field: str = "text",
    context_field: str = "context",
) -> List[Dict[str, Any]]:
    """Extract features for a batch of records."""
    enhanced_records = []

    for i, record in enumerate(records):
        if i % 1000 == 0:
            logger.info(f"Processing record {i}/{len(records)}")

        # Get text and context
        text = record.get(text_field, "")
        context = record.get(context_field, "")

        if not text:
            logger.warning(f"Empty text field for record {i}")
            # Still process to maintain record alignment

        # Extract features
        try:
            features = extractor.extract_features(text, context)

            # Create enhanced record
            enhanced_record = record.copy()

            # Add all features with 'feat_' prefix to avoid conflicts
            for feature_name, feature_value in features.items():
                enhanced_record[f"feat_{feature_name}"] = feature_value

            enhanced_records.append(enhanced_record)

        except Exception as e:
            logger.error(f"Error extracting features for record {i}: {e}")
            # Add record without features to maintain alignment
            enhanced_record = record.copy()
            enhanced_records.append(enhanced_record)

    return enhanced_records


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract interpretable features from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name containing text (default: text)",
    )
    parser.add_argument(
        "--context-field",
        default="context",
        help="Field name containing context (default: context)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)",
    )
    parser.add_argument(
        "--include-lexicons",
        action="store_true",
        default=True,
        help="Include lexicon features (default: True)",
    )
    parser.add_argument(
        "--include-sequence",
        action="store_true",
        default=True,
        help="Include sequence features (default: True)",
    )
    parser.add_argument(
        "--include-linguistic",
        action="store_true",
        default=True,
        help="Include linguistic features (default: True)",
    )
    parser.add_argument(
        "--include-structural",
        action="store_true",
        default=True,
        help="Include structural features (default: True)",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    logger.info("Starting interpretable feature extraction")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Text field: {args.text_field}")
    logger.info(f"Context field: {args.context_field}")
    logger.info(f"Batch size: {args.batch_size}")

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize feature extractor with all features enabled
    extractor = InterpretableFeatureExtractor(
        include_lexicons=args.include_lexicons,
        include_sequence=args.include_sequence,
        include_linguistic=args.include_linguistic,
        include_structural=args.include_structural,
    )

    logger.info("Feature extractor initialized with all feature types enabled")

    # Load data
    logger.info("Loading data...")
    records = load_jsonl_data(args.input)

    if not records:
        logger.error("No records loaded!")
        return 1

    # Process in batches to manage memory
    all_enhanced_records = []

    for i in range(0, len(records), args.batch_size):
        batch_end = min(i + args.batch_size, len(records))
        batch = records[i:batch_end]

        logger.info(
            f"Processing batch {i//args.batch_size + 1}: records {i+1} to {batch_end}"
        )

        # Extract features for batch
        enhanced_batch = extract_features_batch(
            batch, extractor, args.text_field, args.context_field
        )

        all_enhanced_records.extend(enhanced_batch)

    # Report feature extraction statistics
    if all_enhanced_records:
        sample_record = all_enhanced_records[0]
        feature_names = [k for k in sample_record.keys() if k.startswith("feat_")]
        logger.info(f"Extracted {len(feature_names)} features per record")
        logger.info(f"Feature categories:")

        # Count features by category
        categories = {}
        for fname in feature_names:
            category = fname.split("_")[1] if len(fname.split("_")) > 1 else "unknown"
            categories[category] = categories.get(category, 0) + 1

        for category, count in sorted(categories.items()):
            logger.info(f"  {category}: {count} features")

    # Save enhanced data
    logger.info("Saving enhanced dataset...")
    save_jsonl_data(all_enhanced_records, args.output)

    logger.success(f"Feature extraction complete!")
    logger.info(f"Enhanced dataset saved to: {args.output}")
    logger.info(f"Total records processed: {len(all_enhanced_records)}")

    return 0


if __name__ == "__main__":
    exit(main())
