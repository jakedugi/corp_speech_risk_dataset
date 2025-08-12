#!/usr/bin/env python3
"""
Enhance Raw Data with Interpretable Features

This script:
1. Loads raw JSONL data files
2. Extracts interpretable features for each record
3. Appends features as new fields to the original data
4. Saves enhanced data to new files for downstream processing

The enhanced data can then be used with the balanced case split script.

Key features:
- Batch processing for memory efficiency
- Progress tracking with detailed logging
- Feature name prefixing to avoid conflicts
- Context-aware feature extraction from text fields
- Robust error handling and recovery

Usage:
    python scripts/enhance_raw_data_with_features.py \
        --input "data/final_destination/courtlistener_v6_fused_raw_coral_pred/doc_*_text_stage15.jsonl" \
        --output-dir data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_features \
        --text-field text \
        --context-field context \
        --batch-size 1000 \
        --feature-prefix interpretable
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional
import numpy as np
from tqdm import tqdm
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


def load_jsonl_batch(file_path: str, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    """
    Load JSONL file in batches for memory efficiency.

    Args:
        file_path: Path to JSONL file
        batch_size: Number of records per batch

    Yields:
        List of records for each batch
    """
    batch = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    batch.append(record)

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON on line {line_num} in {file_path}: {e}"
                    )
                    continue

            # Yield remaining records
            if batch:
                yield batch

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return


def extract_text_and_context(
    record: Dict[str, Any], text_field: str, context_field: Optional[str]
) -> tuple[str, str]:
    """
    Extract text and context from a record.

    Args:
        record: Data record
        text_field: Field name containing main text
        context_field: Field name containing context (optional)

    Returns:
        Tuple of (text, context)
    """
    text = record.get(text_field, "")
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    context = ""
    if context_field and context_field in record:
        context = record.get(context_field, "")
        if not isinstance(context, str):
            context = str(context) if context is not None else ""

    return text, context


def enhance_records_with_features(
    records: List[Dict[str, Any]],
    text_field: str,
    context_field: Optional[str],
    feature_prefix: str,
    extractor: InterpretableFeatureExtractor,
) -> List[Dict[str, Any]]:
    """
    Enhance records with interpretable features.

    Args:
        records: List of data records
        text_field: Field name containing main text
        context_field: Field name containing context (optional)
        feature_prefix: Prefix for feature field names
        extractor: Feature extractor instance

    Returns:
        List of enhanced records with features
    """
    enhanced_records = []

    for record in records:
        try:
            # Extract text and context
            text, context = extract_text_and_context(record, text_field, context_field)

            # Skip if no text
            if not text.strip():
                print(f"Warning: Skipping record with empty text field '{text_field}'")
                enhanced_records.append(record)
                continue

            # Extract features using the class-based extractor
            features_dict = extractor.extract_features(text, context)

            # Create enhanced record
            enhanced_record = record.copy()

            # Add features with prefix
            for feature_name, feature_value in features_dict.items():
                prefixed_name = f"{feature_prefix}_{feature_name}"
                enhanced_record[prefixed_name] = float(feature_value)

            # Add metadata
            enhanced_record[f"{feature_prefix}_feature_count"] = len(features_dict)
            enhanced_record[f"{feature_prefix}_text_length"] = len(text)
            enhanced_record[f"{feature_prefix}_context_length"] = len(context)

            enhanced_records.append(enhanced_record)

        except Exception as e:
            print(f"Warning: Failed to extract features for record: {e}")
            enhanced_records.append(record)
            continue

    return enhanced_records


def save_enhanced_records(records: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save enhanced records to JSONL file.

    Args:
        records: List of enhanced records
        output_path: Output file path
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
    except Exception as e:
        print(f"Error saving records to {output_path}: {e}")
        raise


def process_single_file(
    input_path: str,
    output_path: str,
    text_field: str,
    context_field: Optional[str],
    feature_prefix: str,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Process a single JSONL file and enhance with features.

    Args:
        input_path: Input file path
        output_path: Output file path
        text_field: Field name containing main text
        context_field: Field name containing context (optional)
        feature_prefix: Prefix for feature field names
        batch_size: Batch size for processing

    Returns:
        Processing statistics
    """
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")

    # Initialize feature extractor
    extractor = InterpretableFeatureExtractor()

    stats = {
        "total_records": 0,
        "processed_records": 0,
        "failed_records": 0,
        "feature_count": 0,  # Will be set after first extraction
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process file in batches
    total_batches = 0
    with open(output_path, "w", encoding="utf-8") as output_file:

        for batch_records in tqdm(
            load_jsonl_batch(input_path, batch_size), desc="Processing batches"
        ):
            total_batches += 1
            stats["total_records"] += len(batch_records)

            try:
                # Enhance batch with features
                enhanced_records = enhance_records_with_features(
                    batch_records, text_field, context_field, feature_prefix, extractor
                )

                # Set feature count from first successful extraction
                if stats["feature_count"] == 0 and enhanced_records:
                    # Count features from first enhanced record
                    feature_fields = [
                        k
                        for k in enhanced_records[0].keys()
                        if k.startswith(f"{feature_prefix}_")
                        and not k.endswith(
                            ("_feature_count", "_text_length", "_context_length")
                        )
                    ]
                    stats["feature_count"] = len(feature_fields)

                # Write enhanced records
                for record in enhanced_records:
                    json.dump(record, output_file, ensure_ascii=False)
                    output_file.write("\n")

                stats["processed_records"] += len(enhanced_records)

            except Exception as e:
                print(f"Error processing batch {total_batches}: {e}")
                stats["failed_records"] += len(batch_records)

                # Write original records if enhancement fails
                for record in batch_records:
                    json.dump(record, output_file, ensure_ascii=False)
                    output_file.write("\n")

    print(f"Completed: {input_path}")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Processed: {stats['processed_records']}")
    print(f"  Failed: {stats['failed_records']}")
    print(f"  Features added: {stats['feature_count']}")
    print()

    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enhance raw JSONL data with interpretable features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file pattern (supports glob patterns)",
    )

    parser.add_argument(
        "--output-dir", required=True, help="Output directory for enhanced files"
    )

    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name containing main text (default: text)",
    )

    parser.add_argument(
        "--context-field", help="Field name containing context (optional)"
    )

    parser.add_argument(
        "--feature-prefix",
        default="interpretable",
        help="Prefix for feature field names (default: interpretable)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for processing (default: 500)",
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"Error: No files found matching pattern: {args.input}")
        return 1

    print(f"Found {len(input_files)} files to process")
    print(f"Output directory: {args.output_dir}")
    print(f"Text field: {args.text_field}")
    print(f"Context field: {args.context_field or 'None'}")
    print(f"Feature prefix: {args.feature_prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"Features to add: [Will be determined from first extraction]")
    print()

    # Process all files
    total_stats = {
        "total_files": len(input_files),
        "processed_files": 0,
        "total_records": 0,
        "processed_records": 0,
        "failed_records": 0,
    }

    for input_file in input_files:
        try:
            # Generate output path
            input_path = Path(input_file)
            output_path = Path(args.output_dir) / input_path.name

            # Process file
            file_stats = process_single_file(
                input_file,
                str(output_path),
                args.text_field,
                args.context_field,
                args.feature_prefix,
                args.batch_size,
            )

            # Update totals
            total_stats["processed_files"] += 1
            total_stats["total_records"] += file_stats["total_records"]
            total_stats["processed_records"] += file_stats["processed_records"]
            total_stats["failed_records"] += file_stats["failed_records"]

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            continue

    # Print final summary
    print("=" * 60)
    print("FEATURE ENHANCEMENT SUMMARY")
    print("=" * 60)
    print(
        f"Total files processed: {total_stats['processed_files']}/{total_stats['total_files']}"
    )
    print(f"Total records: {total_stats['total_records']}")
    print(f"Successfully enhanced: {total_stats['processed_records']}")
    print(f"Failed enhancements: {total_stats['failed_records']}")
    print(
        f"Success rate: {100 * total_stats['processed_records'] / max(1, total_stats['total_records']):.1f}%"
    )
    print(f"Features added per record: {total_stats.get('feature_count', 'Unknown')}")
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
