#!/usr/bin/env python3
"""
Batch add interpretable features to all JSONL files in a directory.

This script processes thousands of individual JSONL files and adds interpretable
features to each record, optimized for memory efficiency and parallel processing.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


def process_single_file(args_tuple):
    """Process a single JSONL file and add features to all records."""
    input_file, output_file, text_field, context_field, extractor_config = args_tuple

    try:
        # Initialize extractor in this process
        extractor = InterpretableFeatureExtractor(**extractor_config)

        # Load records from file
        records = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping invalid JSON on line {line_num} in {input_file}: {e}"
                    )
                    continue

        if not records:
            logger.warning(f"No valid records found in {input_file}")
            return input_file, 0, f"No valid records"

        # Process each record
        enhanced_records = []
        for i, record in enumerate(records):
            # Get text and context
            text = record.get(text_field, "")
            context = record.get(context_field, "")

            if not text:
                logger.warning(f"Empty text field for record {i} in {input_file}")

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
                logger.error(
                    f"Error extracting features for record {i} in {input_file}: {e}"
                )
                # Add record without features to maintain alignment
                enhanced_record = record.copy()
                enhanced_records.append(enhanced_record)

        # Write enhanced records to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for record in enhanced_records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        return input_file, len(enhanced_records), "Success"

    except Exception as e:
        error_msg = f"Error processing {input_file}: {e}"
        logger.error(error_msg)
        return input_file, 0, error_msg


def get_file_pairs(
    input_dir: str, output_dir: str, pattern: str = "*.jsonl"
) -> List[tuple]:
    """Get list of (input_file, output_file) pairs."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    file_pairs = []
    for input_file in input_path.glob(pattern):
        if input_file.is_file():
            # Preserve the same filename in output directory
            output_file = output_path / input_file.name
            file_pairs.append((str(input_file), str(output_file)))

    return file_pairs


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Batch add interpretable features to all JSONL files in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir", required=True, help="Input directory containing JSONL files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for enhanced files"
    )
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
        "--pattern", default="*.jsonl", help="File pattern to match (default: *.jsonl)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be processed without actually processing",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    logger.info("Starting batch interpretable feature extraction")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Text field: {args.text_field}")
    logger.info(f"Context field: {args.context_field}")
    logger.info(f"File pattern: {args.pattern}")

    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1

    # Get file pairs
    logger.info("Scanning for files to process...")
    file_pairs = get_file_pairs(args.input_dir, args.output_dir, args.pattern)

    if not file_pairs:
        logger.error(
            f"No files found matching pattern '{args.pattern}' in {args.input_dir}"
        )
        return 1

    logger.info(f"Found {len(file_pairs)} files to process")

    if args.dry_run:
        logger.info("DRY RUN - showing first 10 files that would be processed:")
        for i, (input_file, output_file) in enumerate(file_pairs[:10]):
            logger.info(
                f"  {i+1}. {os.path.basename(input_file)} -> {os.path.basename(output_file)}"
            )
        if len(file_pairs) > 10:
            logger.info(f"  ... and {len(file_pairs) - 10} more files")
        return 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare extractor configuration
    extractor_config = {
        "include_lexicons": args.include_lexicons,
        "include_sequence": args.include_sequence,
        "include_linguistic": args.include_linguistic,
        "include_structural": args.include_structural,
    }

    logger.info("Feature extractor configuration:")
    for key, value in extractor_config.items():
        logger.info(f"  {key}: {value}")

    # Determine number of workers
    if args.workers is None:
        workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    else:
        workers = args.workers

    logger.info(f"Using {workers} parallel workers")

    # Prepare arguments for workers
    worker_args = []
    for input_file, output_file in file_pairs:
        worker_args.append(
            (
                input_file,
                output_file,
                args.text_field,
                args.context_field,
                extractor_config,
            )
        )

    # Process files in parallel
    start_time = time.time()
    successful = 0
    failed = 0
    total_records = 0

    logger.info(f"Starting parallel processing of {len(file_pairs)} files...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, args): args[0] for args in worker_args
        }

        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_file), 1):
            input_file = future_to_file[future]

            try:
                file_path, record_count, status = future.result()

                if "Success" in status:
                    successful += 1
                    total_records += record_count
                    if i % 100 == 0 or i == len(file_pairs):
                        logger.info(
                            f"Progress: {i}/{len(file_pairs)} files processed ({successful} successful, {failed} failed)"
                        )
                else:
                    failed += 1
                    logger.error(
                        f"Failed to process {os.path.basename(file_path)}: {status}"
                    )

            except Exception as e:
                failed += 1
                logger.error(
                    f"Exception processing {os.path.basename(input_file)}: {e}"
                )

    end_time = time.time()
    duration = end_time - start_time

    # Final summary
    logger.success(f"Batch processing complete!")
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Files processed: {successful} successful, {failed} failed")
    logger.info(f"Total records enhanced: {total_records:,}")
    logger.info(f"Average processing rate: {len(file_pairs)/duration:.1f} files/second")
    logger.info(f"Enhanced files saved to: {args.output_dir}")

    if failed > 0:
        logger.warning(f"⚠️ {failed} files failed to process - check logs above")
        return 1

    # Report on features added
    if successful > 0:
        # Check a sample output file to count features
        sample_files = [f for f in Path(args.output_dir).glob("*.jsonl")]
        if sample_files:
            try:
                with open(sample_files[0], "r") as f:
                    sample_record = json.loads(f.readline())
                feature_count = len(
                    [k for k in sample_record.keys() if k.startswith("feat_")]
                )
                logger.info(f"Added {feature_count} interpretable features per record")

                # Count features by category
                feature_names = [
                    k for k in sample_record.keys() if k.startswith("feat_")
                ]
                categories = {}
                for fname in feature_names:
                    category = (
                        fname.split("_")[1] if len(fname.split("_")) > 1 else "unknown"
                    )
                    categories[category] = categories.get(category, 0) + 1

                logger.info("Feature categories added:")
                for category, count in sorted(categories.items()):
                    logger.info(f"  {category}: {count} features")

            except Exception as e:
                logger.warning(f"Could not analyze sample output file: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
