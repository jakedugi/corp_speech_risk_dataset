#!/usr/bin/env python3
"""
Prepare CORAL ordinal data from fused embeddings and outcomes.

This script:
1. Loads JSONL data with 'fused_emb' and 'final_judgement_real' fields
2. Filters by excluded speakers and outcome thresholds
3. Creates ordinal buckets (Low/Medium/High) from outcome amounts
4. Saves prepared data for CORAL training

Usage:
    python scripts/prepare_coral_data.py \
        --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" \
        --output data/coral_training_data.jsonl \
        --max-threshold 15500000000 \
        --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA"
"""

import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from loguru import logger


def load_jsonl_files(pattern: str) -> List[Dict[str, Any]]:
    """Load data from JSONL files matching the pattern."""
    files = glob.glob(pattern) if "*" in pattern else [pattern]

    all_data = []
    for file_path in files:
        logger.info(f"Loading {file_path}...")
        try:
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if "fused_emb" in data and "final_judgement_real" in data:
                            all_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"JSON decode error in {file_path}:{line_num}: {e}"
                        )
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

    logger.info(f"Loaded {len(all_data)} records from {len(files)} files")
    return all_data


def filter_data(
    data: List[Dict[str, Any]], exclude_speakers: List[str], max_threshold: float
) -> List[Dict[str, Any]]:
    """Filter data by speaker and outcome threshold."""
    filtered_data = []

    for record in data:
        # Filter by speaker
        speaker = record.get("speaker", "Unknown")
        if speaker in exclude_speakers:
            continue

        # Filter by outcome threshold
        outcome = record.get("final_judgement_real")
        if outcome is not None and outcome > max_threshold:
            continue

        # Ensure required fields exist
        if "fused_emb" not in record:
            continue

        filtered_data.append(record)

    logger.info(f"Filtered to {len(filtered_data)} records after exclusions")
    return filtered_data


def create_outcome_buckets(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create ordinal outcome buckets from continuous values."""
    # Extract valid outcome amounts
    amounts = []
    for record in data:
        amount = record.get("final_judgement_real")
        if amount is not None and amount > 0:
            amounts.append(amount)

    if not amounts:
        logger.error("No valid outcome amounts found!")
        return []

    amounts = np.array(amounts)

    # Calculate percentile thresholds
    q33 = np.percentile(amounts, 33.33)
    q67 = np.percentile(amounts, 66.67)

    logger.info(f"Outcome distribution:")
    logger.info(f"  Total records: {len(data)}")
    logger.info(f"  Valid amounts: {len(amounts)}")
    logger.info(f"  Min: ${amounts.min():,.2f}")
    logger.info(f"  33rd percentile: ${q33:,.2f}")
    logger.info(f"  67th percentile: ${q67:,.2f}")
    logger.info(f"  Max: ${amounts.max():,.2f}")

    # Create buckets
    bucketed_data = []
    bucket_counts = {"low": 0, "medium": 0, "high": 0, "missing": 0}

    for record in data:
        amount = record.get("final_judgement_real")

        if amount is None or amount <= 0:
            bucket = "missing"
        elif amount < q33:
            bucket = "low"
        elif amount < q67:
            bucket = "medium"
        else:
            bucket = "high"

        # Create new record with bucket
        new_record = {
            "doc_id": record.get("doc_id"),
            "text": record.get("text"),
            "speaker": record.get("speaker"),
            "fused_emb": record.get("fused_emb"),
            "final_judgement_real": amount,
            "bucket": bucket,
            "_src": record.get("_src"),
        }

        bucketed_data.append(new_record)
        bucket_counts[bucket] += 1

    logger.info(f"Bucket distribution:")
    for bucket, count in bucket_counts.items():
        pct = 100 * count / len(bucketed_data)
        logger.info(f"  {bucket}: {count} ({pct:.1f}%)")

    return bucketed_data


def save_data(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save prepared data to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(data)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare CORAL ordinal data")
    parser.add_argument("--input", required=True, help="Input JSONL pattern")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=15500000000,
        help="Maximum outcome threshold",
    )
    parser.add_argument(
        "--exclude-speakers",
        default="",
        help="Comma-separated list of speakers to exclude",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include records with missing outcome data",
    )

    args = parser.parse_args()

    # Parse excluded speakers
    exclude_speakers = []
    if args.exclude_speakers:
        exclude_speakers = [s.strip() for s in args.exclude_speakers.split(",")]

    logger.info(f"Excluding speakers: {exclude_speakers}")
    logger.info(f"Max threshold: ${args.max_threshold:,.2f}")

    # Load data
    data = load_jsonl_files(args.input)
    if not data:
        logger.error("No data loaded!")
        return

    # Filter data
    filtered_data = filter_data(data, exclude_speakers, args.max_threshold)
    if not filtered_data:
        logger.error("No data after filtering!")
        return

    # Create buckets
    bucketed_data = create_outcome_buckets(filtered_data)
    if not bucketed_data:
        logger.error("No data after bucketing!")
        return

    # Optionally filter out missing values
    if not args.include_missing:
        bucketed_data = [r for r in bucketed_data if r["bucket"] != "missing"]
        logger.info(f"Removed missing values, {len(bucketed_data)} records remaining")

    # Save data
    save_data(bucketed_data, args.output)

    logger.success("Data preparation complete!")


if __name__ == "__main__":
    main()
