#!/usr/bin/env python3
"""
Balanced Case-wise Data Splitting Script

This script:
1. Loads JSONL data and groups by case IDs extracted from '_src' field
2. Separates outlier cases (>5 billion) for special handling
3. Filters excluded speakers
4. Balances train/val/test splits by case size distribution
5. Dumps results to separate files for training and analysis

Key features:
- Case-wise splitting (no data leakage between cases)
- Balanced bucket distribution by case entry counts
- Outlier case handling for extreme monetary amounts
- Speaker filtering with configurable exclusions
- Comprehensive statistics and distribution reporting

Usage:
    python scripts/balanced_case_split.py \
        --input "data/final_destination/courtlistener_v6_fused_raw_coral_pred/doc_*_text_stage15.jsonl" \
        --output-dir data/balanced_splits \
        --outlier-threshold 5000000000 \
        --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA" \
        --train-ratio 0.7 \
        --val-ratio 0.15 \
        --test-ratio 0.15
"""

import argparse
import json
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from loguru import logger
import random


# Pattern to extract case IDs from file paths
CASE_ID_RE = re.compile(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/")


def extract_case_id_from_src(src: str) -> Optional[str]:
    """Extract case ID from source path using the established pattern."""
    if not src:
        return None
    match = CASE_ID_RE.search(src)
    return match.group(1) if match else None


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
                        # Extract case_id if not already present
                        if "case_id" not in data and "_src" in data:
                            case_id = extract_case_id_from_src(data["_src"])
                            if case_id:
                                data["case_id"] = case_id
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


def filter_speakers(
    data: List[Dict[str, Any]], exclude_speakers: List[str]
) -> List[Dict[str, Any]]:
    """Filter out excluded speakers."""
    filtered_data = []

    for record in data:
        speaker = record.get("speaker", "Unknown")
        if speaker not in exclude_speakers:
            filtered_data.append(record)

    logger.info(f"Filtered to {len(filtered_data)} records after speaker exclusions")
    return filtered_data


def separate_outliers(
    data: List[Dict[str, Any]], outlier_threshold: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate outlier cases based on judgment amount threshold."""
    regular_data = []
    outlier_data = []

    for record in data:
        amount = record.get("final_judgement_real")
        if (
            amount is not None
            and isinstance(amount, (int, float))
            and amount > outlier_threshold
        ):
            outlier_data.append(record)
        else:
            regular_data.append(record)

    logger.info(
        f"Separated {len(outlier_data)} outlier records (>{outlier_threshold:,.0f})"
    )
    logger.info(f"Regular data: {len(regular_data)} records")
    return regular_data, outlier_data


def group_by_case(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group data by case ID."""
    case_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in data:
        case_id = record.get("case_id")
        if case_id:
            case_groups[case_id].append(record)
        else:
            logger.warning(f"Record missing case_id: {record.get('doc_id', 'unknown')}")

    logger.info(f"Grouped data into {len(case_groups)} cases")
    return dict(case_groups)


def analyze_case_distribution(
    case_groups: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Analyze case size distribution for balancing."""
    case_sizes = [len(records) for records in case_groups.values()]

    if not case_sizes:
        return {"total_cases": 0, "total_records": 0, "size_distribution": {}}

    stats = {
        "total_cases": len(case_groups),
        "total_records": sum(case_sizes),
        "min_case_size": min(case_sizes),
        "max_case_size": max(case_sizes),
        "mean_case_size": float(np.mean(case_sizes)),
        "median_case_size": float(np.median(case_sizes)),
        "std_case_size": float(np.std(case_sizes)),
    }

    # Create size buckets for balanced distribution
    percentiles = [25, 50, 75, 90, 95]
    size_percentiles = {
        f"p{p}": float(np.percentile(case_sizes, p)) for p in percentiles
    }
    stats.update(size_percentiles)

    # Case size distribution
    size_bins = [1, 5, 10, 25, 50, 100, 250, 500, 1000, float("inf")]
    size_distribution = {}
    for i in range(len(size_bins) - 1):
        min_size = size_bins[i]
        max_size = size_bins[i + 1]
        if max_size == float("inf"):
            label = f"{min_size}+"
            count = sum(1 for size in case_sizes if size >= min_size)
        else:
            label = f"{min_size}-{max_size-1}"
            count = sum(1 for size in case_sizes if min_size <= size < max_size)
        size_distribution[label] = count

    stats["size_distribution"] = size_distribution
    return stats


def create_stratified_buckets(
    case_groups: Dict[str, List[Dict[str, Any]]], n_buckets: int = 5
) -> List[List[str]]:
    """Create stratified buckets based on case sizes for balanced splitting."""
    # Sort cases by size
    case_items = [(case_id, len(records)) for case_id, records in case_groups.items()]
    case_items.sort(key=lambda x: x[1])

    # Create buckets with roughly equal total record counts
    buckets = [[] for _ in range(n_buckets)]
    bucket_sizes = [0] * n_buckets

    # Assign cases to buckets using a greedy approach to balance total records
    for case_id, case_size in case_items:
        # Find bucket with smallest current size
        min_bucket_idx = min(range(n_buckets), key=lambda i: bucket_sizes[i])
        buckets[min_bucket_idx].append(case_id)
        bucket_sizes[min_bucket_idx] += case_size

    logger.info("Bucket distribution:")
    for i, (bucket, size) in enumerate(zip(buckets, bucket_sizes)):
        logger.info(f"  Bucket {i}: {len(bucket)} cases, {size} records")

    return buckets


def balanced_train_val_test_split(
    case_groups: Dict[str, List[Dict[str, Any]]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
]:
    """Create balanced train/val/test splits by case, maintaining size distribution."""

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create stratified buckets
    buckets = create_stratified_buckets(case_groups, n_buckets=10)

    train_cases = []
    val_cases = []
    test_cases = []

    # For each bucket, randomly assign cases to splits while maintaining ratios
    for bucket in buckets:
        # Shuffle cases in bucket
        bucket_cases = bucket.copy()
        random.shuffle(bucket_cases)

        # Calculate split points
        n_cases = len(bucket_cases)
        n_train = int(n_cases * train_ratio)
        n_val = int(n_cases * val_ratio)
        # Test gets the remainder to ensure all cases are assigned

        train_cases.extend(bucket_cases[:n_train])
        val_cases.extend(bucket_cases[n_train : n_train + n_val])
        test_cases.extend(bucket_cases[n_train + n_val :])

    # Build split dictionaries
    train_data = {case_id: case_groups[case_id] for case_id in train_cases}
    val_data = {case_id: case_groups[case_id] for case_id in val_cases}
    test_data = {case_id: case_groups[case_id] for case_id in test_cases}

    # Log split statistics
    train_records = sum(len(records) for records in train_data.values())
    val_records = sum(len(records) for records in val_data.values())
    test_records = sum(len(records) for records in test_data.values())
    total_records = train_records + val_records + test_records

    logger.info("Split statistics:")
    logger.info(
        f"  Train: {len(train_data)} cases, {train_records} records ({train_records/total_records:.1%})"
    )
    logger.info(
        f"  Val: {len(val_data)} cases, {val_records} records ({val_records/total_records:.1%})"
    )
    logger.info(
        f"  Test: {len(test_data)} cases, {test_records} records ({test_records/total_records:.1%})"
    )

    return train_data, val_data, test_data


def save_split_data(
    split_data: Dict[str, List[Dict[str, Any]]], output_path: Path, split_name: str
) -> None:
    """Save split data to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten case groups into single list
    all_records = []
    for records in split_data.values():
        all_records.extend(records)

    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    logger.info(
        f"Saved {split_name} split: {len(split_data)} cases, {len(all_records)} records to {output_path}"
    )


def save_case_metadata(
    case_groups: Dict[str, List[Dict[str, Any]]], output_path: Path, split_name: str
) -> None:
    """Save case-level metadata for analysis."""
    case_metadata = []

    for case_id, records in case_groups.items():
        # Calculate case-level statistics
        amounts = []
        for r in records:
            amount = r.get("final_judgement_real")
            if amount is not None and isinstance(amount, (int, float)):
                amounts.append(amount)

        speakers = [r.get("speaker", "Unknown") for r in records]

        metadata = {
            "case_id": case_id,
            "split": split_name,
            "num_records": len(records),
            "num_amounts": len(amounts),
            "min_amount": float(min(amounts)) if amounts else None,
            "max_amount": float(max(amounts)) if amounts else None,
            "mean_amount": float(sum(amounts) / len(amounts)) if amounts else None,
            "total_amount": float(sum(amounts)) if amounts else None,
            "unique_speakers": len(set(speakers)),
            "speaker_counts": dict(Counter(speakers)),
        }
        case_metadata.append(metadata)

    # Save as JSONL
    metadata_path = output_path.parent / f"{split_name}_case_metadata.jsonl"
    with open(metadata_path, "w") as f:
        for metadata in case_metadata:
            f.write(json.dumps(metadata) + "\n")

    logger.info(
        f"Saved {split_name} case metadata: {len(case_metadata)} cases to {metadata_path}"
    )


def save_statistics_report(
    stats: Dict[str, Any], outlier_stats: Dict[str, Any], output_dir: Path
) -> None:
    """Save comprehensive statistics report."""
    report = {
        "regular_data_stats": stats,
        "outlier_data_stats": outlier_stats,
        "total_cases": stats["total_cases"] + outlier_stats["total_cases"],
        "total_records": stats["total_records"] + outlier_stats["total_records"],
    }

    report_path = output_dir / "split_statistics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved statistics report to {report_path}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Balanced case-wise data splitting")
    parser.add_argument("--input", required=True, help="Input JSONL pattern")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=5000000000,
        help="Threshold for outlier cases (default: 5 billion)",
    )
    parser.add_argument(
        "--exclude-speakers",
        default="",
        help="Comma-separated list of speakers to exclude",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse excluded speakers
    exclude_speakers = []
    if args.exclude_speakers:
        exclude_speakers = [s.strip() for s in args.exclude_speakers.split(",")]

    logger.info("Configuration:")
    logger.info(f"  Input pattern: {args.input}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Outlier threshold: ${args.outlier_threshold:,.0f}")
    logger.info(f"  Excluded speakers: {exclude_speakers}")
    logger.info(
        f"  Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}"
    )
    logger.info(f"  Random seed: {args.random_seed}")

    # Load data
    logger.info("Loading data...")
    data = load_jsonl_files(args.input)
    if not data:
        logger.error("No data loaded!")
        return

    # Filter speakers
    if exclude_speakers:
        logger.info("Filtering speakers...")
        data = filter_speakers(data, exclude_speakers)
        if not data:
            logger.error("No data after speaker filtering!")
            return

    # Separate outliers
    logger.info("Separating outlier cases...")
    regular_data, outlier_data = separate_outliers(data, args.outlier_threshold)

    # Process regular data
    logger.info("Processing regular data...")
    regular_case_groups = group_by_case(regular_data)
    regular_stats = analyze_case_distribution(regular_case_groups)

    # Process outlier data
    logger.info("Processing outlier data...")
    outlier_case_groups = group_by_case(outlier_data)
    outlier_stats = analyze_case_distribution(outlier_case_groups)

    # Create balanced splits for regular data
    if regular_case_groups:
        logger.info("Creating balanced train/val/test splits...")
        train_data, val_data, test_data = balanced_train_val_test_split(
            regular_case_groups,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.random_seed,
        )

        # Save regular data splits
        save_split_data(train_data, output_dir / "train.jsonl", "train")
        save_split_data(val_data, output_dir / "val.jsonl", "val")
        save_split_data(test_data, output_dir / "test.jsonl", "test")

        # Save case metadata for each split
        save_case_metadata(train_data, output_dir / "train.jsonl", "train")
        save_case_metadata(val_data, output_dir / "val.jsonl", "val")
        save_case_metadata(test_data, output_dir / "test.jsonl", "test")

    # Save outlier data separately
    if outlier_case_groups:
        logger.info("Saving outlier data...")
        save_split_data(outlier_case_groups, output_dir / "outliers.jsonl", "outliers")
        save_case_metadata(
            outlier_case_groups, output_dir / "outliers.jsonl", "outliers"
        )

    # Save comprehensive statistics
    save_statistics_report(regular_stats, outlier_stats, output_dir)

    # Print summary
    logger.success("Data splitting complete!")
    logger.info("Summary:")
    logger.info(
        f"  Regular cases: {regular_stats['total_cases']} cases, {regular_stats['total_records']} records"
    )
    logger.info(
        f"  Outlier cases: {outlier_stats['total_cases']} cases, {outlier_stats['total_records']} records"
    )
    logger.info(f"  Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
