#!/usr/bin/env python3
"""
Copy ALL missing features from binary dataset to authoritative dataset.

This script copies all 558 missing fields (feat_new* and interpretable_*) from the
binary dataset to the authoritative dataset using case ID mapping.

Case ID Mapping:
- Binary: anon_1_cv_00476_ded
- Authoritative: 0:12-cv-62086_flsd

We'll use text content + doc_id as the primary matching key since case IDs are anonymized.

Usage:
    python scripts/copy_all_missing_features.py \
        --binary-dir data/final_stratified_kfold_splits_binary_quote_balanced \
        --tertile-dir data/final_stratified_kfold_splits_authoritative \
        --output-dir data/final_stratified_kfold_splits_authoritative_complete \
        --workers 4
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Set, Tuple, List
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import hashlib

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    import json

    HAS_ORJSON = False
    print("Warning: orjson not available, falling back to standard json (slower)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def json_loads(data):
    """Fast JSON loading with orjson fallback."""
    if HAS_ORJSON:
        return orjson.loads(data)
    else:
        return json.loads(data)


def json_dumps(obj):
    """Fast JSON dumping with orjson fallback."""
    if HAS_ORJSON:
        return orjson.dumps(obj).decode("utf-8")
    else:
        return json.dumps(obj)


def create_text_hash(text: str) -> str:
    """Create a hash of the text for fast matching."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def create_matching_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    """Create a composite key for matching records across datasets."""
    text = record.get("text", "")
    doc_id = record.get("doc_id", "")
    text_hash = create_text_hash(text)

    # Use text hash + doc_id + text length as composite key
    text_len = str(len(text))
    return (text_hash, doc_id, text_len)


def get_missing_features(
    binary_record: Dict[str, Any], tertile_record: Dict[str, Any]
) -> Dict[str, Any]:
    """Get all features present in binary but missing in tertile record."""
    missing_features = {}

    for key, value in binary_record.items():
        # Skip metadata fields
        if key in [
            "case_id",
            "doc_id",
            "text",
            "context",
            "outcome_bin",
            "sample_weight",
        ]:
            continue

        # Copy missing features
        if key not in tertile_record:
            missing_features[key] = value

    return missing_features


def build_binary_lookup_table(
    binary_dir: Path, sample_size: int = None
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Build a lookup table from binary dataset using composite keys."""
    logger.info("Building binary dataset lookup table...")

    lookup_table = {}
    total_records = 0
    duplicate_keys = 0

    # Process all files in all folds
    for fold_dir in sorted(binary_dir.glob("fold_*")):
        for file_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            file_path = fold_dir / file_name
            if not file_path.exists():
                continue

            logger.info(f"Processing {file_path}")
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f):
                    if sample_size and total_records >= sample_size:
                        break

                    try:
                        record = json_loads(line.strip())
                        key = create_matching_key(record)

                        if key in lookup_table:
                            duplicate_keys += 1
                        else:
                            lookup_table[key] = record

                        total_records += 1

                        if total_records % 10000 == 0:
                            logger.info(
                                f"Processed {total_records} records, {len(lookup_table)} unique keys"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Error processing line {line_num} in {file_path}: {e}"
                        )
                        continue

    logger.info(
        f"Built lookup table: {len(lookup_table)} unique keys from {total_records} total records"
    )
    logger.info(f"Duplicate keys found: {duplicate_keys}")

    return lookup_table


def process_tertile_file(args_tuple):
    """Process a single tertile file and add missing features."""
    tertile_file, output_file, binary_lookup, file_idx = args_tuple

    logger.info(f"Processing {tertile_file} -> {output_file}")

    stats = {
        "total_records": 0,
        "matched_records": 0,
        "features_added": 0,
        "unmatched_records": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(tertile_file, "r") as f_in, open(output_file, "w") as f_out:
        for line_num, line in enumerate(f_in):
            try:
                tertile_record = json_loads(line.strip())
                key = create_matching_key(tertile_record)

                stats["total_records"] += 1

                # Try to find matching binary record
                if key in binary_lookup:
                    binary_record = binary_lookup[key]

                    # Get missing features
                    missing_features = get_missing_features(
                        binary_record, tertile_record
                    )

                    if missing_features:
                        # Add missing features to tertile record
                        enhanced_record = tertile_record.copy()
                        enhanced_record.update(missing_features)

                        stats["matched_records"] += 1
                        stats["features_added"] += len(missing_features)

                        f_out.write(json_dumps(enhanced_record) + "\n")
                    else:
                        f_out.write(line)
                else:
                    stats["unmatched_records"] += 1
                    f_out.write(line)  # Keep original record

            except Exception as e:
                logger.warning(
                    f"Error processing line {line_num} in {tertile_file}: {e}"
                )
                f_out.write(line)  # Keep original record
                continue

    logger.info(f"Completed {tertile_file}: {stats}")
    return stats


def copy_metadata_files(tertile_dir: Path, output_dir: Path):
    """Copy metadata files from tertile directory to output directory."""
    logger.info("Copying metadata files...")

    metadata_files = [
        "fold_statistics.json",
        "per_fold_metadata.json",
        "dnt_manifest.json",
        "README.md",
    ]

    for file_name in metadata_files:
        src_file = tertile_dir / file_name
        if src_file.exists():
            dst_file = output_dir / file_name
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied {file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy all missing features from binary to tertile dataset"
    )
    parser.add_argument(
        "--binary-dir", type=Path, required=True, help="Binary dataset directory"
    )
    parser.add_argument(
        "--tertile-dir", type=Path, required=True, help="Tertile dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for enhanced dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit binary lookup table size for testing",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify results after copying"
    )

    args = parser.parse_args()

    if not args.binary_dir.exists():
        raise FileNotFoundError(f"Binary directory not found: {args.binary_dir}")
    if not args.tertile_dir.exists():
        raise FileNotFoundError(f"Tertile directory not found: {args.tertile_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build binary lookup table
    binary_lookup = build_binary_lookup_table(args.binary_dir, args.sample_size)

    if not binary_lookup:
        logger.error("No records found in binary dataset!")
        return

    # Collect all tertile files to process
    file_tasks = []
    for fold_dir in sorted(args.tertile_dir.glob("fold_*")):
        output_fold_dir = args.output_dir / fold_dir.name

        for file_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            tertile_file = fold_dir / file_name
            if tertile_file.exists():
                output_file = output_fold_dir / file_name
                file_tasks.append(
                    (tertile_file, output_file, binary_lookup, len(file_tasks))
                )

    logger.info(f"Processing {len(file_tasks)} files with {args.workers} workers...")

    # Process files in parallel
    total_stats = {
        "total_records": 0,
        "matched_records": 0,
        "features_added": 0,
        "unmatched_records": 0,
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_tertile_file, task) for task in file_tasks]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            try:
                stats = future.result()
                for key in total_stats:
                    total_stats[key] += stats[key]
            except Exception as e:
                logger.error(f"Task failed: {e}")

    # Copy metadata files
    copy_metadata_files(args.tertile_dir, args.output_dir)

    # Report final statistics
    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Total records processed: {total_stats['total_records']}")
    logger.info(f"Records with features added: {total_stats['matched_records']}")
    logger.info(f"Total features added: {total_stats['features_added']}")
    logger.info(f"Unmatched records: {total_stats['unmatched_records']}")

    match_rate = (
        (total_stats["matched_records"] / total_stats["total_records"]) * 100
        if total_stats["total_records"] > 0
        else 0
    )
    logger.info(f"Match rate: {match_rate:.1f}%")

    if args.verify:
        verify_results(args.output_dir, args.tertile_dir)


def verify_results(output_dir: Path, original_dir: Path):
    """Verify that features were copied correctly."""
    logger.info("Verifying results...")

    # Check first file in first fold
    original_file = original_dir / "fold_0" / "train.jsonl"
    enhanced_file = output_dir / "fold_0" / "train.jsonl"

    if not enhanced_file.exists():
        logger.error("Enhanced file not found!")
        return

    with open(original_file, "r") as f:
        original_record = json_loads(f.readline().strip())

    with open(enhanced_file, "r") as f:
        enhanced_record = json_loads(f.readline().strip())

    original_features = len(original_record)
    enhanced_features = len(enhanced_record)
    added_features = enhanced_features - original_features

    logger.info(f"Original record: {original_features} fields")
    logger.info(f"Enhanced record: {enhanced_features} fields")
    logger.info(f"Features added: {added_features}")

    # Count feat_new features
    feat_new_count = len([k for k in enhanced_record.keys() if "feat_new" in k])
    logger.info(f"feat_new* features in enhanced record: {feat_new_count}")


if __name__ == "__main__":
    main()
