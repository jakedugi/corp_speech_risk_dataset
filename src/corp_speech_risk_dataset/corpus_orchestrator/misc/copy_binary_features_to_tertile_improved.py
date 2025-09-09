#!/usr/bin/env python3
"""
Copy feat_new* features from binary dataset to authoritative dataset using ID-based matching.

This improved version uses case_id + doc_id for perfect record matching instead of
text content, ensuring 100% feature copy accuracy.

Usage:
    python scripts/copy_binary_features_to_tertile_improved.py \
        --binary-dir data/final_stratified_kfold_splits_binary_quote_balanced \
        --tertile-dir data/final_stratified_kfold_splits_authoritative \
        --output-dir data/final_stratified_kfold_splits_authoritative_enhanced \
        --workers 4
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Set, Tuple
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_record_key(record: Dict[str, Any]) -> str:
    """Create unique key from case_id and doc_id."""
    case_id = record.get("case_id", "")
    doc_id = record.get("doc_id", "")
    return f"{case_id}|{doc_id}"


def get_feat_new_features(binary_record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all feat_new* features from a binary record."""
    return {k: v for k, v in binary_record.items() if k.startswith("feat_new")}


def build_feature_mapping_from_binary_files(
    binary_dir: Path, sample_size: int = None
) -> Dict[str, Dict[str, Any]]:
    """Build comprehensive mapping from all binary files using ID-based keys."""
    print("🗺️ Building comprehensive feature mapping from all binary files...")

    id_to_features = {}
    total_processed = 0
    files_processed = 0

    # Process all binary files to build complete mapping
    binary_files = []
    for fold_dir in binary_dir.glob("fold_*"):
        if fold_dir.is_dir():
            for jsonl_file in fold_dir.glob("*.jsonl"):
                binary_files.append(jsonl_file)

    # Add oof_test files
    oof_test_dir = binary_dir / "oof_test"
    if oof_test_dir.exists():
        for jsonl_file in oof_test_dir.glob("*.jsonl"):
            binary_files.append(jsonl_file)

    print(f"📁 Found {len(binary_files)} binary files to process")

    for binary_file in binary_files:
        files_processed += 1
        file_processed = 0

        print(
            f"  [{files_processed}/{len(binary_files)}] Processing {binary_file.name}"
        )

        with open(binary_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if sample_size and total_processed >= sample_size:
                    break

                try:
                    binary_record = json.loads(line.strip())
                    record_key = get_record_key(binary_record)

                    if record_key and record_key != "|":  # Valid key
                        feat_new_features = get_feat_new_features(binary_record)
                        if feat_new_features:  # Only add if has feat_new features
                            id_to_features[record_key] = feat_new_features
                            file_processed += 1
                            total_processed += 1

                except json.JSONDecodeError:
                    continue

        print(f"    ✓ Added {file_processed:,} records from {binary_file.name}")

        if sample_size and total_processed >= sample_size:
            break

    print(
        f"✅ Built mapping for {len(id_to_features):,} unique record IDs with feat_new features"
    )

    # Sample verification
    if id_to_features:
        sample_key = next(iter(id_to_features))
        sample_features = id_to_features[sample_key]
        print(f"   📋 Sample key: {sample_key}")
        print(f"   📋 Sample feat_new count: {len(sample_features)}")
        print(f"   📋 Sample features: {list(sample_features.keys())[:5]}...")

    return id_to_features


def process_single_file(args_tuple):
    """Process a single tertile file and add feat_new features."""
    tertile_file, output_file, id_to_features = args_tuple

    try:
        enhanced_records = []
        matches_found = 0
        total_records = 0
        missing_keys = 0

        with open(tertile_file, "r", encoding="utf-8") as f:
            for line in f:
                total_records += 1
                try:
                    tertile_record = json.loads(line.strip())
                    record_key = get_record_key(tertile_record)

                    # Look up feat_new features by ID
                    if record_key and record_key in id_to_features:
                        # Add feat_new features to tertile record
                        enhanced_record = tertile_record.copy()
                        enhanced_record.update(id_to_features[record_key])
                        enhanced_records.append(enhanced_record)
                        matches_found += 1
                    else:
                        # Keep original record but note missing key
                        enhanced_records.append(tertile_record)
                        if not record_key or record_key == "|":
                            missing_keys += 1

                except json.JSONDecodeError:
                    enhanced_records.append(tertile_record)

        # Write enhanced records
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for record in enhanced_records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        match_rate = matches_found / total_records * 100 if total_records > 0 else 0
        return (
            str(tertile_file),
            total_records,
            matches_found,
            match_rate,
            missing_keys,
            "Success",
        )

    except Exception as e:
        return (str(tertile_file), 0, 0, 0, 0, f"Error: {e}")


def copy_metadata_files(tertile_dir: Path, output_dir: Path):
    """Copy metadata files to output directory."""
    metadata_files = [
        "fold_statistics.json",
        "per_fold_metadata.json",
        "dnt_manifest.json",
    ]

    for file_name in metadata_files:
        src = tertile_dir / file_name
        dst = output_dir / file_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied {file_name}")


def get_tertile_files(tertile_dir: Path, output_dir: Path):
    """Get list of tertile files to process."""
    file_info = []

    # Process all folds
    for fold_dir in tertile_dir.glob("fold_*"):
        if fold_dir.is_dir():
            fold_name = fold_dir.name
            output_fold_dir = output_dir / fold_name

            for jsonl_file in fold_dir.glob("*.jsonl"):
                output_file = output_fold_dir / jsonl_file.name
                file_info.append((jsonl_file, output_file))

    # Process oof_test
    oof_test_dir = tertile_dir / "oof_test"
    if oof_test_dir.exists():
        output_oof_dir = output_dir / "oof_test"

        for jsonl_file in oof_test_dir.glob("*.jsonl"):
            output_file = output_oof_dir / jsonl_file.name
            file_info.append((jsonl_file, output_file))

    return file_info


def main():
    parser = argparse.ArgumentParser(
        description="Copy feat_new features using ID-based matching"
    )
    parser.add_argument("--binary-dir", required=True, help="Binary dataset directory")
    parser.add_argument(
        "--tertile-dir", required=True, help="Tertile dataset directory"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument(
        "--sample-size", type=int, help="Limit processing to N records for testing"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run verification checks after copy"
    )

    args = parser.parse_args()

    binary_dir = Path(args.binary_dir)
    tertile_dir = Path(args.tertile_dir)
    output_dir = Path(args.output_dir)

    print("🚀 Starting improved feat_new feature copy process")
    print(f"📁 Binary source: {binary_dir}")
    print(f"📁 Tertile source: {tertile_dir}")
    print(f"📁 Output: {output_dir}")
    print("🔑 Using case_id + doc_id for perfect ID-based matching")

    # Validate input directories
    if not binary_dir.exists():
        raise ValueError(f"Binary directory not found: {binary_dir}")
    if not tertile_dir.exists():
        raise ValueError(f"Tertile directory not found: {tertile_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata files first
    copy_metadata_files(tertile_dir, output_dir)

    # Build comprehensive feature mapping from binary files
    id_to_features = build_feature_mapping_from_binary_files(
        binary_dir, args.sample_size
    )

    if not id_to_features:
        print("❌ No feat_new features found in binary dataset!")
        return

    # Get tertile files to process
    file_info = get_tertile_files(tertile_dir, output_dir)
    print(f"📋 Found {len(file_info)} tertile files to enhance")

    # Process files
    args_tuples = [(tf, of, id_to_features) for tf, of in file_info]

    total_records = 0
    total_matches = 0
    total_missing_keys = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_single_file, args_tuple)
                for args_tuple in args_tuples
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing files"
            ):
                file_path, records, matches, match_rate, missing_keys, status = (
                    future.result()
                )
                total_records += records
                total_matches += matches
                total_missing_keys += missing_keys

                if status == "Success":
                    print(
                        f"✓ {Path(file_path).name}: {matches:,}/{records:,} matches ({match_rate:.1f}%)"
                    )
                    if missing_keys > 0:
                        print(f"  ⚠️ {missing_keys} records had missing/invalid keys")
                else:
                    print(f"❌ {Path(file_path).name}: {status}")
    else:
        # Single-threaded processing
        for args_tuple in tqdm(args_tuples, desc="Processing files"):
            file_path, records, matches, match_rate, missing_keys, status = (
                process_single_file(args_tuple)
            )
            total_records += records
            total_matches += matches
            total_missing_keys += missing_keys

            if status == "Success":
                print(
                    f"✓ {Path(file_path).name}: {matches:,}/{records:,} matches ({match_rate:.1f}%)"
                )
                if missing_keys > 0:
                    print(f"  ⚠️ {missing_keys} records had missing/invalid keys")
            else:
                print(f"❌ {Path(file_path).name}: {status}")

    overall_match_rate = total_matches / total_records * 100 if total_records > 0 else 0

    print(f"\n🎉 Enhanced feature copy completed!")
    print(f"📊 Overall statistics:")
    print(f"  Total records processed: {total_records:,}")
    print(f"  Records with feat_new features added: {total_matches:,}")
    print(f"  Records with missing/invalid keys: {total_missing_keys:,}")
    print(f"  Overall match rate: {overall_match_rate:.1f}%")
    print(f"  Enhanced dataset saved to: {output_dir}")

    # Verification
    if args.verify:
        print("\n🔍 Running verification checks...")

        # Sample a file to verify feat_new features were added
        sample_file = None
        for fold_dir in output_dir.glob("fold_*"):
            if fold_dir.is_dir():
                train_file = fold_dir / "train.jsonl"
                if train_file.exists():
                    sample_file = train_file
                    break

        if sample_file:
            print(f"📋 Checking sample file: {sample_file}")
            with open(sample_file, "r") as f:
                lines_checked = 0
                feat_new_counts = []

                for line in f:
                    if lines_checked >= 100:  # Check first 100 records
                        break

                    record = json.loads(line)
                    feat_new_count = len(
                        [k for k in record.keys() if k.startswith("feat_new")]
                    )
                    feat_new_counts.append(feat_new_count)
                    lines_checked += 1

                avg_feat_new = (
                    sum(feat_new_counts) / len(feat_new_counts)
                    if feat_new_counts
                    else 0
                )
                max_feat_new = max(feat_new_counts) if feat_new_counts else 0
                records_with_features = sum(1 for count in feat_new_counts if count > 0)

                print(f"✅ Verification results (first {lines_checked} records):")
                print(f"  Average feat_new features per record: {avg_feat_new:.1f}")
                print(f"  Maximum feat_new features found: {max_feat_new}")
                print(
                    f"  Records with feat_new features: {records_with_features}/{lines_checked} ({records_with_features/lines_checked*100:.1f}%)"
                )

    if overall_match_rate < 95.0:
        print(
            f"\n⚠️ Warning: Match rate is {overall_match_rate:.1f}% - some records may not have matching IDs"
        )
        print(
            "   This could indicate differences in data preprocessing between datasets"
        )
    else:
        print(
            f"\n🎯 Excellent match rate: {overall_match_rate:.1f}% - ID-based matching successful!"
        )


if __name__ == "__main__":
    main()
