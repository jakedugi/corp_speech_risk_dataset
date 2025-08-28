#!/usr/bin/env python3
"""
Create binary classification K-fold splits from a directory of enhanced JSONL files.

This script combines all individual JSONL files into a single dataset and then
creates binary classification splits using the same temporal CV logic.
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

# Import the binary split functions
from stratified_kfold_binary_split import (
    make_leakage_safe_splits_binary,
    extract_case_id,
    normalize_for_hash,
)
import hashlib


def load_all_files_from_directory(
    input_dir: str, pattern: str = "*.jsonl"
) -> List[Dict[str, Any]]:
    """Load and combine all JSONL files from a directory."""
    input_path = Path(input_dir)
    all_records = []

    jsonl_files = list(input_path.glob(pattern))
    logger.info(f"Found {len(jsonl_files)} JSONL files to combine")

    for i, file_path in enumerate(jsonl_files, 1):
        if i % 100 == 0:
            logger.info(f"Processing file {i}/{len(jsonl_files)}: {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping invalid JSON on line {line_num} in {file_path}: {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

    logger.info(
        f"Combined {len(all_records)} total records from {len(jsonl_files)} files"
    )
    return all_records


def save_jsonl_data(records: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"Saved {len(records)} records to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Create binary classification K-fold splits from directory of enhanced JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing enhanced JSONL files",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for K-fold splits"
    )
    parser.add_argument(
        "--k-folds", type=int, default=4, help="Number of folds (default: 4)"
    )
    parser.add_argument(
        "--target-field",
        default="final_judgement_real",
        help="Target field for stratification (default: final_judgement_real)",
    )
    parser.add_argument(
        "--case-id-field",
        default="case_id",
        help="Field name containing case ID (default: case_id)",
    )
    parser.add_argument(
        "--oof-test-ratio",
        type=float,
        default=0.15,
        help="Proportion of latest cases for out-of-fold test (default: 0.15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--intermediate-file",
        help="Optional path to save/load combined dataset before splitting",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip combining step and load from intermediate file",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    logger.info("Starting binary K-fold splitting from directory")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"K-folds: {args.k_folds}")
    logger.info(f"Target field: {args.target_field}")
    logger.info(f"Random seed: {args.random_seed}")

    # Step 1: Load/combine all data
    if args.skip_combine and args.intermediate_file:
        logger.info(f"Loading combined dataset from {args.intermediate_file}")
        if not os.path.exists(args.intermediate_file):
            logger.error(f"Intermediate file does not exist: {args.intermediate_file}")
            return 1

        records = []
        with open(args.intermediate_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} records from intermediate file")
    else:
        # Validate input directory
        if not os.path.exists(args.input_dir):
            logger.error(f"Input directory does not exist: {args.input_dir}")
            return 1

        # Load and combine all files
        logger.info("Combining all JSONL files from directory...")
        records = load_all_files_from_directory(args.input_dir)

        if not records:
            logger.error("No records loaded from directory!")
            return 1

        # Optionally save intermediate combined file
        if args.intermediate_file:
            logger.info(f"Saving combined dataset to {args.intermediate_file}")
            save_jsonl_data(records, args.intermediate_file)

    # Step 2: Convert to DataFrame and prepare for splitting
    logger.info("Converting to DataFrame and preparing for binary classification...")
    df = pd.DataFrame(records)

    # Extract case_id if not present
    if "case_id" not in df.columns:
        if "case_id_clean" in df.columns:
            logger.info("Using case_id_clean as case_id...")
            df["case_id"] = df["case_id_clean"]
        elif "_metadata_src_path" in df.columns:
            logger.info("Extracting case_id from _metadata_src_path...")
            df["case_id"] = df.apply(
                lambda row: extract_case_id(row, "_metadata_src_path"), axis=1
            )
        elif "doc_id" in df.columns:
            logger.info("Using doc_id as case_id...")
            df["case_id"] = df["doc_id"].astype(str)
        else:
            logger.error("No case ID source found in dataset!")
            return 1

    # Add case_year extraction if not present
    if "case_year" not in df.columns:
        logger.info("Extracting case years from case_id...")

        def extract_year_from_case_id(case_id):
            if not case_id:
                return 2020
            import re

            # Handle various patterns
            case_id_str = str(case_id)

            # Try to find 4-digit year
            match = re.search(r"(\d{4})", case_id_str)
            if match:
                year = int(match.group(1))
                if 1950 <= year <= 2030:
                    return year

            # Default fallback based on doc_id ranges (rough approximation)
            try:
                doc_id = int(re.search(r"\d+", case_id_str).group())
                if doc_id < 50000000:
                    return 2018
                elif doc_id < 100000000:
                    return 2019
                elif doc_id < 120000000:
                    return 2020
                elif doc_id < 140000000:
                    return 2021
                else:
                    return 2022
            except:
                return 2020

        df["case_year"] = df["case_id"].apply(extract_year_from_case_id)

    # Add text_hash if not present
    if "text_hash" not in df.columns:
        logger.info("Creating text_hash for deduplication...")
        df["text_hash"] = (
            df["text"].astype(str).apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        )

    # Add normalized text hash for better duplicate detection
    if "text_hash_norm" not in df.columns:
        logger.info("Creating normalized text_hash for robust deduplication...")
        df["text_hash_norm"] = (
            df["text"]
            .astype(str)
            .apply(normalize_for_hash)
            .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        )

    # Check target field
    if args.target_field not in df.columns:
        logger.error(f"Target field '{args.target_field}' not found in data!")
        logger.info(f"Available columns: {list(df.columns)}")
        return 1

    # Report on dataset
    logger.info(f"Dataset summary:")
    logger.info(f"  Total records: {len(df):,}")
    logger.info(f"  Unique cases: {df['case_id'].nunique():,}")
    logger.info(f"  Year range: {df['case_year'].min()}-{df['case_year'].max()}")
    logger.info(f"  Target field: {args.target_field}")
    logger.info(
        f"  Target range: ${df[args.target_field].min():,.0f} - ${df[args.target_field].max():,.0f}"
    )

    # Count features
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    logger.info(f"  Interpretable features: {len(feature_cols)}")

    if len(feature_cols) == 0:
        logger.warning(
            "⚠️ No interpretable features found! Make sure feature extraction was completed."
        )
    else:
        # Show feature categories
        categories = {}
        for fname in feature_cols:
            category = fname.split("_")[1] if len(fname.split("_")) > 1 else "unknown"
            categories[category] = categories.get(category, 0) + 1

        logger.info("  Feature categories:")
        for category, count in sorted(categories.items()):
            logger.info(f"    {category}: {count} features")

    # Step 3: Create binary classification splits
    logger.info("Creating binary classification K-fold splits...")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create leakage-safe splits with binary classification
    result_df = make_leakage_safe_splits_binary(
        df,
        k=args.k_folds,
        oof_ratio=args.oof_test_ratio,
        seed=args.random_seed,
        oof_min_ratio=0.15,
        oof_max_ratio=0.40,
        oof_step=0.05,
        min_class_cases=5,
        min_class_quotes=50,
        oof_criterion="both",
    )

    # Step 4: Save results by fold
    output_dir = Path(args.output_dir)

    for fold_idx in range(args.k_folds + 1):  # +1 to include final training fold
        fold_data = result_df[result_df["fold"] == fold_idx]
        if len(fold_data) == 0:
            continue

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if fold_idx == args.k_folds:  # Final training fold
            # Save train and dev splits
            for split in ["train", "dev"]:
                split_data = fold_data[fold_data["split"] == split]
                if len(split_data) > 0:
                    split_path = fold_dir / f"{split}.jsonl"
                    split_data.to_json(split_path, orient="records", lines=True)

            # Save case IDs
            train_cases = set(fold_data[fold_data["split"] == "train"]["case_id"])
            dev_cases = set(fold_data[fold_data["split"] == "dev"]["case_id"])

            case_ids_path = fold_dir / "case_ids.json"
            with open(case_ids_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_case_ids": list(train_cases),
                        "dev_case_ids": list(dev_cases),
                        "is_final_training_fold": True,
                        "classification_type": "binary",
                    },
                    f,
                    indent=2,
                )
        else:  # Regular CV folds
            # Save split files
            for split in ["train", "val", "test"]:
                split_data = fold_data[fold_data["split"] == split]
                if len(split_data) > 0:
                    split_path = fold_dir / f"{split}.jsonl"
                    split_data.to_json(split_path, orient="records", lines=True)

            # Save case IDs
            train_cases = set(fold_data[fold_data["split"] == "train"]["case_id"])
            val_cases = set(fold_data[fold_data["split"] == "val"]["case_id"])
            test_cases = set(fold_data[fold_data["split"] == "test"]["case_id"])

            case_ids_path = fold_dir / "case_ids.json"
            with open(case_ids_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_case_ids": list(train_cases),
                        "val_case_ids": list(val_cases),
                        "test_case_ids": list(test_cases),
                        "classification_type": "binary",
                    },
                    f,
                    indent=2,
                )

    # Save OOF test data if present
    oof_data = result_df[result_df["split"] == "oof_test"]
    if len(oof_data) > 0:
        oof_dir = output_dir / "oof_test"
        oof_dir.mkdir(parents=True, exist_ok=True)
        oof_path = oof_dir / "test.jsonl"
        oof_data.to_json(oof_path, orient="records", lines=True)

        # Save OOF case IDs
        oof_cases = list(set(oof_data["case_id"]))
        oof_case_ids_path = oof_dir / "case_ids.json"
        with open(oof_case_ids_path, "w", encoding="utf-8") as f:
            json.dump(
                {"test_case_ids": oof_cases, "classification_type": "binary"},
                f,
                indent=2,
            )

    # Save DNT manifest
    manifest = {
        "do_not_train": result_df.attrs.get("do_not_train", []),
        "classification_type": "binary",
        "source": "directory_combined",
        "interpretable_features": len(feature_cols),
    }
    manifest_path = output_dir / "dnt_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save per-fold metadata
    per_fold_metadata = {
        "binning": {
            "method": "train_only_binary_median_tie_safe",
            "fold_edges": result_df.attrs.get("per_fold_bin_edges", {}),
            "classification_type": "binary",
        },
        "weights": result_df.attrs.get("per_fold_weights", {}),
        "methodology": "temporal_rolling_origin_with_adaptive_oof_binary",
        "oof_growth": result_df.attrs.get("oof_growth_metadata", {}),
        "source": "directory_combined",
        "adaptive_features": {
            "oof_class_guarantee": True,
            "stratified_eval_blocks": True,
            "adaptive_val_frac": True,
            "tie_safe_binary": True,
            "classification_type": "binary",
        },
    }
    metadata_path = output_dir / "per_fold_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(per_fold_metadata, f, indent=2)

    # Save fold statistics
    stats = {
        "methodology": "temporal_rolling_origin_with_dnt_binary_from_directory",
        "classification_type": "binary",
        "source": "directory_combined",
        "folds": args.k_folds,
        "final_training_fold": True,
        "total_folds_including_final": args.k_folds + 1,
        "oof_test_ratio": args.oof_test_ratio,
        "total_records": len(result_df),
        "total_cases": result_df["case_id"].nunique(),
        "interpretable_features": len(feature_cols),
        "dnt_columns": len(manifest["do_not_train"]),
        "stratification_approach": "outcome_only_binary",
        "support_handling": "weighting_only",
        "leakage_prevention": {
            "temporal_splits": "rolling_origin",
            "per_fold_binning": "training_data_only_binary_median",
            "dnt_policy": "wrap_not_drop_expanded",
            "text_deduplication": "eval_vs_train_global",
            "support_policy": "weighting_not_stratification",
        },
        "binning_strategy": {
            "method": "train_only_binary_median",
            "bins": ["lower", "higher"],
            "quantiles": [0.5],
            "temporal_purity": "preserved",
            "composite_labels": "disabled",
            "per_fold_edges_saved": True,
        },
        "support_strategy": {
            "method": "inverse_sqrt_weighting",
            "clipping": [0.25, 4.0],
            "normalization": "per_fold",
            "tertiles": "reporting_only_dnt",
        },
    }

    stats_path = output_dir / "fold_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.success("Binary K-fold splitting complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Classification type: BINARY (lower/higher)")
    logger.info(f"Total records: {len(result_df):,}")
    logger.info(f"Total cases: {result_df['case_id'].nunique():,}")
    logger.info(f"Interpretable features: {len(feature_cols)}")
    logger.info(f"DNT columns: {len(manifest['do_not_train'])}")

    return 0


if __name__ == "__main__":
    exit(main())
