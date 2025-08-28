#!/usr/bin/env python3
"""
Safely append only new interpretable features to JSONL files without overwriting existing features.

This script specifically adds only the new features that were recently implemented
while preserving all existing features and data structure.
"""

import argparse
import os
import sys

import shutil
from pathlib import Path
from typing import Dict, Any, List, Set
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from loguru import logger

try:
    import orjson as json

    USE_ORJSON = True
    logger.info("Using orjson for faster JSON processing")
except ImportError:
    import json

    USE_ORJSON = False
    logger.warning("orjson not available, using standard json (slower)")

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


# List of all new features we're adding (101 total: 31 original + 19 additional + 7 final + 24 new + 20 newest)
NEW_FEATURE_NAMES = [
    # First batch: Original 31 features
    "new_deception_near_first_person_count",
    "new_deception_near_first_person_norm",
    "new_number_near_guarantee_count",
    "new_number_near_guarantee_norm",
    "new_caps_near_superlatives_count",
    "new_caps_near_superlatives_norm",
    "new_contrast_discourse_near_deception_count",
    "new_contrast_discourse_near_deception_norm",
    "new_low_certainty_modal_clustering",
    "new_hedges_near_future_tense_count",
    "new_hedges_near_future_tense_norm",
    "new_sentence_length_variation_near_risks",
    "new_neutral_to_compliance_transition_rate",
    "new_word_count_in_conditional_sentences",
    "new_conditional_word_share",
    "new_deception_to_contrast_transition_count",
    "new_deception_to_contrast_transition_norm",
    "new_num_approx_modifier_proximity",
    "new_num_approx_modifier_proximity_norm",
    "new_num_range_expression_count",
    "new_num_range_expression_norm",
    "new_percent_share_of_numbers",
    "new_hedged_number_ratio",
    "new_date_specificity_index",
    "new_parenthetical_token_share",
    "new_clause_initial_conditional_share",
    "new_contrast_then_hedge_count",
    "new_contrast_then_hedge_norm",
    "new_negated_superlative_proximity",
    "new_negated_superlative_proximity_norm",
    "new_attribution_verb_density",
    # Second batch: Additional 19 features
    "new2_attr_org_vs_person_ratio",
    "new2_attr_sentence_hedge_rate",
    "new2_negated_deception_norm",
    "new2_deception_near_disclaimer_norm",
    "new2_neutral_run_max_ratio",
    "new2_neutral_run_mean",
    "new2_risk_category_entropy",
    "new2_deception_attr_min_dist",
    "new2_modal_balance_index",
    "new2_hedged_deception_share",
    "new2_deception_near_attribution_prox",
    "new2_attr_verb_near_neutral_transition",
    "new2_deception_cluster_density",
    "new2_attr_verb_in_deception_sentences",
    "new2_neutral_to_deception_transition_rate",
    "new2_attribution_verb_clustering_score",
    "new2_deception_to_attribution_transition",
    "new2_neutral_transition_near_deception",
    "new2_attributed_deception_ratio",
    # Third batch: Final 7 features (non-redundant)
    "new3_attributed_sentence_share",
    "new3_attribution_lead_vs_first_risk",
    "new3_attributed_risk_share",
    "new3_neutral_edge_coverage",
    "new3_neutral_sandwich_flag",
    "new3_conditionalized_risk_share",
    "new3_scope_gated_risk_share",
    # Fourth batch: 24 additional features (unique, non-redundant)
    "new4_hedge_cluster_density",
    "new4_compliance_cluster_density",
    "new4_disclaimer_cluster_density",
    "new4_scope_limiter_cluster_density",
    "new4_neutral_to_hedge_transition_rate",
    "new4_neutral_to_scope_limiter_transition_rate",
    "new4_hedge_to_neutral_transition_rate",
    "new4_neutral_to_disclaimer_transition_rate",
    "new4_disclaimer_to_neutral_transition_rate",
    "new4_neutral_transition_symmetry",
    "new4_attribution_verb_near_compliance",
    "new4_attribution_verb_near_scope_limiter",
    "new4_attribution_verb_near_disclaimer",
    "new4_attribution_verb_near_hedge_transition",
    "new4_scope_limiter_run_mean",
    "new4_disclaimer_run_mean",
    "new4_compliance_edge_coverage",
    "new4_hedge_edge_coverage",
    "new4_attribution_lead_lag_signed",
    "new4_risk_core_span_ratio",
    "new4_deception_isolation_score",
    "new4_neutral_recovery_after_deception",
    "new4_conditionalized_attr_share",
    "new4_tail_neutral_gain",
    # Fifth batch: 20 newest features (advanced attribution, edge, and risk analysis)
    "new5_positive_qualifiers_cluster_density",
    "new5_clarity_cluster_density",
    "new5_neutral_to_positive_qualifiers_transition_rate",
    "new5_positive_qualifiers_to_neutral_transition_rate",
    "new5_attribution_verb_near_positive_qualifiers",
    "new5_policy_procedure_edge_coverage",
    "new5_remediation_edge_coverage",
    "new5_policy_procedure_run_mean",
    "new5_neutral_recovery_after_ambiguity",
    "new5_conditionalized_positive_share",
    "new5_attr_edge_span_coverage",
    "new5_attr_hedge_cohesion",
    "new5_attr_contrast_after_rate",
    "new5_neutral_slope_5bins",
    "new5_deesc_half_life",
    "new5_risk_gate_lead_distance",
    "new5_risk_clause_hedge_ratio",
    "new5_epistemic_over_deontic_in_attr",
    "new5_negated_risk_in_attr_rate",
    "new5_neutral_bridge_rate",
]


def get_existing_features(record: Dict[str, Any]) -> Set[str]:
    """Get all existing feature names from a record."""
    existing_features = set()
    for key in record.keys():
        if key.startswith("feat_") or key.startswith("interpretable_"):
            existing_features.add(key)
    return existing_features


def extract_only_new_features(
    extractor: InterpretableFeatureExtractor,
    text: str,
    context: str,
    existing_features: Set[str],
) -> Dict[str, float]:
    """Extract only the new features that don't already exist."""
    # Extract ALL features first
    all_features = extractor.extract_features(text, context)

    # Filter to only new features
    new_features = {}
    for feature_name in NEW_FEATURE_NAMES:
        if feature_name in all_features:
            # Check if this feature already exists with any prefix
            feature_exists = any(
                existing_key.endswith(feature_name) or feature_name in existing_key
                for existing_key in existing_features
            )

            if not feature_exists:
                new_features[f"feat_{feature_name}"] = all_features[feature_name]

    return new_features


def process_single_file(args_tuple):
    """Process a single JSONL file and add only new features."""
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
                    if USE_ORJSON:
                        record = json.loads(line)
                    else:
                        record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError if not USE_ORJSON else ValueError as e:
                    logger.warning(
                        f"Skipping invalid JSON on line {line_num} in {input_file}: {e}"
                    )
                    continue

        if not records:
            logger.warning(f"No valid records found in {input_file}")
            return input_file, 0, "No valid records"

        # Check if any new features are missing
        sample_record = records[0] if records else {}
        existing_features = get_existing_features(sample_record)

        missing_new_features = []
        for feature_name in NEW_FEATURE_NAMES:
            feature_exists = any(
                existing_key.endswith(feature_name) or feature_name in existing_key
                for existing_key in existing_features
            )
            if not feature_exists:
                missing_new_features.append(feature_name)

        if not missing_new_features:
            return input_file, len(records), "All features exist"

        logger.info(
            f"Adding {len(missing_new_features)} missing features to {os.path.basename(input_file)} ({len(records)} records)"
        )

        # Process each record
        enhanced_records = []
        features_added_count = 0
        total_records = len(records)

        for i, record in enumerate(records):
            # Progress reporting for large files
            if total_records > 1000 and i % 1000 == 0 and i > 0:
                logger.info(
                    f"  Progress: {i}/{total_records} records processed ({i/total_records*100:.1f}%)"
                )

            # Get text and context
            text = record.get(text_field, "")
            context = record.get(context_field, "")

            if not text:
                enhanced_records.append(record)
                continue

            # Extract only new features
            try:
                existing_features = get_existing_features(record)
                new_features = extract_only_new_features(
                    extractor, text, context, existing_features
                )

                # Create enhanced record
                enhanced_record = record.copy()

                # Add only new features
                for feature_name, feature_value in new_features.items():
                    enhanced_record[feature_name] = feature_value
                    features_added_count += 1

                enhanced_records.append(enhanced_record)

            except Exception as e:
                logger.error(
                    f"Error extracting features for record {i} in {input_file}: {e}"
                )
                # Add record without new features to maintain alignment
                enhanced_records.append(record)

        # Write enhanced records to a temporary file first
        temp_file = output_file + ".tmp"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(temp_file, "w", encoding="utf-8") as f:
            for record in enhanced_records:
                if USE_ORJSON:
                    json_bytes = json.dumps(record, option=json.OPT_SERIALIZE_NUMPY)
                    f.write(json_bytes.decode("utf-8"))
                else:
                    json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        # Atomically replace the original file
        shutil.move(temp_file, output_file)

        return (
            input_file,
            len(enhanced_records),
            f"Success - added {features_added_count} feature instances",
        )

    except Exception as e:
        error_msg = f"Error processing {input_file}: {e}"
        logger.error(error_msg)
        return input_file, 0, error_msg


def get_kfold_file_pairs(kfold_dir: str) -> List[tuple]:
    """Get list of (input_file, output_file) pairs for k-fold structure."""
    kfold_path = Path(kfold_dir)

    file_pairs = []

    # Process each fold directory
    for fold_dir in kfold_path.glob("fold_*"):
        if fold_dir.is_dir():
            # Process each JSONL file in the fold
            for jsonl_file in fold_dir.glob("*.jsonl"):
                # Output to the same location (in-place update)
                file_pairs.append((str(jsonl_file), str(jsonl_file)))

    # Also process oof_test if it exists
    oof_test_dir = kfold_path / "oof_test"
    if oof_test_dir.exists():
        for jsonl_file in oof_test_dir.glob("*.jsonl"):
            file_pairs.append((str(jsonl_file), str(jsonl_file)))

    return file_pairs


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Safely append only new interpretable features to k-fold JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "kfold_dir", help="K-fold directory containing fold_* subdirectories"
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
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=False,
        help="Create backup copies before modifying files",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<8} | {message}",
    )

    logger.info("Starting safe append of new interpretable features")
    logger.info(f"K-fold directory: {args.kfold_dir}")
    logger.info(f"Text field: {args.text_field}")
    logger.info(f"Context field: {args.context_field}")
    logger.info(f"Number of new features to add: {len(NEW_FEATURE_NAMES)}")

    # Validate input directory
    if not os.path.exists(args.kfold_dir):
        logger.error(f"K-fold directory does not exist: {args.kfold_dir}")
        return 1

    # Get file pairs
    logger.info("Scanning for k-fold files to process...")
    file_pairs = get_kfold_file_pairs(args.kfold_dir)

    if not file_pairs:
        logger.error(f"No JSONL files found in k-fold structure at {args.kfold_dir}")
        return 1

    logger.info(f"Found {len(file_pairs)} files to process")

    if args.dry_run:
        logger.info("DRY RUN - showing files that would be processed:")
        for i, (input_file, output_file) in enumerate(file_pairs):
            rel_path = os.path.relpath(input_file, args.kfold_dir)
            logger.info(f"  {i+1}. {rel_path}")
        return 0

    # Create backups if requested
    if args.backup:
        logger.info("Creating backup copies...")
        backup_dir = f"{args.kfold_dir}_backup_{int(time.time())}"
        shutil.copytree(args.kfold_dir, backup_dir)
        logger.info(f"Backup created at: {backup_dir}")

    # Prepare extractor configuration (all features enabled)
    extractor_config = {
        "include_lexicons": True,
        "include_sequence": True,
        "include_linguistic": True,
        "include_structural": True,
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
    skipped = 0
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
                elif "All features exist" in status:
                    skipped += 1
                    total_records += record_count
                else:
                    failed += 1
                    logger.error(
                        f"Failed to process {os.path.basename(file_path)}: {status}"
                    )

                if i % 5 == 0 or i == len(file_pairs):
                    logger.info(
                        f"Progress: {i}/{len(file_pairs)} files processed ({successful} enhanced, {skipped} skipped, {failed} failed)"
                    )

            except Exception as e:
                failed += 1
                logger.error(
                    f"Exception processing {os.path.basename(input_file)}: {e}"
                )

    end_time = time.time()
    duration = end_time - start_time

    # Final summary
    logger.success(f"Safe feature append complete!")
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(
        f"Files processed: {successful} enhanced, {skipped} skipped, {failed} failed"
    )
    logger.info(f"Total records processed: {total_records:,}")
    logger.info(f"Average processing rate: {len(file_pairs)/duration:.1f} files/second")

    if failed > 0:
        logger.warning(f"⚠️ {failed} files failed to process - check logs above")
        return 1

    # Report on features added
    if successful > 0:
        logger.success(
            f"✅ Successfully added {len(NEW_FEATURE_NAMES)} new features to {successful} files"
        )
        logger.info("New features added:")
        for i, feature_name in enumerate(NEW_FEATURE_NAMES, 1):
            logger.info(f"  {i:2d}. feat_{feature_name}")

    if skipped > 0:
        logger.info(f"ℹ️ {skipped} files already had all new features and were skipped")

    return 0


if __name__ == "__main__":
    exit(main())
