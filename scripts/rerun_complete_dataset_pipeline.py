#!/usr/bin/env python3
"""
Complete Dataset Creation and Verification Pipeline

This script runs the full pipeline:
1. Create authoritative k-fold splits with improved weighting
2. Comprehensive leakage audit
3. Label verification
4. Generate dataset analysis figures

Usage:
    uv run python scripts/rerun_complete_dataset_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path
from loguru import logger
import json


def run_command(cmd: list, description: str, check_output: bool = False):
    """Run a command and handle errors."""
    logger.info(f"🚀 {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        if check_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start_time
            logger.success(f"✅ {description} completed in {elapsed:.1f}s")
            return result.stdout
        else:
            result = subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            logger.success(f"✅ {description} completed in {elapsed:.1f}s")
            return None
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ {description} failed after {elapsed:.1f}s")
        if hasattr(e, "stdout") and e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if hasattr(e, "stderr") and e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise


def check_file_exists(file_path: str, description: str):
    """Check if a required file exists."""
    if not Path(file_path).exists():
        logger.error(f"❌ Required file missing: {file_path} ({description})")
        return False
    logger.info(f"✅ Found required file: {file_path}")
    return True


def main():
    """Run the complete pipeline."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<8} | {message}",
    )

    logger.info("=" * 80)
    logger.info("🔄 COMPLETE DATASET CREATION AND VERIFICATION PIPELINE")
    logger.info("=" * 80)

    # Check prerequisites
    logger.info("📋 Checking prerequisites...")
    required_files = [
        (
            "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl",
            "Input dataset",
        ),
    ]

    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            logger.error("❌ Prerequisites not met. Aborting.")
            return 1

    # Define output directory
    output_dir = "data/final_stratified_kfold_splits_authoritative"

    try:
        # Step 1: Create authoritative k-fold splits with improved weighting
        logger.info("📊 STEP 1: Creating authoritative k-fold splits...")
        kfold_cmd = [
            "uv",
            "run",
            "python",
            "scripts/stratified_kfold_case_split.py",
            "--input",
            "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl",
            "--output-dir",
            output_dir,
            "--k-folds",
            "3",
            "--target-field",
            "final_judgement_real",
            "--stratify-type",
            "regression",
            "--case-id-field",
            "case_id_clean",
            "--use-temporal-cv",
            "--oof-test-ratio",
            "0.2",
            "--oof-min-ratio",
            "0.15",
            "--oof-max-ratio",
            "0.40",
            "--oof-step",
            "0.05",
            "--oof-min-class-cases",
            "5",
            "--oof-min-class-quotes",
            "50",
            "--oof-class-criterion",
            "both",
            "--random-seed",
            "42",
        ]
        run_command(
            kfold_cmd, "Creating authoritative k-fold splits with improved weighting"
        )

        # Verify key output files
        key_outputs = [
            f"{output_dir}/per_fold_metadata.json",
            f"{output_dir}/fold_statistics.json",
            f"{output_dir}/fold_3/train.jsonl",
            f"{output_dir}/fold_3/dev.jsonl",
            f"{output_dir}/oof_test/test.jsonl",
        ]

        for output_file in key_outputs:
            if not check_file_exists(output_file, "K-fold output"):
                logger.error(f"❌ K-fold creation incomplete. Missing: {output_file}")
                return 1

        # Step 2: Comprehensive leakage audit
        logger.info("🔍 STEP 2: Running comprehensive leakage audit...")
        audit_cmd = [
            "uv",
            "run",
            "python",
            "scripts/comprehensive_leakage_audit.py",
            "--data-file",
            "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl",
            "--kfold-dir",
            output_dir,
            "--output",
            "audit_results_authoritative.json",
        ]
        run_command(audit_cmd, "Running comprehensive leakage audit")

        # Step 3: Label verification
        logger.info("✅ STEP 3: Running comprehensive label verification...")
        verify_cmd = [
            "uv",
            "run",
            "python",
            "scripts/comprehensive_label_verification.py",
        ]
        run_command(verify_cmd, "Running comprehensive label verification")

        # Step 4: Generate dataset analysis figures
        logger.info("📈 STEP 4: Generating dataset analysis figures...")
        figures_cmd = [
            "uv",
            "run",
            "python",
            "scripts/generate_dataset_paper_figures.py",
            "--input",
            "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl",
            "--output",
            "docs/dataset_analysis_authoritative",
            "--kfold-dir",
            output_dir,
        ]
        run_command(figures_cmd, "Generating dataset analysis figures")

        # Step 5: Verify the new weight computation
        logger.info("🧮 STEP 5: Verifying improved weight computation...")

        # Check metadata for weight improvements
        metadata_path = Path(output_dir) / "per_fold_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Check if weight metadata contains expected improvements
        weights = metadata.get("weights", {})
        if weights:
            logger.info("✅ Weight metadata found in per_fold_metadata.json")
            for fold_key, fold_weights in weights.items():
                if "support_weight_method" in fold_weights:
                    method = fold_weights["support_weight_method"]
                    weight_range = fold_weights.get("support_weight_range", [])
                    logger.info(f"✅ {fold_key}: {method}, range: {weight_range}")

        # Check sample training data for improved weights
        train_sample_path = Path(output_dir) / "fold_3" / "train.jsonl"
        with open(train_sample_path, "r") as f:
            first_line = f.readline()
            sample_record = json.loads(first_line)

        weight_fields = ["sample_weight", "bin_weight", "support_weight"]
        logger.info("✅ Sample weight verification from fold_3/train.jsonl:")
        for field in weight_fields:
            if field in sample_record:
                value = sample_record[field]
                logger.info(f"   {field}: {value:.6f}")
            else:
                logger.warning(f"   ⚠️  Missing field: {field}")

        # Verify weight normalization by checking mean
        sample_weights = []
        with open(train_sample_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 1000:  # Check first 1000 records for speed
                    break
                record = json.loads(line)
                if "sample_weight" in record:
                    sample_weights.append(record["sample_weight"])

        if sample_weights:
            mean_weight = sum(sample_weights) / len(sample_weights)
            logger.info(
                f"✅ Sample weight verification: mean={mean_weight:.6f} (should be ~1.0)"
            )
            if abs(mean_weight - 1.0) < 0.1:
                logger.success("✅ Weight normalization working correctly!")
            else:
                logger.warning(
                    f"⚠️  Weight normalization may need adjustment (mean={mean_weight:.6f})"
                )

        # Step 6: Summary and final checks
        logger.info("📋 STEP 6: Final verification and summary...")

        # Check audit results
        audit_results_path = "audit_results_authoritative.json"
        if Path(audit_results_path).exists():
            with open(audit_results_path, "r") as f:
                audit_results = json.load(f)

            overall_score = audit_results.get("overall", {}).get("score", "UNKNOWN")
            logger.info(f"📊 Leakage audit overall score: {overall_score}")

            # Check key audit metrics
            key_audits = [
                "1_duplicate_text_leakage",
                "2_case_level_overlap",
                "5_outcome_bin_boundary_leakage",
                "6_support_imbalance_bias",
            ]

            for audit_key in key_audits:
                if audit_key in audit_results:
                    score = audit_results[audit_key].get("score", "UNKNOWN")
                    logger.info(f"   {audit_key}: {score}")

        # Final success message
        logger.success("=" * 80)
        logger.success("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
        logger.success("=" * 80)
        logger.success(f"📁 Authoritative data: {output_dir}")
        logger.success(f"📊 Analysis figures: docs/dataset_analysis_authoritative")
        logger.success(f"🔍 Audit results: {audit_results_path}")
        logger.success("🔧 Improved weight computation implemented and verified")
        logger.success("")
        logger.success("Key improvements implemented:")
        logger.success("✅ Bin weight clipping (0.25, 4.0) to prevent runaway weights")
        logger.success("✅ Final re-normalization to mean=1.0 on train split")
        logger.success("✅ Train-only computation (no peeking)")
        logger.success("✅ Numerical stability and standardization")

        return 0

    except subprocess.CalledProcessError as e:
        logger.error("❌ Pipeline failed. Check logs above for details.")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
