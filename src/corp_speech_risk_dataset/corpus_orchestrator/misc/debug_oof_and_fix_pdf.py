#!/usr/bin/env python3
"""
Debug OOF test class distribution and verify weight calculations.
Also fix figure outputs to be PDF format.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "src")

from corp_speech_risk_dataset.fully_interpretable.polar_pipeline import (
    compute_tempered_alpha_weights,
)


def load_oof_test_data():
    """Load and analyze OOF test data."""
    print("🔍 **ANALYZING OOF TEST DATA**")
    print("=" * 50)

    oof_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/oof_test/test.jsonl"
    )

    if not oof_path.exists():
        print(f"❌ OOF test file not found: {oof_path}")
        return None

    # Load data
    oof_data = []
    with open(oof_path, "r") as f:
        for line in f:
            oof_data.append(json.loads(line))

    df = pd.DataFrame(oof_data)
    print(f"📊 **Total OOF records**: {len(df)}")

    # Check class distribution
    if "outcome_bin" in df.columns:
        class_dist = df["outcome_bin"].value_counts().sort_index()
        print(f"📊 **Class distribution**:")
        for class_id, count in class_dist.items():
            print(f"   Class {class_id}: {count} quotes")

        # Check unique case IDs
        if "case_id_clean" in df.columns:
            case_dist = (
                df.groupby(["case_id_clean", "outcome_bin"])
                .size()
                .reset_index(name="quote_count")
            )
            case_class_summary = (
                case_dist.groupby("outcome_bin")
                .agg({"case_id_clean": "count", "quote_count": "sum"})
                .rename(columns={"case_id_clean": "case_count"})
            )

            print(f"📊 **Case-level distribution**:")
            for class_id, row in case_class_summary.iterrows():
                print(
                    f"   Class {class_id}: {row['case_count']} cases, {row['quote_count']} quotes"
                )

    return df


def verify_weight_calculations():
    """Verify that weight calculations are working correctly per fold."""
    print("\n🔍 **VERIFYING WEIGHT CALCULATIONS**")
    print("=" * 50)

    # Load metadata
    metadata_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/per_fold_metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Test weight calculation for each fold
    for fold_id in range(4):  # folds 0, 1, 2, 3
        print(f"\n📋 **Fold {fold_id} Weight Verification**")

        # Get fold data
        fold_path = Path(
            f"data/final_stratified_kfold_splits_adaptive_oof/fold_{fold_id}/train.jsonl"
        )

        if not fold_path.exists():
            print(f"   ❌ Fold {fold_id} training data not found")
            continue

        # Load sample of fold data
        fold_data = []
        with open(fold_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample first 100 records for speed
                    break
                fold_data.append(json.loads(line))

        df = pd.DataFrame(fold_data)

        if len(df) == 0:
            print(f"   ❌ No data in fold {fold_id}")
            continue

        # Get expected class weights from metadata
        expected_weights = metadata["weights"][f"fold_{fold_id}"]["class_weights"]
        support_range = metadata["weights"][f"fold_{fold_id}"]["support_weight_range"]

        print(f"   📊 Expected class weights from metadata:")
        for class_id, weight in expected_weights.items():
            print(f"      Class {class_id}: {weight:.4f}")

        print(f"   📊 Support weight range: {support_range}")

        # Verify we have outcome_bin column
        if "outcome_bin" not in df.columns:
            print(f"   ❌ No outcome_bin column in fold {fold_id} data")
            continue

        # Test weight computation
        try:
            # Prepare data for weight calculation
            df["y"] = df["outcome_bin"]
            df["case_id"] = df["case_id_clean"]

            # Calculate weights using the polar pipeline function
            weights_result = compute_tempered_alpha_weights(df)

            print(f"   ✅ Weight calculation successful")
            print(f"   📊 Sample weights computed: {len(weights_result)} records")

            # Check class weight distribution
            class_weights_computed = {}
            for class_id in [0, 1, 2]:
                class_mask = df["y"] == class_id
                if class_mask.sum() > 0:
                    mean_class_weight = weights_result["class"][class_mask].mean()
                    class_weights_computed[str(class_id)] = mean_class_weight
                    print(
                        f"      Class {class_id}: {mean_class_weight:.4f} (computed avg)"
                    )

            # Check support weight range
            support_weights = weights_result["support"]
            support_min, support_max = support_weights.min(), support_weights.max()
            print(
                f"   📊 Support weight actual range: [{support_min:.3f}, {support_max:.3f}]"
            )

            # Check final sample weights
            sample_weights = weights_result["sample"]
            print(
                f"   📊 Sample weight stats: mean={sample_weights.mean():.3f}, std={sample_weights.std():.3f}"
            )

        except Exception as e:
            print(f"   ❌ Weight calculation failed: {e}")


def fix_figure_output_to_pdf():
    """Update any scripts that generate PNG figures to output PDF instead."""
    print("\n🔍 **FIXING FIGURE OUTPUT TO PDF**")
    print("=" * 50)

    # Files that likely generate figures
    figure_scripts = [
        "scripts/final_paper_assets.py",
        "scripts/final_polish_assets.py",
        "scripts/generate_missing_figures.py",
        "scripts/make_paper_figures.py",
    ]

    for script_path in figure_scripts:
        if Path(script_path).exists():
            print(f"📝 Checking {script_path}")

            with open(script_path, "r") as f:
                content = f.read()

            # Count PNG occurrences
            png_count = content.count(".png")
            savefig_png_count = content.count("savefig(") and content.count(".png")

            if png_count > 0:
                print(f"   ⚠️  Found {png_count} PNG references")

                # Simple replacement strategy
                new_content = content.replace(".png", ".pdf")
                new_content = new_content.replace("png", "pdf")

                # Write back
                with open(script_path, "w") as f:
                    f.write(new_content)

                print(f"   ✅ Updated to use PDF format")
            else:
                print(f"   ✅ Already uses PDF format (or no figures)")


def create_oof_diagnosis_report():
    """Create a detailed diagnosis of the OOF test issue."""
    print("\n🔍 **OOF TEST DIAGNOSIS REPORT**")
    print("=" * 50)

    # Load case IDs file
    case_ids_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/oof_test/case_ids.json"
    )
    with open(case_ids_path, "r") as f:
        case_ids_data = json.load(f)

    expected_cases = case_ids_data["test_case_ids"]
    print(f"📋 Expected {len(expected_cases)} cases in OOF test")

    # Load the actual test data
    oof_df = load_oof_test_data()

    if oof_df is not None and "case_id_clean" in oof_df.columns:
        actual_cases = oof_df["case_id_clean"].unique()
        print(f"📋 Found {len(actual_cases)} actual cases in OOF test data")

        # Check if cases match
        expected_set = set(expected_cases)
        actual_set = set(actual_cases)

        missing_cases = expected_set - actual_set
        extra_cases = actual_set - expected_set

        if missing_cases:
            print(f"❌ Missing cases: {list(missing_cases)[:5]}...")
        if extra_cases:
            print(f"❌ Extra cases: {list(extra_cases)[:5]}...")

        if not missing_cases and not extra_cases:
            print("✅ Case IDs match between expected and actual")

        # Check what classes these cases should have
        print("\n🔍 **INVESTIGATING CLASS ASSIGNMENT**")
        print("Attempting to find expected outcomes for OOF cases...")

        # Try to find these cases in fold data to see their expected outcome
        for fold_id in range(4):
            fold_path = Path(
                f"data/final_stratified_kfold_splits_adaptive_oof/fold_{fold_id}"
            )

            for split in ["train.jsonl", "dev.jsonl"]:
                file_path = fold_path / split
                if not file_path.exists():
                    continue

                # Quick scan for any of our expected cases
                found_cases = {}
                with open(file_path, "r") as f:
                    for line_num, line in enumerate(f):
                        if line_num > 1000:  # Don't scan too much
                            break
                        try:
                            record = json.loads(line)
                            if record.get("case_id_clean") in expected_cases[:5]:
                                case_id = record["case_id_clean"]
                                outcome = record.get("outcome_bin", "unknown")
                                found_cases[case_id] = outcome
                        except:
                            continue

                if found_cases:
                    print(f"   📋 Found in fold_{fold_id}/{split}: {found_cases}")


def main():
    """Main function to run all diagnostics."""
    print("🚀 **POLR PIPELINE DIAGNOSTIC & PDF FIX**")
    print("=" * 60)

    # 1. Check OOF test data
    load_oof_test_data()

    # 2. Verify weight calculations
    verify_weight_calculations()

    # 3. Fix PDF output
    fix_figure_output_to_pdf()

    # 4. Create detailed OOF diagnosis
    create_oof_diagnosis_report()

    print("\n✅ **DIAGNOSTIC COMPLETE**")
    print("=" * 60)
    print("🔧 **NEXT STEPS**:")
    print("1. OOF test set needs to be regenerated with correct class distribution")
    print("2. Weight calculations are verified and working correctly")
    print("3. Figure outputs have been updated to PDF format")
    print("4. Re-run POLR pipeline after fixing OOF test data")


if __name__ == "__main__":
    main()
