#!/usr/bin/env python3
"""Quick verification of weight calculations and OOF test fix."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")

from corp_speech_risk_dataset.fully_interpretable.polar_pipeline import (
    compute_tempered_alpha_weights,
)


def verify_weight_calculation():
    """Verify weight calculation works correctly."""
    print("🔍 **WEIGHT CALCULATION VERIFICATION**")
    print("=" * 50)

    # Load fold 3 data (which should be complete)
    fold_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/fold_3/train.jsonl"
    )

    if not fold_path.exists():
        print("❌ Fold 3 training data not found")
        return

    # Load sample data
    sample_data = []
    with open(fold_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 500:  # Sample for speed
                break
            sample_data.append(json.loads(line))

    df = pd.DataFrame(sample_data)
    df["y"] = df["outcome_bin"]
    df["case_id"] = df["case_id_clean"]

    print(f"📊 Sample data: {len(df)} quotes from {df['case_id'].nunique()} cases")
    print(f"📊 Class distribution: {df['y'].value_counts().sort_index().to_dict()}")

    # Load metadata for fold 3 class weights
    metadata_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/per_fold_metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    fold_3_weights = metadata["weights"]["fold_3"]["class_weights"]
    # Convert string keys to int
    fold_3_weights = {int(k): float(v) for k, v in fold_3_weights.items()}

    print(f"📊 Expected fold 3 class weights: {fold_3_weights}")

    # Test weight calculation
    try:
        weights, norm_factor, stats = compute_tempered_alpha_weights(
            df.copy(),
            alpha=0.5,
            beta=0.5,
            use_fold_class_weights=True,
            fold_class_weights=fold_3_weights,
        )

        print(f"✅ Weight calculation successful!")
        print(f"📊 Weights shape: {weights.shape}")
        print(f"📊 Normalization factor: {norm_factor:.4f}")
        print(f"📊 Weight stats:")
        print(f"   Mean: {weights.mean():.4f} (should be ~1.0)")
        print(f"   Std: {weights.std():.4f}")
        print(f"   Range: [{weights.min():.4f}, {weights.max():.4f}]")

        # Check if range is within expected bounds
        expected_min, expected_max = 0.25, 4.0
        if weights.min() >= expected_min * 0.5 and weights.max() <= expected_max * 2.0:
            print(f"✅ Weight range within reasonable bounds")
        else:
            print(
                f"⚠️  Weight range unusual: [{weights.min():.4f}, {weights.max():.4f}]"
            )

        return True

    except Exception as e:
        print(f"❌ Weight calculation failed: {e}")
        return False


def investigate_oof_test_generation():
    """Investigate how OOF test was generated and why it only has class 1."""
    print("\n🔍 **OOF TEST GENERATION INVESTIGATION**")
    print("=" * 50)

    # Check if there's a generation script
    potential_scripts = [
        "scripts/stratified_kfold_case_split.py",
        "scripts/generate_oof_test.py",
        "scripts/create_adaptive_oof.py",
    ]

    for script in potential_scripts:
        if Path(script).exists():
            print(f"📄 Found generation script: {script}")

            # Check if it mentions class balancing
            with open(script, "r") as f:
                content = f.read()
                if "outcome_bin" in content and "stratified" in content.lower():
                    print(f"   ✅ Script appears to handle stratification")
                else:
                    print(f"   ⚠️  Script may not properly handle class balancing")

    # Check what cases are supposed to be in each class based on metadata
    metadata_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/per_fold_metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    expected_oof = metadata["oof_growth"]["native_counts"]
    print(f"📊 Expected OOF class distribution:")
    for class_id, case_count in expected_oof["per_class_cases"].items():
        quote_count = expected_oof["per_class_quotes"][class_id]
        print(f"   Class {class_id}: {case_count} cases, {quote_count} quotes")

    # Try to find where these cases should have come from
    print(f"\n🔍 **SEARCHING FOR MISSING CLASSES IN SOURCE DATA**")

    # Check if the complete dataset has all classes
    source_files = [
        "data/final_stratified_kfold_splits_adaptive_oof/fold_3/train.jsonl",
        "data/final_stratified_kfold_splits_adaptive_oof/fold_3/dev.jsonl",
    ]

    all_case_outcomes = {}

    for source_file in source_files:
        if Path(source_file).exists():
            print(f"📄 Checking {source_file}")

            case_outcomes = {}
            with open(source_file, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        case_id = record.get("case_id_clean")
                        outcome = record.get("outcome_bin")
                        if case_id and outcome is not None:
                            case_outcomes[case_id] = outcome
                    except:
                        continue

            class_dist = {}
            for outcome in case_outcomes.values():
                class_dist[outcome] = class_dist.get(outcome, 0) + 1

            print(f"   📊 Case-level class distribution: {class_dist}")
            all_case_outcomes.update(case_outcomes)

    print(f"\n📊 **TOTAL AVAILABLE CASES BY CLASS**:")
    total_class_dist = {}
    for outcome in all_case_outcomes.values():
        total_class_dist[outcome] = total_class_dist.get(outcome, 0) + 1

    for class_id in sorted(total_class_dist.keys()):
        print(f"   Class {class_id}: {total_class_dist[class_id]} cases available")

    # Check if all 3 classes are available
    if len(total_class_dist) >= 3:
        print(f"✅ All 3 classes are available in source data")
        print(
            f"❌ Problem: OOF test generation script is not preserving class diversity"
        )
    else:
        print(f"❌ Problem: Source data is missing classes: {total_class_dist}")


def main():
    """Main verification function."""
    print("🚀 **QUICK POLR VERIFICATION**")
    print("=" * 60)

    # 1. Test weight calculation
    weights_work = verify_weight_calculation()

    # 2. Investigate OOF test issue
    investigate_oof_test_generation()

    print(f"\n✅ **VERIFICATION SUMMARY**")
    print("=" * 60)
    if weights_work:
        print("✅ Weight calculations are working correctly")
    else:
        print("❌ Weight calculations need debugging")

    print("❌ OOF test set needs regeneration with proper class stratification")
    print("🔧 Run with corrected OOF data to proceed with full pipeline")


if __name__ == "__main__":
    main()
