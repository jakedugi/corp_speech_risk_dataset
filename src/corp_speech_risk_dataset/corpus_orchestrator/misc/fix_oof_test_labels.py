#!/usr/bin/env python3
"""
Fix OOF test labels by applying correct fold 3 cutoffs to raw outcome values.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def fix_oof_test_labels():
    """Fix OOF test labels using correct fold 3 cutoffs."""
    print("🔧 **FIXING OOF TEST LABELS**")
    print("=" * 50)

    # Load fold 3 cutoffs from metadata
    metadata_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/per_fold_metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    fold_3_edges = metadata["binning"]["fold_edges"]["fold_3"]
    cutoff_low = fold_3_edges[0]  # 710,257.86
    cutoff_high = fold_3_edges[1]  # 9,600,000.00

    print(f"📊 Fold 3 cutoffs: [{cutoff_low:,.2f}, {cutoff_high:,.2f}]")
    print(f"   Class 0 (low): < {cutoff_low:,.2f}")
    print(f"   Class 1 (medium): {cutoff_low:,.2f} to {cutoff_high:,.2f}")
    print(f"   Class 2 (high): > {cutoff_high:,.2f}")

    # Load OOF test data
    oof_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/oof_test/test.jsonl"
    )

    records = []
    with open(oof_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"📊 Loaded {len(records)} OOF test records")

    # Fix labels and track changes
    fixes_made = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    outcome_ranges = {0: [], 1: [], 2: []}

    for record in records:
        raw_outcome = record["final_judgement_real"]
        old_bin = record["outcome_bin"]

        # Apply correct binning logic
        if raw_outcome < cutoff_low:
            new_bin = 0  # Low
        elif raw_outcome <= cutoff_high:
            new_bin = 1  # Medium
        else:
            new_bin = 2  # High

        # Update record
        record["outcome_bin"] = new_bin

        # Track changes
        if old_bin != new_bin:
            fixes_made += 1

        class_counts[new_bin] += 1
        outcome_ranges[new_bin].append(raw_outcome)

    print(f"📊 **LABEL CORRECTIONS**:")
    print(f"   Fixed {fixes_made} out of {len(records)} labels")

    print(f"📊 **NEW CLASS DISTRIBUTION**:")
    for class_id in [0, 1, 2]:
        count = class_counts[class_id]
        if count > 0:
            min_val = min(outcome_ranges[class_id])
            max_val = max(outcome_ranges[class_id])
            print(
                f"   Class {class_id}: {count} quotes (${min_val:,.0f} to ${max_val:,.0f})"
            )
        else:
            print(f"   Class {class_id}: {count} quotes (empty)")

    # Write corrected data back
    backup_path = oof_path.with_suffix(".jsonl.backup")
    oof_path.rename(backup_path)
    print(f"💾 Created backup: {backup_path}")

    with open(oof_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"✅ Updated OOF test file: {oof_path}")

    # Verify against expected metadata
    expected_counts = metadata["oof_growth"]["native_counts"]["per_class_quotes"]
    print(f"📊 **COMPARISON WITH METADATA**:")
    for class_id in ["0", "1", "2"]:
        expected = expected_counts[class_id]
        actual = class_counts[int(class_id)]
        match = "✅" if abs(expected - actual) < 50 else "❌"  # Allow small variance
        print(f"   Class {class_id}: Expected {expected}, Got {actual} {match}")


def verify_all_folds_have_correct_labels():
    """Verify all fold data has correct labels."""
    print("\n🔍 **VERIFYING ALL FOLD LABELS**")
    print("=" * 50)

    # Load metadata
    metadata_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/per_fold_metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    for fold_id in range(4):
        print(f"\n📋 **Fold {fold_id}**")

        fold_edges = metadata["binning"]["fold_edges"][f"fold_{fold_id}"]
        print(f"   Cutoffs: [{fold_edges[0]:,.2f}, {fold_edges[1]:,.2f}]")

        # Check each split
        fold_dir = Path(
            f"data/final_stratified_kfold_splits_adaptive_oof/fold_{fold_id}"
        )
        for split in ["train.jsonl", "dev.jsonl"]:
            file_path = fold_dir / split
            if not file_path.exists():
                continue

            # Sample check first 100 records
            sample_records = []
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    sample_records.append(json.loads(line))

            if len(sample_records) == 0:
                continue

            # Check label consistency
            label_errors = 0
            for record in sample_records:
                raw_outcome = record["final_judgement_real"]
                actual_bin = record["outcome_bin"]

                # Calculate expected bin
                if raw_outcome < fold_edges[0]:
                    expected_bin = 0
                elif raw_outcome <= fold_edges[1]:
                    expected_bin = 1
                else:
                    expected_bin = 2

                if actual_bin != expected_bin:
                    label_errors += 1

            status = "✅" if label_errors == 0 else f"❌ ({label_errors} errors)"
            print(f"   {split}: {len(sample_records)} samples {status}")


def main():
    """Main function."""
    print("🚀 **OOF TEST LABEL FIX**")
    print("=" * 60)

    # 1. Fix OOF test labels
    fix_oof_test_labels()

    # 2. Verify other folds
    verify_all_folds_have_correct_labels()

    print(f"\n✅ **FIX COMPLETE**")
    print("=" * 60)
    print("🔧 **NEXT STEPS**:")
    print("1. ✅ OOF test labels corrected using fold 3 cutoffs")
    print("2. 🚀 Run full POLR pipeline with corrected data")
    print("3. 📊 All 3 classes should now be present in evaluation")


if __name__ == "__main__":
    main()
