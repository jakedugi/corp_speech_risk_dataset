#!/usr/bin/env python3
"""
Fix boundary labeling errors in ALL fold data to match exact documented methodology.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def fix_fold_boundary_labels(fold_id, fold_edges):
    """Fix boundary labeling errors for a specific fold."""
    print(f"\n🔧 **FIXING FOLD {fold_id} BOUNDARY LABELS**")

    cutoff_low, cutoff_high = fold_edges
    print(f"   Cutoffs: [${cutoff_low:,.2f}, ${cutoff_high:,.2f}]")

    def get_correct_bin(outcome_value):
        """Correct binning logic matching documented methodology."""
        if outcome_value < cutoff_low:
            return 0  # low
        elif outcome_value <= cutoff_high:
            return 1  # medium
        else:
            return 2  # high

    # Fix each split in this fold
    fold_dir = Path(f"data/final_stratified_kfold_splits_adaptive_oof/fold_{fold_id}")
    splits_to_fix = ["train.jsonl", "dev.jsonl"]

    total_fixes = 0

    for split_file in splits_to_fix:
        file_path = fold_dir / split_file
        if not file_path.exists():
            continue

        print(f"   📄 Processing {split_file}...")

        # Load records
        records = []
        with open(file_path, "r") as f:
            for line in f:
                records.append(json.loads(line))

        # Fix labels
        split_fixes = 0
        class_counts = {0: 0, 1: 0, 2: 0}

        for record in records:
            raw_outcome = record["final_judgement_real"]
            old_bin = record["outcome_bin"]
            correct_bin = get_correct_bin(raw_outcome)

            if old_bin != correct_bin:
                record["outcome_bin"] = correct_bin
                split_fixes += 1
                if split_fixes <= 3:  # Show first few fixes
                    print(f"      🔧 ${raw_outcome:,.0f}: {old_bin} → {correct_bin}")

            class_counts[correct_bin] += 1

        total_fixes += split_fixes

        # Write corrected data back
        if split_fixes > 0:
            backup_path = file_path.with_suffix(".jsonl.backup")
            if not backup_path.exists():  # Don't overwrite existing backup
                file_path.rename(backup_path)
                print(f"      💾 Created backup: {backup_path.name}")

            with open(file_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            print(f"      ✅ Fixed {split_fixes} labels in {len(records)} records")
        else:
            print(f"      ✅ No fixes needed in {len(records)} records")

        print(f"      📊 Distribution: {dict(class_counts)}")

    print(f"   📊 **Total fixes for fold {fold_id}**: {total_fixes}")
    return total_fixes


def main():
    """Fix boundary labels in all folds."""
    print("🚀 **FIXING ALL FOLD BOUNDARY LABELS**")
    print("=" * 70)

    # Load metadata
    metadata_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/per_fold_metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Fix each fold
    total_fixes_all_folds = 0

    for fold_id in range(4):  # folds 0, 1, 2, 3
        fold_edges = metadata["binning"]["fold_edges"][f"fold_{fold_id}"]
        fold_fixes = fix_fold_boundary_labels(fold_id, fold_edges)
        total_fixes_all_folds += fold_fixes

    print(f"\n✅ **BOUNDARY LABEL FIX COMPLETE**")
    print("=" * 70)
    print(f"📊 **Total fixes across all folds**: {total_fixes_all_folds}")

    if total_fixes_all_folds > 0:
        print("🔧 **Boundary handling corrected**:")
        print("   ✅ Class 0 (low): < cutoff_low")
        print("   ✅ Class 1 (medium): >= cutoff_low AND <= cutoff_high")
        print("   ✅ Class 2 (high): > cutoff_high")
        print("\n🚀 **Now ready for POLR training pipeline!**")
    else:
        print("✅ **No fixes needed** - all labels already correct!")


if __name__ == "__main__":
    main()
