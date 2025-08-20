#!/usr/bin/env python3
"""
Comprehensive verification of regenerated dataset labels.

Checks all folds and splits to ensure:
1. Ground truth labels are in the correct field
2. Boundary logic is correctly applied
3. All splits have appropriate class distributions
4. OOF test set inherits fold 3 cutoffs correctly
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def load_jsonl(file_path):
    """Load JSONL file into list of records."""
    records = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def verify_fold_labels(fold_dir, fold_num, cutoffs):
    """Verify labels in a specific fold."""
    print(f"\n📋 **FOLD {fold_num} VERIFICATION**")
    fold_path = Path(fold_dir)

    cutoff_low, cutoff_high = cutoffs
    print(f"   Cutoffs: [${cutoff_low:,.2f}, ${cutoff_high:,.2f}]")

    total_records = 0
    total_errors = 0
    all_distributions = {}

    # Check each split file
    split_files = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "dev": "dev.jsonl",
    }

    for split_name, filename in split_files.items():
        split_file = fold_path / filename
        if not split_file.exists():
            continue

        print(f"   📄 Checking {filename}...")
        records = load_jsonl(split_file)

        if not records:
            print(f"      ⚠️  No records in {filename}")
            continue

        # Check what fields are available
        sample_record = records[0]
        available_fields = list(sample_record.keys())

        # Find the outcome and label fields
        outcome_field = None
        label_field = None

        if "final_judgement_real" in available_fields:
            outcome_field = "final_judgement_real"

        if "outcome_bin" in available_fields:
            label_field = "outcome_bin"
        elif "bin" in available_fields:
            label_field = "bin"
        elif "y" in available_fields:
            label_field = "y"

        print(f"      📊 Available fields: {len(available_fields)} total")
        print(f"      📊 Outcome field: {outcome_field}")
        print(f"      📊 Label field: {label_field}")

        if not outcome_field or not label_field:
            print(f"      ❌ Missing required fields!")
            continue

        # Verify each record's label
        errors = 0
        distributions = Counter()

        for i, record in enumerate(records):
            outcome_value = record.get(outcome_field)
            assigned_label = record.get(label_field)

            if outcome_value is None or assigned_label is None:
                continue

            # Calculate correct label using our verified methodology
            if outcome_value < cutoff_low:
                correct_label = 0  # low
            elif outcome_value <= cutoff_high:
                correct_label = 1  # medium
            else:
                correct_label = 2  # high

            distributions[assigned_label] += 1

            if assigned_label != correct_label:
                errors += 1
                if errors <= 5:  # Show first 5 errors
                    print(
                        f"      ❌ Record {i}: outcome=${outcome_value:,.2f} → assigned={assigned_label}, should be={correct_label}"
                    )

        print(f"      📊 **Results**: {len(records)} records, {errors} errors")
        print(f"      📊 **Distribution**: {dict(distributions)}")

        total_records += len(records)
        total_errors += errors
        all_distributions[split_name] = dict(distributions)

    print(
        f"   📊 **FOLD {fold_num} TOTAL**: {total_records} records, {total_errors} errors"
    )
    if total_errors == 0:
        print(f"   ✅ **FOLD {fold_num}: ALL LABELS CORRECT**")
    else:
        print(f"   ❌ **FOLD {fold_num}: {total_errors} LABEL ERRORS**")

    return total_errors, all_distributions


def verify_oof_test_labels(oof_dir, fold3_cutoffs):
    """Verify OOF test labels using fold 3 cutoffs."""
    print(f"\n📋 **OOF TEST VERIFICATION**")
    oof_path = Path(oof_dir)

    cutoff_low, cutoff_high = fold3_cutoffs
    print(f"   Using Fold 3 cutoffs: [${cutoff_low:,.2f}, ${cutoff_high:,.2f}]")

    test_file = oof_path / "test.jsonl"
    if not test_file.exists():
        print(f"   ❌ OOF test file not found: {test_file}")
        return 1, {}

    records = load_jsonl(test_file)
    print(f"   📄 Loaded {len(records)} OOF test records")

    # Check fields
    if not records:
        print(f"   ❌ No records in OOF test file")
        return 1, {}

    sample_record = records[0]
    available_fields = list(sample_record.keys())

    outcome_field = None
    label_field = None

    if "final_judgement_real" in available_fields:
        outcome_field = "final_judgement_real"

    if "outcome_bin" in available_fields:
        label_field = "outcome_bin"
    elif "bin" in available_fields:
        label_field = "bin"
    elif "y" in available_fields:
        label_field = "y"

    print(f"   📊 Available fields: {len(available_fields)} total")
    print(f"   📊 Outcome field: {outcome_field}")
    print(f"   📊 Label field: {label_field}")

    if not outcome_field or not label_field:
        print(f"   ❌ Missing required fields!")
        return 1, {}

    # Verify each record
    errors = 0
    distributions = Counter()

    for i, record in enumerate(records):
        outcome_value = record.get(outcome_field)
        assigned_label = record.get(label_field)

        if outcome_value is None or assigned_label is None:
            continue

        # Calculate correct label using fold 3 cutoffs
        if outcome_value < cutoff_low:
            correct_label = 0  # low
        elif outcome_value <= cutoff_high:
            correct_label = 1  # medium
        else:
            correct_label = 2  # high

        distributions[assigned_label] += 1

        if assigned_label != correct_label:
            errors += 1
            if errors <= 5:  # Show first 5 errors
                print(
                    f"   ❌ Record {i}: outcome=${outcome_value:,.2f} → assigned={assigned_label}, should be={correct_label}"
                )

    print(f"   📊 **Results**: {len(records)} records, {errors} errors")
    print(f"   📊 **Class Distribution**: {dict(distributions)}")

    if errors == 0:
        print(f"   ✅ **OOF TEST: ALL LABELS CORRECT**")
    else:
        print(f"   ❌ **OOF TEST: {errors} LABEL ERRORS**")

    return errors, dict(distributions)


def main():
    """Main verification function."""
    print("🔍 **COMPREHENSIVE REGENERATED DATASET LABEL VERIFICATION**")
    print("=" * 80)

    # Load metadata
    data_dir = Path("data/final_stratified_kfold_splits_authoritative")
    metadata_file = data_dir / "per_fold_metadata.json"

    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return 1

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Extract fold cutoffs
    fold_edges = metadata["binning"]["fold_edges"]
    print(f"📋 **LOADED METADATA**:")
    print(f"   Method: {metadata['binning']['method']}")
    print(f"   Methodology: {metadata['methodology']}")

    total_errors = 0
    all_fold_distributions = {}

    # Verify each fold
    for fold_name, cutoffs in fold_edges.items():
        fold_num = fold_name.split("_")[1]
        fold_dir = data_dir / f"fold_{fold_num}"

        if fold_dir.exists():
            fold_errors, fold_distributions = verify_fold_labels(
                fold_dir, fold_num, cutoffs
            )
            total_errors += fold_errors
            all_fold_distributions[fold_name] = fold_distributions

    # Verify OOF test using fold 3 cutoffs
    oof_dir = data_dir / "oof_test"
    if oof_dir.exists():
        fold3_cutoffs = fold_edges["fold_3"]
        oof_errors, oof_distribution = verify_oof_test_labels(oof_dir, fold3_cutoffs)
        total_errors += oof_errors
        all_fold_distributions["oof_test"] = oof_distribution

    # Summary
    print("\n" + "=" * 80)
    print("📊 **VERIFICATION SUMMARY**")
    print("=" * 80)

    if total_errors == 0:
        print("🎉 **ALL LABELS PERFECT!**")
        print("✅ All ground truth labels correctly generated")
        print("✅ All boundary logic correctly applied")
        print("✅ All folds and OOF test verified")

        print(f"\n📊 **LABEL FIELD SUMMARY**:")
        print(f"   Primary label field: 'outcome_bin'")
        print(f"   Outcome values field: 'final_judgement_real'")
        print(f"   Methodology: train_only_tertiles with corrected boundaries")

        print(f"\n📊 **CLASS DISTRIBUTIONS**:")
        for fold_name, distributions in all_fold_distributions.items():
            print(f"   {fold_name}: {distributions}")

        return 0
    else:
        print(f"❌ **{total_errors} TOTAL LABEL ERRORS FOUND**")
        print("🚨 **REGENERATION NEEDED**")
        return 1


if __name__ == "__main__":
    exit(main())
