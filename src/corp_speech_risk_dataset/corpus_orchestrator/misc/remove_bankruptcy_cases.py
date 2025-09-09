#!/usr/bin/env python3
"""
Remove bankruptcy cases from the dataset entirely.
These cases have different legal contexts and are causing year extraction issues.
"""

import json
import re
from pathlib import Path


def extract_case_id(src_path: str) -> str:
    """Extract case ID from _src path."""
    match = re.search(r"/([^/]*:\d+-[^/]+_[^/]+)/entries/", src_path)
    if match:
        return match.group(1)
    match = re.search(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/", src_path)
    if match:
        return match.group(1)
    return "unknown"


def is_bankruptcy_case(case_id: str) -> bool:
    """Identify bankruptcy cases by court codes and patterns."""
    bankruptcy_patterns = [
        "_nysb",  # New York Southern District Bankruptcy
        "_paeb",  # Pennsylvania Eastern District Bankruptcy
        "_ksb",  # Kansas Bankruptcy
        "_nyeb",  # New York Eastern District Bankruptcy
        "_canb",  # California Northern District Bankruptcy
        "_txsb",  # Texas Southern District Bankruptcy
        "_flsb",  # Florida Southern District Bankruptcy
        # Add other bankruptcy court patterns as needed
    ]

    # Check for bankruptcy court suffixes
    for pattern in bankruptcy_patterns:
        if case_id.endswith(pattern):
            return True

    # Check for specific bankruptcy cases ONLY (not Court of Appeals)
    bankruptcy_cases = [
        "09-11435_nysb",  # New York Bankruptcy - REMOVE
        "17-00276_paeb",  # Pennsylvania Bankruptcy - REMOVE
        "15-10116_ksb",  # Kansas Bankruptcy - REMOVE
        # Keep: 24-60040_ca5 and 24-10951_ca5 (Court of Appeals, NOT bankruptcy)
    ]

    return case_id in bankruptcy_cases


def remove_bankruptcy_cases():
    """Remove all bankruptcy cases from the dataset."""

    print("🗑️  REMOVING BANKRUPTCY CASES FROM DATASET")
    print("=" * 60)

    input_file = "data/enhanced_combined/final_clean_dataset_leakage_safe.jsonl"
    output_file = "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl"

    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return

    bankruptcy_cases_found = set()
    bankruptcy_records_removed = 0
    total_records = 0
    kept_records = 0

    print("Processing dataset...")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            if line_num % 5000 == 0:
                print(f"  Processed {line_num:,} records...")

            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract case ID
            src_path = record.get("_src") or record.get("_metadata_src_path", "")
            if src_path:
                case_id = extract_case_id(src_path)

                # Check if this is a bankruptcy case
                if is_bankruptcy_case(case_id):
                    bankruptcy_cases_found.add(case_id)
                    bankruptcy_records_removed += 1
                    continue  # Skip this record

            # Keep this record
            outfile.write(line)
            kept_records += 1

    print(f"✅ Processing complete!")
    print()
    print(f"📊 REMOVAL SUMMARY:")
    print(f"   Total records processed: {total_records:,}")
    print(f"   Records kept: {kept_records:,}")
    print(f"   Records removed: {bankruptcy_records_removed:,}")
    print(f"   Removal rate: {bankruptcy_records_removed/total_records*100:.2f}%")
    print()

    print(f"🏛️  BANKRUPTCY CASES REMOVED:")
    for case_id in sorted(bankruptcy_cases_found):
        print(f"   - {case_id}")
    print(f"   Total cases removed: {len(bankruptcy_cases_found)}")

    print(f"\n💾 Clean dataset saved to: {output_file}")

    # Update k-fold splits to remove bankruptcy cases
    update_kfold_splits(bankruptcy_cases_found)

    return bankruptcy_cases_found, kept_records, bankruptcy_records_removed


def update_kfold_splits(bankruptcy_cases_to_remove):
    """Update k-fold splits to remove bankruptcy cases."""

    print(f"\n🔄 UPDATING K-FOLD SPLITS...")

    kfold_dir = Path("data/final_stratified_kfold_splits_leakage_safe")
    if not kfold_dir.exists():
        print(f"❌ K-fold directory not found: {kfold_dir}")
        return

    for fold_dir in kfold_dir.glob("fold_*"):
        if not fold_dir.is_dir():
            continue

        case_ids_file = fold_dir / "case_ids.json"
        if not case_ids_file.exists():
            continue

        # Load fold data
        with open(case_ids_file, "r") as f:
            fold_data = json.load(f)

        # Remove bankruptcy cases from each split
        updated = False
        for split_name in ["train_case_ids", "val_case_ids", "test_case_ids"]:
            if split_name in fold_data:
                original_count = len(fold_data[split_name])
                fold_data[split_name] = [
                    case_id
                    for case_id in fold_data[split_name]
                    if case_id not in bankruptcy_cases_to_remove
                ]
                new_count = len(fold_data[split_name])

                if original_count != new_count:
                    updated = True
                    print(
                        f"   {fold_dir.name} {split_name}: {original_count} → {new_count} cases"
                    )

        # Save updated fold data
        if updated:
            with open(case_ids_file, "w") as f:
                json.dump(fold_data, f, indent=2)

    print("✅ K-fold splits updated!")


if __name__ == "__main__":
    bankruptcy_cases, kept_records, removed_records = remove_bankruptcy_cases()
