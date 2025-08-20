#!/usr/bin/env python3
"""
Fix the 2 appellate court cases that don't have extractable years.
These follow the pattern: YY-NNNNN_caX (e.g., 24-10951_ca5)
"""

import json
import re
from pathlib import Path


def extract_year_from_case_id_enhanced(case_id: str) -> int | None:
    """Enhanced year extraction that handles appellate court patterns."""
    if not case_id or case_id == "unknown":
        return None

    # NEW: Handle appellate court pattern: YY-NNNNN_caX (e.g., 24-10951_ca5)
    match = re.search(r"^(\d{2})-\d+_ca\d+$", case_id)
    if match:
        year_suffix = int(match.group(1))
        # Convert 2-digit year to 4-digit (24 -> 2024)
        if year_suffix <= 30:  # Assume 24 = 2024, not 1924
            return 2000 + year_suffix
        else:
            return 1900 + year_suffix

    # Try pattern: [district:]YY-[type]-[number]_[court]
    match = re.search(r"(\d{1,2})-(?:cv|cr|md|misc|civ)", case_id)
    if match:
        year_suffix = int(match.group(1))
        # Convert 2-digit year to 4-digit (21 -> 2021, 99 -> 1999)
        if year_suffix <= 30:  # Assume 21 = 2021, not 1921
            return 2000 + year_suffix
        else:
            return 1900 + year_suffix

    # Try pattern: [district:][20]YY-[type] (4-digit year)
    match = re.search(r"(?:^|\D)(\d{4})-(?:cv|cr|md|misc|civ)", case_id)
    if match:
        return int(match.group(1))

    # Try alternative pattern: number:number-type
    match = re.search(r"(\d+):(\d+)-", case_id)
    if match:
        year = int(match.group(2))
        if year > 90:
            return 1900 + year
        else:
            return 2000 + year

    # Try 4-digit year anywhere in the string
    match = re.search(r"(\d{4})", case_id)
    if match:
        year = int(match.group(1))
        if 1950 <= year <= 2030:
            return year

    return None


def test_enhanced_extraction():
    """Test the enhanced extraction function."""
    test_cases = [
        "24-10951_ca5",  # Should extract 2024
        "24-60040_ca5",  # Should extract 2024
        "1:21-cv-01234_nysd",  # Should extract 2021
        "3:16-cv-03615_cand",  # Should extract 2016
    ]

    print("Testing enhanced year extraction:")
    for case_id in test_cases:
        year = extract_year_from_case_id_enhanced(case_id)
        print(f"  {case_id} -> {year}")

    return True


def verify_fix():
    """Verify that the fix works for all cases in the dataset."""
    file_path = "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl"

    cases_with_years = set()
    cases_without_years = set()
    all_case_ids = set()

    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            case_id = record.get("case_id_clean")
            if not case_id:
                continue

            all_case_ids.add(case_id)

            # Try to extract year with enhanced function
            year = extract_year_from_case_id_enhanced(case_id)

            if year is not None:
                cases_with_years.add(case_id)
            else:
                cases_without_years.add(case_id)

    print(f"\nVerification results:")
    print(f"Total unique cases: {len(all_case_ids)}")
    print(f"Cases with extractable years: {len(cases_with_years)}")
    print(f"Cases without extractable years: {len(cases_without_years)}")

    if cases_without_years:
        print(f"Remaining cases without years:")
        for case_id in sorted(cases_without_years):
            print(f"  {case_id}")
        return False
    else:
        print("✓ SUCCESS: All cases now have extractable years!")
        return True


def main():
    print("=" * 60)
    print("FIXING APPELLATE COURT CASE YEAR EXTRACTION")
    print("=" * 60)

    # Test the enhanced extraction
    test_enhanced_extraction()

    # Verify the fix works
    if verify_fix():
        print(f"\n{'='*60}")
        print("READY TO UPDATE CODE:")
        print("The enhanced extraction function can handle appellate court cases.")
        print("Update extract_year_from_case_id() in the relevant scripts.")
        print("=" * 60)
    else:
        print(f"\n{'='*60}")
        print("ERROR: Some cases still don't have extractable years.")
        print("Manual intervention may be required.")
        print("=" * 60)


if __name__ == "__main__":
    main()
