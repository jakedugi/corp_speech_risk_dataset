#!/usr/bin/env python3
"""
Update the comprehensive leakage audit to use manual year mappings
for cases that don't have extractable years.
"""

import json
import re
from cases_without_years_template import case_year_mapping


def extract_case_id(src_path: str) -> str:
    """Extract case ID from _src path."""
    match = re.search(r"/([^/]*:\d+-[^/]+_[^/]+)/entries/", src_path)
    if match:
        return match.group(1)
    match = re.search(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/", src_path)
    if match:
        return match.group(1)
    return "unknown"


def extract_date_from_case_id_improved(case_id: str) -> int:
    """Extract year from case ID with manual mapping for edge cases."""

    # First check manual mapping
    if case_id in case_year_mapping:
        return case_year_mapping[case_id]

    # Standard extraction logic
    match = re.search(r":(\d{2,4})-", case_id)
    if match:
        year = int(match.group(1))
        # Handle 2-digit years
        if year < 50:
            return 2000 + year
        elif year < 100:
            return 1900 + year
        return year

    # Fallback for cases without colon (like bankruptcy cases)
    match = re.search(r"^(\d{2,4})-", case_id)
    if match:
        year = int(match.group(1))
        if year < 50:
            return 2000 + year
        elif year < 100:
            return 1900 + year
        return year

    return None


def test_improved_extraction():
    """Test the improved year extraction."""

    print("🧪 TESTING IMPROVED YEAR EXTRACTION")
    print("=" * 50)

    # Test cases that were problematic
    test_cases = [
        "09-11435_nysb",
        "24-60040_ca5",
        "24-10951_ca5",
        "17-00276_paeb",
        "15-10116_ksb",
        "1:19-cv-02184_dcd",  # Standard case
        "3:25-cv-00025_ctd",  # Future case
    ]

    print("Test Results:")
    print("-" * 40)
    for case_id in test_cases:
        year = extract_date_from_case_id_improved(case_id)
        status = "✅" if year is not None else "❌"
        source = "Manual" if case_id in case_year_mapping else "Extracted"
        print(f"{case_id:<20} -> {year} {status} ({source})")

    print("\n📊 Coverage Analysis:")
    print("-" * 30)

    # Count how many cases we can now extract years for
    cases_with_years = 0
    cases_without_years = 0
    all_cases = set()

    with open("data/enhanced_combined/final_clean_dataset_leakage_safe.jsonl") as f:
        for line in f:
            record = json.loads(line)
            src_path = record.get("_src") or record.get("_metadata_src_path", "")
            if src_path:
                case_id = extract_case_id(src_path)
                all_cases.add(case_id)

    for case_id in all_cases:
        year = extract_date_from_case_id_improved(case_id)
        if year is not None:
            cases_with_years += 1
        else:
            cases_without_years += 1
            print(f"Still missing: {case_id}")

    print(f"Total cases: {len(all_cases)}")
    print(f"With years: {cases_with_years}")
    print(f"Without years: {cases_without_years}")
    print(f"Coverage: {cases_with_years/len(all_cases)*100:.1f}%")


if __name__ == "__main__":
    test_improved_extraction()
