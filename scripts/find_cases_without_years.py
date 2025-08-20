#!/usr/bin/env python3
"""
Find cases that don't have extractable years for manual input.
"""

import json
import re
from collections import defaultdict


def extract_case_id(src_path: str) -> str:
    """Extract case ID from _src path."""
    match = re.search(r"/([^/]*:\d+-[^/]+_[^/]+)/entries/", src_path)
    if match:
        return match.group(1)
    match = re.search(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/", src_path)
    if match:
        return match.group(1)
    return "unknown"


def extract_date_from_case_id(case_id: str) -> int:
    """Extract year from case ID if possible."""
    match = re.search(r":(\d{2,4})-", case_id)
    if match:
        year = int(match.group(1))
        # Handle 2-digit years
        if year < 50:
            return 2000 + year
        elif year < 100:
            return 1900 + year
        return year
    return None


def find_cases_without_years():
    """Find all cases that don't have extractable years."""

    print("🔍 FINDING CASES WITHOUT EXTRACTABLE YEARS")
    print("=" * 60)

    cases_without_years = {}
    cases_with_years = {}
    all_cases = set()

    # Process the dataset
    print("Processing dataset...")
    with open("data/enhanced_combined/final_clean_dataset_leakage_safe.jsonl") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 5000 == 0:
                print(f"  Processed {line_num:,} records...")

            try:
                record = json.loads(line)
            except:
                continue

            # Extract case ID from source path
            src_path = record.get("_src") or record.get("_metadata_src_path", "")
            if src_path:
                case_id = extract_case_id(src_path)
                all_cases.add(case_id)

                # Try to extract year
                year = extract_date_from_case_id(case_id)

                if year is None:
                    # Store info about cases without years
                    if case_id not in cases_without_years:
                        cases_without_years[case_id] = {
                            "case_id": case_id,
                            "src_path": src_path,
                            "sample_text": record.get("text", "")[:200],
                            "record_count": 0,
                        }
                    cases_without_years[case_id]["record_count"] += 1
                else:
                    if case_id not in cases_with_years:
                        cases_with_years[case_id] = year

    print(f"✅ Processing complete!")
    print(f"📊 Total unique cases: {len(all_cases)}")
    print(f"📊 Cases with extractable years: {len(cases_with_years)}")
    print(f"📊 Cases WITHOUT extractable years: {len(cases_without_years)}")
    print()

    if cases_without_years:
        print("🚨 CASES WITHOUT EXTRACTABLE YEARS:")
        print("-" * 80)
        print(f"{'Case ID':<30} {'Records':<10} {'Sample Text':<50}")
        print("-" * 80)

        # Sort by record count (most records first)
        sorted_cases = sorted(
            cases_without_years.items(),
            key=lambda x: x[1]["record_count"],
            reverse=True,
        )

        for case_id, info in sorted_cases:
            record_count = info["record_count"]
            sample_text = info["sample_text"].replace("\n", " ").strip()
            sample_text = (
                sample_text[:47] + "..." if len(sample_text) > 50 else sample_text
            )

            print(f"{case_id:<30} {record_count:<10} {sample_text:<50}")

        print("-" * 80)
        print()

        # Analyze patterns of cases without years
        print("🔍 ANALYSIS OF CASES WITHOUT YEARS:")
        print("-" * 50)

        # Group by pattern
        patterns = defaultdict(list)
        for case_id in cases_without_years.keys():
            if case_id == "unknown":
                patterns["unknown"].append(case_id)
            elif ":" not in case_id:
                patterns["no_colon"].append(case_id)
            elif not re.search(r"\d+", case_id):
                patterns["no_digits"].append(case_id)
            else:
                patterns["other"].append(case_id)

        for pattern, case_list in patterns.items():
            print(f"{pattern}: {len(case_list)} cases")
            for case_id in case_list[:3]:
                print(f"  Example: {case_id}")
            if len(case_list) > 3:
                print(f"  ... and {len(case_list) - 3} more")
            print()

        # Create manual input template
        print("📝 MANUAL INPUT TEMPLATE:")
        print("-" * 40)
        print("# Copy this template and fill in the years manually")
        print("case_year_mapping = {")

        for case_id, info in sorted_cases:
            src_path = info["src_path"]
            print(
                f"    '{case_id}': None,  # {record_count} records - {src_path[:60]}..."
            )

        print("}")
        print()

        # Save to file for easy editing
        output_file = "cases_without_years_template.py"
        with open(output_file, "w") as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""\n')
            f.write("Manual year mapping for cases without extractable years.\n")
            f.write("Fill in the None values with the correct years.\n")
            f.write('"""\n\n')
            f.write("case_year_mapping = {\n")

            for case_id, info in sorted_cases:
                record_count = info["record_count"]
                src_path = info["src_path"]
                f.write(
                    f"    '{case_id}': None,  # {record_count} records - {src_path}\n"
                )

            f.write("}\n")

        print(f"💾 Template saved to: {output_file}")
        print("   Edit this file to add the correct years for each case.")

    else:
        print("✅ All cases have extractable years!")

    return cases_without_years, cases_with_years


if __name__ == "__main__":
    cases_without_years, cases_with_years = find_cases_without_years()
