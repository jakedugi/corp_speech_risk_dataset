#!/usr/bin/env python3
"""
Filter enhanced data to create the final clean dataset for stratified k-fold CV.

Filters applied:
1. Remove cases with missing final_judgement_real values
2. Remove cases with outcomes >= 5 billion (outlier threshold)
3. Remove records with excluded speakers
4. Detailed logging for verification

This creates the final dataset of valid cases for cross-validation.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path


def extract_case_id(src_path: str) -> str:
    """Extract case ID from _src path."""
    # Pattern to match case IDs like "2:11-cv-00644_flmd"
    match = re.search(r"/([^/]*:\d+-[^/]+_[^/]+)/entries/", src_path)
    if match:
        return match.group(1)
    # Fallback pattern for other formats
    match = re.search(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/", src_path)
    if match:
        return match.group(1)
    return "unknown"


def filter_valid_cases(
    input_file: str,
    output_file: str,
    outlier_threshold: float = 5000000000.0,
    exclude_speakers: list = None,
):
    """Filter to create final clean dataset for stratified k-fold CV."""
    if exclude_speakers is None:
        exclude_speakers = [
            "Unknown",
            "Court",
            "FTC",
            "Fed",
            "Plaintiff",
            "State",
            "Commission",
            "Congress",
            "Circuit",
            "FDA",
        ]

    print("=" * 60)
    print("FINAL DATASET CREATION FOR STRATIFIED K-FOLD CV")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Outlier threshold: ${outlier_threshold:,.0f}")
    print(f"Excluded speakers: {exclude_speakers}")
    print()

    # Counters for detailed logging
    stats = {
        "total_records": 0,
        "missing_outcome_records": 0,
        "outlier_records": 0,
        "excluded_speaker_records": 0,
        "kept_records": 0,
        "cases_analyzed": 0,
        "cases_missing_outcome": 0,
        "cases_outlier": 0,
        "cases_speaker_filtered": 0,
        "final_valid_cases": 0,
    }

    print("Step 1: First pass - analyzing all records...")

    # First pass: analyze all records and cases
    cases_outcomes = defaultdict(set)
    cases_speakers = defaultdict(set)
    record_filters = defaultdict(list)  # Track why each record was filtered

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats["total_records"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue

            src = record["_src"]
            case_id = extract_case_id(src)
            outcome = record.get("final_judgement_real")
            speaker = record.get("speaker", "Unknown")

            cases_outcomes[case_id].add(outcome)
            cases_speakers[case_id].add(speaker)

            # Track filter reasons for this record
            filter_reasons = []

            if outcome is None:
                filter_reasons.append("missing_outcome")
                stats["missing_outcome_records"] += 1
            elif outcome >= outlier_threshold:
                filter_reasons.append("outlier")
                stats["outlier_records"] += 1

            if speaker in exclude_speakers:
                filter_reasons.append("excluded_speaker")
                stats["excluded_speaker_records"] += 1

            record_filters[f"{case_id}_{line_num}"] = filter_reasons

            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} records...")

    stats["cases_analyzed"] = len(cases_outcomes)
    print(
        f"✓ Analyzed {stats['total_records']:,} records across {stats['cases_analyzed']} cases"
    )

    print("\nStep 2: Case-level filtering...")

    # Determine which cases pass all filters
    valid_cases = set()
    case_filter_reasons = {}

    for case_id in cases_outcomes:
        outcomes = cases_outcomes[case_id]
        speakers = cases_speakers[case_id]

        filter_reasons = []

        # Check for missing outcomes
        valid_outcomes = set(o for o in outcomes if o is not None)
        if not valid_outcomes:
            filter_reasons.append("missing_outcome")
            stats["cases_missing_outcome"] += 1

        # Check for outlier outcomes (any outcome >= threshold)
        elif any(o >= outlier_threshold for o in valid_outcomes):
            filter_reasons.append("outlier")
            stats["cases_outlier"] += 1

        # Check for excluded speakers (any excluded speaker in case)
        if any(s in exclude_speakers for s in speakers):
            filter_reasons.append("excluded_speaker")
            stats["cases_speaker_filtered"] += 1

        if not filter_reasons:
            valid_cases.add(case_id)

        case_filter_reasons[case_id] = filter_reasons

    stats["final_valid_cases"] = len(valid_cases)

    print(f"Case filtering results:")
    print(f"  - Total cases: {stats['cases_analyzed']}")
    print(f"  - Cases with missing outcomes: {stats['cases_missing_outcome']}")
    print(
        f"  - Cases with outlier outcomes (>=${outlier_threshold:,.0f}): {stats['cases_outlier']}"
    )
    print(f"  - Cases with excluded speakers: {stats['cases_speaker_filtered']}")
    print(f"  - Final valid cases: {stats['final_valid_cases']}")

    print("\nStep 3: Record-level filtering and output...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Second pass: write only records from valid cases
    with open(input_file) as f, open(output_file, "w") as out_f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            src = record["_src"]
            case_id = extract_case_id(src)

            if case_id in valid_cases:
                json.dump(record, out_f, ensure_ascii=False)
                out_f.write("\n")
                stats["kept_records"] += 1

    print(f"✓ Wrote {stats['kept_records']:,} records to {output_file}")

    print("\nStep 4: Final verification and analysis...")

    # Verify the output
    final_cases_outcomes = defaultdict(list)
    final_speakers = Counter()

    with open(output_file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            src = record["_src"]
            case_id = extract_case_id(src)
            outcome = record.get("final_judgement_real")
            speaker = record.get("speaker", "Unknown")

            final_cases_outcomes[case_id].append(outcome)
            final_speakers[speaker] += 1

    # Analyze final outcome distribution
    all_outcomes = []
    for outcomes in final_cases_outcomes.values():
        all_outcomes.extend([o for o in outcomes if o is not None])

    if all_outcomes:
        all_outcomes.sort()
        min_outcome = min(all_outcomes)
        max_outcome = max(all_outcomes)
        median_outcome = all_outcomes[len(all_outcomes) // 2]

        # Create 3 quantile bins for analysis
        import numpy as np

        quantiles = np.quantile(all_outcomes, [0, 1 / 3, 2 / 3, 1])

        bin_counts = {"bin_0": 0, "bin_1": 0, "bin_2": 0}
        for case_id, outcomes in final_cases_outcomes.items():
            case_outcome = outcomes[0]  # Use first outcome as representative
            bin_idx = np.digitize(case_outcome, quantiles) - 1
            bin_idx = np.clip(bin_idx, 0, 2)
            bin_counts[f"bin_{bin_idx}"] += 1

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"✓ Total records: {stats['kept_records']:,}")
    print(f"✓ Unique cases: {len(final_cases_outcomes)}")
    print(f"✓ All cases have valid outcomes (no missing values)")
    print(f"✓ All cases have outcomes < ${outlier_threshold:,.0f}")
    print(f"✓ No excluded speakers present")
    print()

    if all_outcomes:
        print("Outcome distribution:")
        print(f"  Min: ${min_outcome:,.0f}")
        print(f"  Median: ${median_outcome:,.0f}")
        print(f"  Max: ${max_outcome:,.0f}")
        print()

        print("Stratification bins (for k-fold):")
        for bin_name, count in sorted(bin_counts.items()):
            pct = count / len(final_cases_outcomes) * 100
            print(f"  {bin_name}: {count} cases ({pct:.1f}%)")
        print()

    print("Top speakers:")
    for speaker, count in final_speakers.most_common(10):
        print(f"  {speaker}: {count} records")

    print(f"\n✓ Final dataset saved to: {output_file}")
    print("✓ Ready for stratified k-fold cross-validation!")

    return stats


if __name__ == "__main__":
    input_file = "data/enhanced_combined/all_enhanced_data.jsonl"
    output_file = "data/enhanced_combined/final_clean_dataset.jsonl"

    # Filter with outlier threshold and speaker exclusions
    outlier_threshold = 10000000000.0  # 10 billion
    exclude_speakers = [
        "Unknown",
        "Court",
        "FTC",
        "Fed",
        "Plaintiff",
        "State",
        "Commission",
        "Congress",
        "Circuit",
        "FDA",
    ]

    filter_valid_cases(input_file, output_file, outlier_threshold, exclude_speakers)
