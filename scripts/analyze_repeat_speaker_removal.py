#!/usr/bin/env python3
"""
Analyze the Impact of Removing Repeat Speakers

This script investigates dropping all speakers who appear in multiple cases
to achieve perfect speaker disjointness in k-fold cross-validation.

Analysis includes:
1. Identify speakers appearing in multiple cases
2. Calculate data loss from removing repeat speakers
3. Analyze impact on outcome distribution and stratification
4. Test feasibility of speaker-disjoint k-fold splits after removal
"""

import json
import re
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def extract_case_id(src_path: str) -> str:
    """Extract case ID from _src path."""
    match = re.search(r"/([^/]*:\d+-[^/]+_[^/]+)/entries/", src_path)
    if match:
        return match.group(1)
    match = re.search(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/", src_path)
    if match:
        return match.group(1)
    return "unknown"


def analyze_repeat_speaker_impact(file_path: str) -> Dict:
    """Analyze impact of removing speakers who appear in multiple cases."""
    print("=" * 80)
    print("REPEAT SPEAKER REMOVAL ANALYSIS")
    print("=" * 80)

    # Data structures
    speaker_to_cases = defaultdict(set)  # speaker -> set of cases
    speaker_to_records = defaultdict(list)  # speaker -> list of records
    case_to_speakers = defaultdict(set)  # case -> set of speakers
    case_outcomes = {}  # case -> outcome
    case_records = defaultdict(list)  # case -> list of records

    print("Loading data...")
    total_records = 0

    with open(file_path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            case_id = extract_case_id(record["_src"])
            speaker = record.get("speaker", "Unknown")
            outcome = record.get("final_judgement_real")

            speaker_to_cases[speaker].add(case_id)
            speaker_to_records[speaker].append(record)
            case_to_speakers[case_id].add(speaker)
            case_outcomes[case_id] = outcome
            case_records[case_id].append(record)

            total_records += 1

            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} records...")

    print(f"✓ Loaded {total_records:,} records from {len(case_records)} cases")

    # Step 1: Identify repeat speakers
    print("\n" + "=" * 60)
    print("1. IDENTIFYING REPEAT SPEAKERS")
    print("=" * 60)

    single_case_speakers = []
    repeat_speakers = []

    for speaker, cases in speaker_to_cases.items():
        if len(cases) == 1:
            single_case_speakers.append(speaker)
        else:
            repeat_speakers.append(
                {
                    "speaker": speaker,
                    "cases": list(cases),
                    "num_cases": len(cases),
                    "num_records": len(speaker_to_records[speaker]),
                }
            )

    # Sort repeat speakers by impact (number of records)
    repeat_speakers.sort(key=lambda x: x["num_records"], reverse=True)

    print(f"Total speakers: {len(speaker_to_cases):,}")
    print(
        f"Single-case speakers: {len(single_case_speakers):,} ({len(single_case_speakers)/len(speaker_to_cases)*100:.1f}%)"
    )
    print(
        f"Multi-case speakers: {len(repeat_speakers):,} ({len(repeat_speakers)/len(speaker_to_cases)*100:.1f}%)"
    )

    print(f"\nTop 10 repeat speakers by record count:")
    for i, rs in enumerate(repeat_speakers[:10]):
        print(
            f"  {i+1:2}. {rs['speaker']:<20}: {rs['num_records']:5} records across {rs['num_cases']} cases"
        )

    # Step 2: Calculate data loss from removal
    print("\n" + "=" * 60)
    print("2. DATA LOSS CALCULATION")
    print("=" * 60)

    repeat_speaker_names = [rs["speaker"] for rs in repeat_speakers]
    records_to_remove = sum(
        len(speaker_to_records[speaker]) for speaker in repeat_speaker_names
    )
    cases_to_remove = set()

    # Find cases that would lose ALL their records
    for case_id, speakers in case_to_speakers.items():
        remaining_speakers = speakers - set(repeat_speaker_names)
        if not remaining_speakers:  # Case would have no speakers left
            cases_to_remove.add(case_id)

    # Calculate records in cases that would be completely removed
    records_in_removed_cases = sum(
        len(case_records[case_id]) for case_id in cases_to_remove
    )

    # Calculate remaining dataset
    remaining_records = total_records - records_to_remove
    remaining_cases = len(case_records) - len(cases_to_remove)

    print(
        f"Records from repeat speakers: {records_to_remove:,} ({records_to_remove/total_records*100:.1f}%)"
    )
    print(
        f"Cases losing ALL speakers: {len(cases_to_remove):,} ({len(cases_to_remove)/len(case_records)*100:.1f}%)"
    )
    print(f"Records in completely removed cases: {records_in_removed_cases:,}")
    print(f"\nAfter removal:")
    print(
        f"  Remaining records: {remaining_records:,} ({remaining_records/total_records*100:.1f}%)"
    )
    print(
        f"  Remaining cases: {remaining_cases:,} ({remaining_cases/len(case_records)*100:.1f}%)"
    )

    # Step 3: Analyze impact on outcome distribution
    print("\n" + "=" * 60)
    print("3. OUTCOME DISTRIBUTION IMPACT")
    print("=" * 60)

    # Original outcome distribution
    valid_cases_original = [
        case_id for case_id, outcome in case_outcomes.items() if outcome is not None
    ]
    valid_outcomes_original = [
        case_outcomes[case_id] for case_id in valid_cases_original
    ]

    # Remaining cases after removal
    valid_cases_remaining = [
        case_id for case_id in valid_cases_original if case_id not in cases_to_remove
    ]
    valid_outcomes_remaining = [
        case_outcomes[case_id] for case_id in valid_cases_remaining
    ]

    print(f"Valid cases - Original: {len(valid_cases_original):,}")
    print(f"Valid cases - After removal: {len(valid_cases_remaining):,}")
    print(
        f"Valid cases lost: {len(valid_cases_original) - len(valid_cases_remaining):,}"
    )

    if len(valid_outcomes_remaining) >= 3:
        # Calculate original bins
        valid_outcomes_original.sort()
        quantiles_original = np.quantile(valid_outcomes_original, [0, 1 / 3, 2 / 3, 1])

        # Calculate remaining bins
        valid_outcomes_remaining.sort()
        quantiles_remaining = np.quantile(
            valid_outcomes_remaining, [0, 1 / 3, 2 / 3, 1]
        )

        print(
            f"\nOriginal quantile boundaries: {[f'${x:,.0f}' for x in quantiles_original]}"
        )
        print(
            f"Remaining quantile boundaries: {[f'${x:,.0f}' for x in quantiles_remaining]}"
        )

        # Assign bins and compare distributions
        def assign_bins(outcomes, quantiles):
            bins = []
            for outcome in outcomes:
                bin_idx = np.digitize(outcome, quantiles) - 1
                bin_idx = np.clip(bin_idx, 0, 2)
                bins.append(f"bin_{bin_idx}")
            return bins

        original_bins = assign_bins(
            [case_outcomes[case_id] for case_id in valid_cases_original],
            quantiles_original,
        )
        remaining_bins = assign_bins(
            [case_outcomes[case_id] for case_id in valid_cases_remaining],
            quantiles_remaining,
        )

        original_dist = Counter(original_bins)
        remaining_dist = Counter(remaining_bins)

        print(f"\nOriginal distribution:")
        for bin_name in ["bin_0", "bin_1", "bin_2"]:
            count = original_dist.get(bin_name, 0)
            pct = count / len(original_bins) * 100 if original_bins else 0
            print(f"  {bin_name}: {count:3} cases ({pct:5.1f}%)")

        print(f"Remaining distribution:")
        for bin_name in ["bin_0", "bin_1", "bin_2"]:
            count = remaining_dist.get(bin_name, 0)
            pct = count / len(remaining_bins) * 100 if remaining_bins else 0
            print(f"  {bin_name}: {count:3} cases ({pct:5.1f}%)")

        # Check if still balanced
        remaining_counts = [remaining_dist.get(f"bin_{i}", 0) for i in range(3)]
        if remaining_counts:
            balance_ratio = (
                max(remaining_counts) / min(remaining_counts)
                if min(remaining_counts) > 0
                else float("inf")
            )
            balance_quality = (
                "EXCELLENT"
                if balance_ratio < 1.2
                else (
                    "GOOD"
                    if balance_ratio < 1.5
                    else "MODERATE" if balance_ratio < 2.0 else "POOR"
                )
            )
            print(f"\nBalance quality: {balance_quality} (ratio: {balance_ratio:.2f})")

    # Step 4: Test speaker disjointness after removal
    print("\n" + "=" * 60)
    print("4. SPEAKER DISJOINTNESS TEST")
    print("=" * 60)

    # Create filtered dataset
    filtered_records = []
    filtered_case_data = defaultdict(list)

    for case_id, records in case_records.items():
        if case_id not in cases_to_remove:
            case_filtered_records = []
            for record in records:
                speaker = record.get("speaker", "Unknown")
                if speaker not in repeat_speaker_names:
                    case_filtered_records.append(record)
                    filtered_records.append(record)

            if case_filtered_records:  # Only keep cases with remaining records
                filtered_case_data[case_id] = case_filtered_records

    # Check speaker disjointness in filtered data
    filtered_speaker_to_cases = defaultdict(set)
    for case_id, records in filtered_case_data.items():
        for record in records:
            speaker = record.get("speaker", "Unknown")
            filtered_speaker_to_cases[speaker].add(case_id)

    multi_case_speakers_after = [
        speaker
        for speaker, cases in filtered_speaker_to_cases.items()
        if len(cases) > 1
    ]

    print(f"Speakers in filtered data: {len(filtered_speaker_to_cases):,}")
    print(f"Multi-case speakers after removal: {len(multi_case_speakers_after):,}")

    if len(multi_case_speakers_after) == 0:
        print("✅ PERFECT SPEAKER DISJOINTNESS ACHIEVED!")
    else:
        print(f"⚠️  Still have {len(multi_case_speakers_after)} multi-case speakers")
        print("Top remaining multi-case speakers:")
        for speaker in multi_case_speakers_after[:5]:
            cases = list(filtered_speaker_to_cases[speaker])
            print(f"  {speaker}: {len(cases)} cases")

    # Step 5: Test k-fold feasibility
    print("\n" + "=" * 60)
    print("5. K-FOLD FEASIBILITY TEST")
    print("=" * 60)

    print(
        f"Filtered dataset: {len(filtered_records):,} records, {len(filtered_case_data):,} cases"
    )

    if len(filtered_case_data) >= 5:
        print("✅ Sufficient cases for 5-fold CV")

        # Test simple k-fold balance
        cases_per_fold = len(filtered_case_data) // 5
        remaining_cases = len(filtered_case_data) % 5

        print(f"Cases per fold: {cases_per_fold} (with {remaining_cases} extra)")

        # Test stratification quality with filtered data
        if len(valid_cases_remaining) >= 15:  # Need at least 3 per bin for 5-fold
            print("✅ Sufficient cases for stratified 5-fold CV")
        else:
            print("⚠️  May not have enough cases for robust stratified CV")
    else:
        print("❌ Insufficient cases for 5-fold CV")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    data_retention = remaining_records / total_records * 100
    case_retention = len(filtered_case_data) / len(case_records) * 100

    if (
        data_retention >= 80
        and case_retention >= 80
        and len(multi_case_speakers_after) == 0
    ):
        recommendation = "✅ RECOMMENDED: Remove repeat speakers"
        print("✅ REMOVING REPEAT SPEAKERS IS HIGHLY RECOMMENDED")
        print(f"   - Achieves perfect speaker disjointness")
        print(
            f"   - Retains {data_retention:.1f}% of records and {case_retention:.1f}% of cases"
        )
        print(f"   - Maintains feasible k-fold cross-validation")
    elif data_retention >= 60 and len(multi_case_speakers_after) <= 5:
        recommendation = "⚠️  CONDITIONAL: Consider removal with caveats"
        print("⚠️  REMOVING REPEAT SPEAKERS IS CONDITIONALLY RECOMMENDED")
        print(f"   - Significant improvement in speaker disjointness")
        print(
            f"   - Moderate data loss: {100-data_retention:.1f}% records, {100-case_retention:.1f}% cases"
        )
        print(f"   - May need methodology adjustments")
    else:
        recommendation = "❌ NOT RECOMMENDED: Too much data loss"
        print("❌ REMOVING REPEAT SPEAKERS IS NOT RECOMMENDED")
        print(
            f"   - Excessive data loss: {100-data_retention:.1f}% records, {100-case_retention:.1f}% cases"
        )
        print(f"   - Alternative approaches needed")

    return {
        "total_speakers": len(speaker_to_cases),
        "repeat_speakers": len(repeat_speakers),
        "repeat_speaker_pct": len(repeat_speakers) / len(speaker_to_cases) * 100,
        "records_to_remove": records_to_remove,
        "records_retention_pct": data_retention,
        "cases_to_remove": len(cases_to_remove),
        "cases_retention_pct": case_retention,
        "speaker_disjointness_achieved": len(multi_case_speakers_after) == 0,
        "remaining_multi_case_speakers": len(multi_case_speakers_after),
        "kfold_feasible": len(filtered_case_data) >= 5,
        "stratified_kfold_feasible": len(valid_cases_remaining) >= 15,
        "recommendation": recommendation,
        "filtered_cases": len(filtered_case_data),
        "filtered_records": len(filtered_records),
    }


def create_filtered_dataset(input_file: str, output_file: str) -> None:
    """Create filtered dataset with repeat speakers removed."""
    print(f"\nCreating filtered dataset: {output_file}")

    # First pass: identify repeat speakers
    speaker_to_cases = defaultdict(set)

    with open(input_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                case_id = extract_case_id(record["_src"])
                speaker = record.get("speaker", "Unknown")
                speaker_to_cases[speaker].add(case_id)
            except json.JSONDecodeError:
                continue

    # Identify repeat speakers
    repeat_speakers = {
        speaker for speaker, cases in speaker_to_cases.items() if len(cases) > 1
    }

    print(f"Removing {len(repeat_speakers):,} repeat speakers...")

    # Second pass: filter records
    records_written = 0
    cases_written = set()

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                speaker = record.get("speaker", "Unknown")

                if speaker not in repeat_speakers:
                    f_out.write(line)
                    records_written += 1
                    cases_written.add(extract_case_id(record["_src"]))
            except json.JSONDecodeError:
                continue

    print(
        f"✓ Created filtered dataset: {records_written:,} records, {len(cases_written):,} cases"
    )


def main():
    """Main analysis function."""
    input_file = "data/enhanced_combined/final_clean_dataset.jsonl"

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return

    # Run analysis
    results = analyze_repeat_speaker_impact(input_file)

    # Save results
    results_file = "data/repeat_speaker_removal_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Analysis results saved to: {results_file}")

    # Offer to create filtered dataset
    if (
        results["speaker_disjointness_achieved"]
        and results["records_retention_pct"] >= 70
    ):
        print(f"\n🤔 Create filtered dataset without repeat speakers?")
        response = input("   Create filtered dataset? (y/n): ").lower().strip()

        if response == "y":
            output_file = (
                "data/enhanced_combined/final_clean_dataset_no_repeat_speakers.jsonl"
            )
            create_filtered_dataset(input_file, output_file)
            print(f"✅ Filtered dataset ready for speaker-disjoint k-fold CV!")


if __name__ == "__main__":
    main()
