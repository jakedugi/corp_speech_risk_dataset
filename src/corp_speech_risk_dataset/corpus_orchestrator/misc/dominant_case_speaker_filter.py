#!/usr/bin/env python3
"""
Dominant-Case Speaker Filtering for Perfect Speaker Disjointness

This script implements a smart filtering strategy:
1. Keep all single-case speakers (no change)
2. For multi-case speakers, retain only quotes from their "dominant case"
   (the case where they have the most quotes)

This eliminates speaker leakage while maximizing data retention and preserving
the most representative context for each speaker.
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


def analyze_dominant_case_filtering(input_file: str) -> Dict:
    """Analyze the impact of dominant-case speaker filtering."""
    print("=" * 80)
    print("DOMINANT-CASE SPEAKER FILTERING ANALYSIS")
    print("=" * 80)

    # Data structures
    speaker_case_counts = defaultdict(
        lambda: defaultdict(int)
    )  # speaker -> case -> count
    all_records = []
    case_outcomes = {}
    case_records = defaultdict(list)

    print("Loading data and analyzing speaker-case distributions...")

    with open(input_file) as f:
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

            speaker_case_counts[speaker][case_id] += 1
            all_records.append(record)
            case_outcomes[case_id] = outcome
            case_records[case_id].append(record)

            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} records...")

    total_records = len(all_records)
    total_cases = len(case_records)

    print(f"✓ Loaded {total_records:,} records from {total_cases:,} cases")

    # Step 1: Identify dominant cases for each speaker
    print("\n" + "=" * 60)
    print("1. IDENTIFYING DOMINANT CASES")
    print("=" * 60)

    speaker_dominant_case = {}  # speaker -> dominant_case_id
    single_case_speakers = 0
    multi_case_speakers = 0

    for speaker, case_counts in speaker_case_counts.items():
        if len(case_counts) == 1:
            # Single-case speaker - keep as is
            single_case_speakers += 1
            speaker_dominant_case[speaker] = list(case_counts.keys())[0]
        else:
            # Multi-case speaker - find dominant case
            multi_case_speakers += 1
            dominant_case = max(case_counts.items(), key=lambda x: x[1])[0]
            speaker_dominant_case[speaker] = dominant_case

    print(f"Single-case speakers: {single_case_speakers:,}")
    print(f"Multi-case speakers: {multi_case_speakers:,}")
    print(f"Total speakers: {len(speaker_case_counts):,}")

    # Analyze multi-case speakers in detail
    multi_case_analysis = []
    for speaker, case_counts in speaker_case_counts.items():
        if len(case_counts) > 1:
            dominant_case = speaker_dominant_case[speaker]
            dominant_count = case_counts[dominant_case]
            total_count = sum(case_counts.values())
            retention_pct = dominant_count / total_count * 100

            multi_case_analysis.append(
                {
                    "speaker": speaker,
                    "total_records": total_count,
                    "dominant_case": dominant_case,
                    "dominant_records": dominant_count,
                    "retention_pct": retention_pct,
                    "cases_involved": len(case_counts),
                    "records_dropped": total_count - dominant_count,
                }
            )

    # Sort by total records (impact)
    multi_case_analysis.sort(key=lambda x: x["total_records"], reverse=True)

    print(f"\nTop 10 multi-case speakers and their dominant cases:")
    for i, analysis in enumerate(multi_case_analysis[:10]):
        print(
            f"  {i+1:2}. {analysis['speaker']:<15}: "
            f"{analysis['dominant_records']:4}/{analysis['total_records']:4} records "
            f"({analysis['retention_pct']:5.1f}%) in {analysis['dominant_case']}"
        )

    # Step 2: Calculate filtering impact
    print("\n" + "=" * 60)
    print("2. FILTERING IMPACT CALCULATION")
    print("=" * 60)

    records_to_keep = 0
    records_to_drop = 0
    cases_affected = set()
    cases_emptied = set()

    for record in all_records:
        case_id = extract_case_id(record["_src"])
        speaker = record.get("speaker", "Unknown")
        dominant_case = speaker_dominant_case[speaker]

        if case_id == dominant_case:
            records_to_keep += 1
        else:
            records_to_drop += 1
            cases_affected.add(case_id)

    # Check which cases become empty
    filtered_case_records = defaultdict(int)
    for record in all_records:
        case_id = extract_case_id(record["_src"])
        speaker = record.get("speaker", "Unknown")
        dominant_case = speaker_dominant_case[speaker]

        if case_id == dominant_case:
            filtered_case_records[case_id] += 1

    cases_emptied = set(case_records.keys()) - set(filtered_case_records.keys())
    final_cases = len(case_records) - len(cases_emptied)

    print(
        f"Records to keep: {records_to_keep:,} ({records_to_keep/total_records*100:.1f}%)"
    )
    print(
        f"Records to drop: {records_to_drop:,} ({records_to_drop/total_records*100:.1f}%)"
    )
    print(f"Cases affected by filtering: {len(cases_affected):,}")
    print(f"Cases completely emptied: {len(cases_emptied):,}")
    print(
        f"Final cases remaining: {final_cases:,} ({final_cases/total_cases*100:.1f}%)"
    )

    # Step 3: Analyze outcome distribution impact
    print("\n" + "=" * 60)
    print("3. OUTCOME DISTRIBUTION IMPACT")
    print("=" * 60)

    # Original valid cases
    valid_cases_original = [
        case_id for case_id, outcome in case_outcomes.items() if outcome is not None
    ]

    # Remaining valid cases after filtering
    valid_cases_remaining = [
        case_id for case_id in valid_cases_original if case_id in filtered_case_records
    ]

    print(f"Valid cases - Original: {len(valid_cases_original):,}")
    print(f"Valid cases - After filtering: {len(valid_cases_remaining):,}")
    print(
        f"Valid cases lost: {len(valid_cases_original) - len(valid_cases_remaining):,}"
    )

    if len(valid_cases_remaining) >= 3:
        # Calculate outcome distributions
        original_outcomes = [case_outcomes[case_id] for case_id in valid_cases_original]
        remaining_outcomes = [
            case_outcomes[case_id] for case_id in valid_cases_remaining
        ]

        original_outcomes.sort()
        remaining_outcomes.sort()

        # Original quantiles
        quantiles_original = np.quantile(original_outcomes, [0, 1 / 3, 2 / 3, 1])
        quantiles_remaining = np.quantile(remaining_outcomes, [0, 1 / 3, 2 / 3, 1])

        print(
            f"\nOriginal quantile boundaries: {[f'${x:,.0f}' for x in quantiles_original]}"
        )
        print(
            f"Filtered quantile boundaries: {[f'${x:,.0f}' for x in quantiles_remaining]}"
        )

        # Calculate bin distributions
        def assign_bins(case_ids, quantiles):
            bins = []
            for case_id in case_ids:
                outcome = case_outcomes[case_id]
                bin_idx = np.digitize(outcome, quantiles) - 1
                bin_idx = np.clip(bin_idx, 0, 2)
                bins.append(f"bin_{bin_idx}")
            return bins

        original_bins = assign_bins(valid_cases_original, quantiles_original)
        remaining_bins = assign_bins(valid_cases_remaining, quantiles_remaining)

        original_dist = Counter(original_bins)
        remaining_dist = Counter(remaining_bins)

        print(f"\nOriginal distribution:")
        for bin_name in ["bin_0", "bin_1", "bin_2"]:
            count = original_dist.get(bin_name, 0)
            pct = count / len(original_bins) * 100 if original_bins else 0
            print(f"  {bin_name}: {count:3} cases ({pct:5.1f}%)")

        print(f"Filtered distribution:")
        for bin_name in ["bin_0", "bin_1", "bin_2"]:
            count = remaining_dist.get(bin_name, 0)
            pct = count / len(remaining_bins) * 100 if remaining_bins else 0
            print(f"  {bin_name}: {count:3} cases ({pct:5.1f}%)")

        # Balance quality
        remaining_counts = [remaining_dist.get(f"bin_{i}", 0) for i in range(3)]
        if remaining_counts and min(remaining_counts) > 0:
            balance_ratio = max(remaining_counts) / min(remaining_counts)
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
        else:
            print(f"\nBalance quality: POOR (some bins empty)")

    # Step 4: Verify speaker disjointness
    print("\n" + "=" * 60)
    print("4. SPEAKER DISJOINTNESS VERIFICATION")
    print("=" * 60)

    # Check if any speaker appears in multiple cases after filtering
    filtered_speaker_cases = defaultdict(set)

    for record in all_records:
        case_id = extract_case_id(record["_src"])
        speaker = record.get("speaker", "Unknown")
        dominant_case = speaker_dominant_case[speaker]

        if case_id == dominant_case:  # Only count records we're keeping
            filtered_speaker_cases[speaker].add(case_id)

    multi_case_speakers_after = [
        speaker for speaker, cases in filtered_speaker_cases.items() if len(cases) > 1
    ]

    print(f"Speakers in filtered data: {len(filtered_speaker_cases):,}")
    print(f"Multi-case speakers after filtering: {len(multi_case_speakers_after):,}")

    if len(multi_case_speakers_after) == 0:
        print("✅ PERFECT SPEAKER DISJOINTNESS ACHIEVED!")
    else:
        print(f"❌ Still have {len(multi_case_speakers_after)} multi-case speakers:")
        for speaker in multi_case_speakers_after[:5]:
            cases = list(filtered_speaker_cases[speaker])
            print(f"  {speaker}: {cases}")

    # Step 5: K-fold feasibility
    print("\n" + "=" * 60)
    print("5. K-FOLD FEASIBILITY ASSESSMENT")
    print("=" * 60)

    print(f"Filtered dataset: {records_to_keep:,} records, {final_cases:,} cases")

    if final_cases >= 5:
        cases_per_fold = final_cases // 5
        extra_cases = final_cases % 5
        print(
            f"✅ Sufficient for 5-fold CV: {cases_per_fold} cases per fold (+{extra_cases} extra)"
        )

        if len(valid_cases_remaining) >= 15:
            print("✅ Sufficient for stratified 5-fold CV")
        else:
            print("⚠️  Limited stratification options")
    else:
        print("❌ Insufficient cases for 5-fold CV")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 80)

    data_retention = records_to_keep / total_records * 100
    case_retention = final_cases / total_cases * 100

    if (
        len(multi_case_speakers_after) == 0
        and data_retention >= 70
        and case_retention >= 85
        and final_cases >= 15
    ):
        recommendation = "✅ HIGHLY RECOMMENDED"
        print("✅ DOMINANT-CASE FILTERING IS HIGHLY RECOMMENDED")
    elif len(multi_case_speakers_after) <= 2 and data_retention >= 60:
        recommendation = "⚠️  CONDITIONALLY RECOMMENDED"
        print("⚠️  DOMINANT-CASE FILTERING IS CONDITIONALLY RECOMMENDED")
    else:
        recommendation = "❌ NOT RECOMMENDED"
        print("❌ DOMINANT-CASE FILTERING IS NOT RECOMMENDED")

    print(
        f"   Data retention: {data_retention:.1f}% records, {case_retention:.1f}% cases"
    )
    print(f"   Speaker disjointness: {len(multi_case_speakers_after) == 0}")
    print(f"   K-fold feasibility: {final_cases >= 5}")

    return {
        "strategy": "dominant_case_filtering",
        "total_records_original": total_records,
        "total_cases_original": total_cases,
        "records_retained": records_to_keep,
        "cases_retained": final_cases,
        "data_retention_pct": data_retention,
        "case_retention_pct": case_retention,
        "speaker_disjointness_achieved": len(multi_case_speakers_after) == 0,
        "remaining_multi_case_speakers": len(multi_case_speakers_after),
        "kfold_feasible": final_cases >= 5,
        "stratified_kfold_feasible": len(valid_cases_remaining) >= 15,
        "recommendation": recommendation,
        "speaker_dominant_case": speaker_dominant_case,
        "multi_case_analysis": multi_case_analysis[:20],  # Top 20 for storage
    }


def create_dominant_case_filtered_dataset(
    input_file: str, output_file: str, speaker_dominant_case: Dict[str, str]
) -> None:
    """Create filtered dataset using dominant-case strategy."""
    print(f"\nCreating dominant-case filtered dataset: {output_file}")

    records_written = 0
    records_skipped = 0
    cases_written = set()

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                case_id = extract_case_id(record["_src"])
                speaker = record.get("speaker", "Unknown")

                # Check if this record should be kept
                dominant_case = speaker_dominant_case.get(speaker)

                if case_id == dominant_case:
                    f_out.write(line)
                    records_written += 1
                    cases_written.add(case_id)
                else:
                    records_skipped += 1

            except json.JSONDecodeError:
                continue

            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} records...")

    print(f"✓ Filtered dataset created:")
    print(f"  Records written: {records_written:,}")
    print(f"  Records skipped: {records_skipped:,}")
    print(f"  Cases written: {len(cases_written):,}")
    print(
        f"  Data retention: {records_written/(records_written + records_skipped)*100:.1f}%"
    )


def main():
    """Main execution function."""
    input_file = "data/enhanced_combined/final_clean_dataset.jsonl"

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return

    # Run analysis
    print("🔍 Running dominant-case speaker filtering analysis...")
    results = analyze_dominant_case_filtering(input_file)

    # Save results
    results_file = "data/dominant_case_filtering_analysis.json"
    with open(results_file, "w") as f:
        # Convert speaker_dominant_case for JSON serialization
        results_copy = results.copy()
        json.dump(results_copy, f, indent=2, default=str)

    print(f"\n📄 Analysis results saved to: {results_file}")

    # Create filtered dataset if recommended
    if "RECOMMENDED" in results["recommendation"]:
        output_file = (
            "data/enhanced_combined/final_clean_dataset_dominant_case_filtered.jsonl"
        )
        create_dominant_case_filtered_dataset(
            input_file, output_file, results["speaker_dominant_case"]
        )
        print(f"\n🎉 SUCCESS! Filtered dataset ready for speaker-disjoint k-fold CV!")
        print(f"   Next step: Run k-fold splits on {output_file}")
    else:
        print(f"\n⚠️  Filtering not recommended based on current analysis")


if __name__ == "__main__":
    main()
