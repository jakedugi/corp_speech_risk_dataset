#!/usr/bin/env python3
"""
Analyze enhanced data to confirm case counts and outcome distributions.
"""

import json
import re
from collections import defaultdict, Counter


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


def analyze_enhanced_data(file_path: str):
    """Analyze the enhanced data file."""
    print("Analyzing enhanced data...")

    cases_outcomes = defaultdict(set)
    total_records = 0

    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue

            total_records += 1
            record = json.loads(line)

            # Extract case ID
            src = record["_src"]
            case_id = extract_case_id(src)

            # Extract outcome
            outcome = record.get("final_judgement_real")
            cases_outcomes[case_id].add(outcome)

    # Analyze by case
    valid_cases = 0
    missing_cases = 0
    case_outcomes_summary = []

    for case_id, outcomes in cases_outcomes.items():
        # Remove None values to check for valid outcomes
        valid_outcomes = set(o for o in outcomes if o is not None)

        if valid_outcomes:
            valid_cases += 1
            # Take the first valid outcome as representative
            representative_outcome = next(iter(valid_outcomes))
            case_outcomes_summary.append((case_id, representative_outcome))
        else:
            missing_cases += 1

    total_cases = len(cases_outcomes)

    print(f"\n=== ENHANCED DATA ANALYSIS ===")
    print(f"Total records: {total_records:,}")
    print(f"Total cases: {total_cases}")
    print(f"Valid outcome cases: {valid_cases}")
    print(f"Missing outcome cases: {missing_cases}")
    print(f"Valid percentage: {valid_cases/total_cases:.1%}")

    # Analyze outcome distribution for valid cases
    if case_outcomes_summary:
        outcomes = [outcome for _, outcome in case_outcomes_summary]

        # Create bins for regression analysis (like the k-fold script)
        bin_edges = [float(x) for x in sorted(outcomes)]
        if len(bin_edges) >= 3:
            # Create 3 quantile bins
            import numpy as np

            quantiles = np.quantile(bin_edges, [0, 1 / 3, 2 / 3, 1])

            binned_outcomes = []
            for outcome in outcomes:
                bin_idx = np.digitize(outcome, quantiles) - 1
                bin_idx = np.clip(bin_idx, 0, 2)
                binned_outcomes.append(f"bin_{bin_idx}")

            bin_counts = Counter(binned_outcomes)
            print(f"\nOutcome distribution (3 bins):")
            for bin_name, count in sorted(bin_counts.items()):
                print(f"  {bin_name}: {count} cases ({count/valid_cases:.1%})")

    # Sample case IDs
    print(f"\nSample case IDs:")
    sample_cases = sorted(list(cases_outcomes.keys()))[:10]
    for case_id in sample_cases:
        outcomes = cases_outcomes[case_id]
        valid_outcomes = [o for o in outcomes if o is not None]
        if valid_outcomes:
            print(f"  {case_id}: {valid_outcomes[0]:,.0f}")
        else:
            print(f"  {case_id}: missing")


if __name__ == "__main__":
    analyze_enhanced_data("data/enhanced_combined/all_enhanced_data.jsonl")
