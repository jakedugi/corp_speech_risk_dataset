#!/usr/bin/env python3
"""
Analyze speaker-case relationships to determine feasibility of speaker-disjoint k-fold splits.

This script investigates:
1. How many speakers appear in multiple cases
2. Distribution of speakers across cases
3. Feasibility of case-wise + speaker-disjoint splitting
4. Impact on stratification quality

Critical for determining if we can fix the speaker leakage problem.
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


def analyze_speaker_case_relationships(file_path: str) -> Dict:
    """Analyze how speakers are distributed across cases."""
    print("=" * 70)
    print("SPEAKER-CASE RELATIONSHIP ANALYSIS")
    print("=" * 70)

    # Data structures
    speaker_to_cases = defaultdict(set)  # speaker -> set of cases
    case_to_speakers = defaultdict(set)  # case -> set of speakers
    case_outcomes = {}  # case -> outcome
    case_sizes = {}  # case -> number of records

    print("Loading data...")
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
            case_to_speakers[case_id].add(speaker)
            case_outcomes[case_id] = outcome
            case_sizes[case_id] = case_sizes.get(case_id, 0) + 1

            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} records...")

    print(f"✓ Loaded {line_num:,} records from {len(case_to_speakers)} cases")

    # Analysis 1: Speaker distribution across cases
    print("\n" + "=" * 50)
    print("1. SPEAKER DISTRIBUTION ANALYSIS")
    print("=" * 50)

    speakers_in_multiple_cases = 0
    speakers_in_single_case = 0
    max_cases_per_speaker = 0

    for speaker, cases in speaker_to_cases.items():
        num_cases = len(cases)
        if num_cases > 1:
            speakers_in_multiple_cases += 1
        else:
            speakers_in_single_case += 1
        max_cases_per_speaker = max(max_cases_per_speaker, num_cases)

    total_speakers = len(speaker_to_cases)
    pct_multi_case = speakers_in_multiple_cases / total_speakers * 100

    print(f"Total unique speakers: {total_speakers:,}")
    print(
        f"Speakers in single case: {speakers_in_single_case:,} ({100-pct_multi_case:.1f}%)"
    )
    print(
        f"Speakers in multiple cases: {speakers_in_multiple_cases:,} ({pct_multi_case:.1f}%)"
    )
    print(f"Max cases per speaker: {max_cases_per_speaker}")

    # Distribution of cases per speaker
    cases_per_speaker = [len(cases) for cases in speaker_to_cases.values()]
    print(f"\nCases per speaker distribution:")
    print(f"  Mean: {np.mean(cases_per_speaker):.2f}")
    print(f"  Median: {np.median(cases_per_speaker):.1f}")
    print(f"  95th percentile: {np.percentile(cases_per_speaker, 95):.1f}")

    # Analysis 2: Case-level speaker diversity
    print("\n" + "=" * 50)
    print("2. CASE-LEVEL SPEAKER ANALYSIS")
    print("=" * 50)

    speakers_per_case = [len(speakers) for speakers in case_to_speakers.values()]
    print(f"Speakers per case distribution:")
    print(f"  Mean: {np.mean(speakers_per_case):.1f}")
    print(f"  Median: {np.median(speakers_per_case):.1f}")
    print(f"  Min: {min(speakers_per_case)}")
    print(f"  Max: {max(speakers_per_case)}")

    # Analysis 3: Speaker disjointness feasibility
    print("\n" + "=" * 50)
    print("3. SPEAKER DISJOINTNESS FEASIBILITY")
    print("=" * 50)

    # Create speaker-case graph to analyze connectivity
    print("Analyzing speaker-case connectivity...")

    # Find connected components (cases connected through shared speakers)
    case_clusters = []
    processed_cases = set()

    for case_id in case_to_speakers:
        if case_id in processed_cases:
            continue

        # BFS to find all cases connected through speakers
        cluster = set()
        queue = [case_id]

        while queue:
            current_case = queue.pop(0)
            if current_case in processed_cases:
                continue

            processed_cases.add(current_case)
            cluster.add(current_case)

            # Find all speakers in this case
            case_speakers = case_to_speakers[current_case]

            # Find all other cases with these speakers
            for speaker in case_speakers:
                for connected_case in speaker_to_cases[speaker]:
                    if connected_case not in processed_cases:
                        queue.append(connected_case)

        case_clusters.append(cluster)

    # Analyze cluster sizes
    cluster_sizes = [len(cluster) for cluster in case_clusters]
    print(f"Number of case clusters: {len(case_clusters)}")
    print(f"Cluster size distribution:")
    print(f"  Mean: {np.mean(cluster_sizes):.1f}")
    print(f"  Median: {np.median(cluster_sizes):.1f}")
    print(f"  Max: {max(cluster_sizes)}")
    print(
        f"  Largest cluster: {max(cluster_sizes)} cases ({max(cluster_sizes)/len(case_to_speakers)*100:.1f}%)"
    )

    # Analysis 4: Impact on stratification
    print("\n" + "=" * 50)
    print("4. STRATIFICATION IMPACT ANALYSIS")
    print("=" * 50)

    # Calculate outcome bins
    valid_outcomes = [
        outcome for outcome in case_outcomes.values() if outcome is not None
    ]
    valid_outcomes.sort()

    if len(valid_outcomes) >= 3:
        quantiles = np.quantile(valid_outcomes, [0, 1 / 3, 2 / 3, 1])

        # Assign bins to cases
        case_bins = {}
        for case_id, outcome in case_outcomes.items():
            if outcome is not None:
                bin_idx = np.digitize(outcome, quantiles) - 1
                bin_idx = np.clip(bin_idx, 0, 2)
                case_bins[case_id] = f"bin_{bin_idx}"

        # Analyze bin distribution within clusters
        print("Outcome bin distribution within case clusters:")
        cluster_bin_analysis = []

        for i, cluster in enumerate(case_clusters):
            cluster_bins = [case_bins.get(case_id, "missing") for case_id in cluster]
            bin_counts = Counter(cluster_bins)

            cluster_info = {
                "cluster_id": i,
                "size": len(cluster),
                "bin_counts": dict(bin_counts),
                "has_all_bins": len([b for b in bin_counts if b.startswith("bin_")])
                >= 3,
            }
            cluster_bin_analysis.append(cluster_info)

        # Count clusters with all three bins
        clusters_with_all_bins = sum(
            1 for c in cluster_bin_analysis if c["has_all_bins"]
        )
        largest_cluster_info = max(cluster_bin_analysis, key=lambda x: x["size"])

        print(
            f"Clusters with all 3 outcome bins: {clusters_with_all_bins}/{len(case_clusters)} ({clusters_with_all_bins/len(case_clusters)*100:.1f}%)"
        )
        print(f"Largest cluster bin distribution: {largest_cluster_info['bin_counts']}")

    # Analysis 5: Recommendations
    print("\n" + "=" * 50)
    print("5. RECOMMENDATIONS")
    print("=" * 50)

    if pct_multi_case > 50:
        print("❌ HIGH SPEAKER OVERLAP DETECTED")
        print(f"   {pct_multi_case:.1f}% of speakers appear in multiple cases")
        print("   Speaker-disjoint splits will severely impact balance")
    elif pct_multi_case > 20:
        print("⚠️  MODERATE SPEAKER OVERLAP")
        print(f"   {pct_multi_case:.1f}% of speakers appear in multiple cases")
        print("   Speaker-disjoint splits may be feasible but will reduce balance")
    else:
        print("✅ LOW SPEAKER OVERLAP")
        print(f"   Only {pct_multi_case:.1f}% of speakers appear in multiple cases")
        print("   Speaker-disjoint splits should be feasible")

    if max(cluster_sizes) > len(case_to_speakers) * 0.5:
        print("❌ LARGE CONNECTED COMPONENT")
        print(
            f"   Largest cluster contains {max(cluster_sizes)} cases ({max(cluster_sizes)/len(case_to_speakers)*100:.1f}%)"
        )
        print("   True speaker disjointness will break most of the dataset")
    elif len(case_clusters) >= 5:
        print("✅ GOOD CLUSTER DISTRIBUTION")
        print(f"   {len(case_clusters)} clusters can support k-fold CV")
    else:
        print("⚠️  FEW CLUSTERS")
        print(f"   Only {len(case_clusters)} clusters - may limit k-fold options")

    return {
        "total_speakers": total_speakers,
        "total_cases": len(case_to_speakers),
        "speakers_in_multiple_cases": speakers_in_multiple_cases,
        "pct_multi_case": pct_multi_case,
        "max_cases_per_speaker": max_cases_per_speaker,
        "case_clusters": case_clusters,
        "cluster_sizes": cluster_sizes,
        "largest_cluster_size": max(cluster_sizes),
        "cluster_bin_analysis": (
            cluster_bin_analysis if "cluster_bin_analysis" in locals() else []
        ),
    }


def test_speaker_disjoint_splits(analysis: Dict, k_folds: int = 5) -> Dict:
    """Test feasibility of speaker-disjoint k-fold splits."""
    print("\n" + "=" * 50)
    print("6. SPEAKER-DISJOINT K-FOLD TEST")
    print("=" * 50)

    case_clusters = analysis["case_clusters"]

    if len(case_clusters) < k_folds:
        print(f"❌ INSUFFICIENT CLUSTERS: {len(case_clusters)} < {k_folds} folds")
        print("   Cannot create speaker-disjoint k-fold splits")
        return {"feasible": False, "reason": "insufficient_clusters"}

    # Sort clusters by size (largest first)
    sorted_clusters = sorted(case_clusters, key=len, reverse=True)

    # Try to distribute clusters to folds
    folds = [[] for _ in range(k_folds)]
    fold_sizes = [0] * k_folds

    for cluster in sorted_clusters:
        # Assign to smallest fold
        smallest_fold = min(range(k_folds), key=lambda i: fold_sizes[i])
        folds[smallest_fold].extend(cluster)
        fold_sizes[smallest_fold] += len(cluster)

    print(f"✅ SPEAKER-DISJOINT SPLITS POSSIBLE")
    print(f"Cases per fold: {fold_sizes}")
    print(f"Balance quality:")
    print(f"  Min cases: {min(fold_sizes)}")
    print(f"  Max cases: {max(fold_sizes)}")
    print(f"  Ratio: {max(fold_sizes)/min(fold_sizes):.2f}")

    balance_quality = (
        "GOOD"
        if max(fold_sizes) / min(fold_sizes) < 1.5
        else "MODERATE" if max(fold_sizes) / min(fold_sizes) < 2.0 else "POOR"
    )
    print(f"  Quality: {balance_quality}")

    return {
        "feasible": True,
        "folds": folds,
        "fold_sizes": fold_sizes,
        "balance_ratio": max(fold_sizes) / min(fold_sizes),
        "balance_quality": balance_quality,
    }


def main():
    """Main analysis function."""
    input_file = "data/enhanced_combined/final_clean_dataset.jsonl"

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return

    # Run main analysis
    analysis = analyze_speaker_case_relationships(input_file)

    # Test speaker-disjoint feasibility
    speaker_disjoint_test = test_speaker_disjoint_splits(analysis)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if analysis["pct_multi_case"] > 50:
        print("🚨 SPEAKER LEAKAGE IS A MAJOR PROBLEM")
        print("   Current k-fold splits have complete speaker overlap")
        print("   Model will likely overfit to speaker patterns")
        print("   Need to either:")
        print("   1. Accept reduced balance with speaker-disjoint splits")
        print("   2. Acknowledge limitation in methodology section")
        print("   3. Run supplementary analysis showing performance drop")
    else:
        print("✅ SPEAKER LEAKAGE IS MANAGEABLE")
        print("   Can implement speaker-disjoint splits without major impact")

    if speaker_disjoint_test["feasible"]:
        print(f"\n✅ SPEAKER-DISJOINT K-FOLD IS FEASIBLE")
        print(f"   Balance quality: {speaker_disjoint_test['balance_quality']}")
        print(f"   Fold size ratio: {speaker_disjoint_test['balance_ratio']:.2f}")
    else:
        print(f"\n❌ SPEAKER-DISJOINT K-FOLD NOT FEASIBLE")
        print(f"   Reason: {speaker_disjoint_test['reason']}")

    return analysis, speaker_disjoint_test


if __name__ == "__main__":
    main()
