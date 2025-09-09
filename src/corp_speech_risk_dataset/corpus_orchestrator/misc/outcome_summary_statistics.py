#!/usr/bin/env python3
"""
outcome_summary_statistics.py

Comprehensive summary statistics analyzer for case outcome imputation results.
Analyzes the distribution of final_judgement_real values from the imputed outcomes,
providing detailed statistics by case and by individual observations/entries.

Usage:
    python scripts/outcome_summary_statistics.py \
        --outcomes-dir data/outcomes/courtlistener_v1 \
        --max-threshold 5000000000.00

Features:
- Complete distribution analysis (min, max, mean, median, std, skewness, kurtosis)
- Percentile analysis (quartiles, deciles, and custom percentiles)
- Support counts for 25th and 33rd percentile buckets
- Case-level and entry-level statistics
- Optional filtering for extreme values above threshold
- Comprehensive visualization-ready output

Author: Jake Dugan <jake.dugan@ed.ac.uk>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict, Counter
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def load_outcome_data(outcomes_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all final_judgement_real values from the outcomes directory.

    Args:
        outcomes_dir: Path to outcomes directory containing case folders

    Returns:
        Tuple of (entry_level_df, case_level_df)
    """
    print(f"Loading outcome data from: {outcomes_dir}")

    entry_data = []
    case_data = defaultdict(list)

    # Find all JSONL files
    jsonl_files = list(outcomes_dir.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files to process")

    processed_files = 0
    skipped_files = 0

    for jsonl_path in jsonl_files:
        # Extract case name from path
        case_name = jsonl_path.relative_to(outcomes_dir).parts[0]
        doc_id = jsonl_path.stem

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        final_amount = data.get("final_judgement_real")

                        if final_amount is not None:
                            entry_data.append(
                                {
                                    "case_name": case_name,
                                    "doc_id": doc_id,
                                    "file_path": str(jsonl_path),
                                    "line_number": line_num,
                                    "final_judgement_real": float(final_amount),
                                }
                            )
                            case_data[case_name].append(float(final_amount))

                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        print(
                            f"Warning: Skipping invalid line {line_num} in {jsonl_path}: {e}"
                        )
                        continue

            processed_files += 1

        except Exception as e:
            print(f"Error processing {jsonl_path}: {e}")
            skipped_files += 1
            continue

    print(
        f"Processed {processed_files} files successfully, skipped {skipped_files} files"
    )

    # Create entry-level DataFrame
    entry_df = pd.DataFrame(entry_data)
    print(f"Loaded {len(entry_df)} total entries")

    # Create case-level DataFrame (aggregate by case)
    case_aggregates = []
    for case_name, amounts in case_data.items():
        case_aggregates.append(
            {
                "case_name": case_name,
                "entry_count": len(amounts),
                "mean_amount": np.mean(amounts),
                "median_amount": np.median(amounts),
                "min_amount": np.min(amounts),
                "max_amount": np.max(amounts),
                "std_amount": np.std(amounts),
                "unique_amounts": len(set(amounts)),
                "most_common_amount": (
                    Counter(amounts).most_common(1)[0][0] if amounts else None
                ),
            }
        )

    case_df = pd.DataFrame(case_aggregates)
    print(f"Aggregated data for {len(case_df)} unique cases")

    return entry_df, case_df


def filter_extreme_values(
    df: pd.DataFrame, amount_col: str, max_threshold: float
) -> pd.DataFrame:
    """Filter out extreme values above the threshold."""
    original_count = len(df)
    filtered_df = df[df[amount_col] <= max_threshold].copy()
    filtered_count = len(filtered_df)
    excluded_count = original_count - filtered_count

    if excluded_count > 0:
        print(
            f"Filtered out {excluded_count} entries with amounts > ${max_threshold:,.2f}"
        )
        print(f"Retained {filtered_count} entries for analysis")

    return filtered_df


def compute_distribution_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive distribution statistics."""
    if len(values) == 0:
        return {}

    return {
        "count": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "var": np.var(values),
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
        "skewness": stats.skew(values) if len(values) > 2 else np.nan,
        "kurtosis": stats.kurtosis(values) if len(values) > 3 else np.nan,
        "mad": stats.median_abs_deviation(
            values, scale="normal"
        ),  # Median Absolute Deviation
    }


def compute_percentile_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute percentile-based statistics."""
    if len(values) == 0:
        return {}

    percentiles = [0, 1, 5, 10, 25, 33.33, 50, 66.67, 75, 90, 95, 99, 100]
    percentile_dict = {}

    for p in percentiles:
        key = f"p{p:g}" if p == int(p) else f"p{p:.2f}".replace(".", "_")
        percentile_dict[key] = np.percentile(values, p)

    return percentile_dict


def compute_percentile_bucket_support(values: np.ndarray) -> Dict[str, Dict]:
    """Compute support counts for percentile buckets."""
    if len(values) == 0:
        return {}

    results = {}

    # 25th percentile buckets (quartiles)
    q25_thresholds = np.percentile(values, [0, 25, 50, 75, 100])
    q25_labels = ["0-25th", "25-50th", "50-75th", "75-100th"]
    q25_counts = []

    for i in range(len(q25_thresholds) - 1):
        if i == len(q25_thresholds) - 2:  # Last bucket includes max value
            count = np.sum(
                (values >= q25_thresholds[i]) & (values <= q25_thresholds[i + 1])
            )
        else:
            count = np.sum(
                (values >= q25_thresholds[i]) & (values < q25_thresholds[i + 1])
            )
        q25_counts.append(count)

    results["quartile_buckets"] = {
        "labels": q25_labels,
        "counts": q25_counts,
        "percentages": [count / len(values) * 100 for count in q25_counts],
        "thresholds": q25_thresholds.tolist(),
    }

    # 33rd percentile buckets (terciles)
    q33_thresholds = np.percentile(values, [0, 33.33, 66.67, 100])
    q33_labels = ["0-33rd", "33-67th", "67-100th"]
    q33_counts = []

    for i in range(len(q33_thresholds) - 1):
        if i == len(q33_thresholds) - 2:  # Last bucket includes max value
            count = np.sum(
                (values >= q33_thresholds[i]) & (values <= q33_thresholds[i + 1])
            )
        else:
            count = np.sum(
                (values >= q33_thresholds[i]) & (values < q33_thresholds[i + 1])
            )
        q33_counts.append(count)

    results["tercile_buckets"] = {
        "labels": q33_labels,
        "counts": q33_counts,
        "percentages": [count / len(values) * 100 for count in q33_counts],
        "thresholds": q33_thresholds.tolist(),
    }

    return results


def analyze_zero_vs_nonzero(values: np.ndarray) -> Dict[str, Any]:
    """Analyze distribution of zero vs non-zero values."""
    zero_count = np.sum(values == 0)
    nonzero_count = np.sum(values > 0)
    negative_count = np.sum(values < 0)

    return {
        "zero_count": zero_count,
        "zero_percentage": zero_count / len(values) * 100 if len(values) > 0 else 0,
        "nonzero_count": nonzero_count,
        "nonzero_percentage": (
            nonzero_count / len(values) * 100 if len(values) > 0 else 0
        ),
        "negative_count": negative_count,
        "negative_percentage": (
            negative_count / len(values) * 100 if len(values) > 0 else 0
        ),
        "positive_mean": np.mean(values[values > 0]) if nonzero_count > 0 else np.nan,
        "positive_median": (
            np.median(values[values > 0]) if nonzero_count > 0 else np.nan
        ),
    }


def format_currency(value: float) -> str:
    """Format value as currency."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def print_summary_stats(stats: Dict[str, float], title: str) -> None:
    """Print formatted summary statistics."""
    print(f"\n{title}")
    print("=" * len(title))

    print(f"Count:               {stats.get('count', 0):,}")
    print(f"Mean:                {format_currency(stats.get('mean', 0))}")
    print(f"Median:              {format_currency(stats.get('median', 0))}")
    print(f"Standard Deviation:  {format_currency(stats.get('std', 0))}")
    print(f"Variance:            {format_currency(stats.get('var', 0))}")
    print(f"Minimum:             {format_currency(stats.get('min', 0))}")
    print(f"Maximum:             {format_currency(stats.get('max', 0))}")
    print(f"Range:               {format_currency(stats.get('range', 0))}")
    print(f"Skewness:            {stats.get('skewness', np.nan):.4f}")
    print(f"Kurtosis:            {stats.get('kurtosis', np.nan):.4f}")
    print(f"Median Abs Dev:      {format_currency(stats.get('mad', 0))}")


def print_percentile_stats(percentiles: Dict[str, float], title: str) -> None:
    """Print formatted percentile statistics."""
    print(f"\n{title}")
    print("=" * len(title))

    # Standard percentiles
    key_percentiles = [
        "p0",
        "p1",
        "p5",
        "p10",
        "p25",
        "p33_33",
        "p50",
        "p66_67",
        "p75",
        "p90",
        "p95",
        "p99",
        "p100",
    ]
    labels = [
        "0th (Min)",
        "1st",
        "5th",
        "10th",
        "25th (Q1)",
        "33rd",
        "50th (Median)",
        "67th",
        "75th (Q3)",
        "90th",
        "95th",
        "99th",
        "100th (Max)",
    ]

    for key, label in zip(key_percentiles, labels):
        if key in percentiles:
            print(f"{label:<15}: {format_currency(percentiles[key])}")


def print_bucket_analysis(bucket_stats: Dict[str, Dict], title: str) -> None:
    """Print formatted bucket analysis."""
    print(f"\n{title}")
    print("=" * len(title))

    # Quartile buckets
    if "quartile_buckets" in bucket_stats:
        print("\nQuartile Buckets (25th percentile):")
        print("-" * 40)
        q_data = bucket_stats["quartile_buckets"]
        for i, (label, count, pct, threshold) in enumerate(
            zip(
                q_data["labels"],
                q_data["counts"],
                q_data["percentages"],
                q_data["thresholds"][:-1],
            )
        ):
            next_threshold = q_data["thresholds"][i + 1]
            print(
                f"{label:<12}: {count:>6,} ({pct:>5.1f}%) | Range: {format_currency(threshold)} - {format_currency(next_threshold)}"
            )

    # Tercile buckets
    if "tercile_buckets" in bucket_stats:
        print("\nTercile Buckets (33rd percentile):")
        print("-" * 40)
        t_data = bucket_stats["tercile_buckets"]
        for i, (label, count, pct, threshold) in enumerate(
            zip(
                t_data["labels"],
                t_data["counts"],
                t_data["percentages"],
                t_data["thresholds"][:-1],
            )
        ):
            next_threshold = t_data["thresholds"][i + 1]
            print(
                f"{label:<12}: {count:>6,} ({pct:>5.1f}%) | Range: {format_currency(threshold)} - {format_currency(next_threshold)}"
            )


def print_zero_nonzero_analysis(zero_stats: Dict[str, Any], title: str) -> None:
    """Print zero vs non-zero analysis."""
    print(f"\n{title}")
    print("=" * len(title))

    print(
        f"Zero values:         {zero_stats['zero_count']:,} ({zero_stats['zero_percentage']:.1f}%)"
    )
    print(
        f"Positive values:     {zero_stats['nonzero_count']:,} ({zero_stats['nonzero_percentage']:.1f}%)"
    )
    print(
        f"Negative values:     {zero_stats['negative_count']:,} ({zero_stats['negative_percentage']:.1f}%)"
    )
    print(f"Positive mean:       {format_currency(zero_stats['positive_mean'])}")
    print(f"Positive median:     {format_currency(zero_stats['positive_median'])}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive summary statistics for case outcome imputation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--outcomes-dir",
        type=Path,
        default=Path("data/outcomes/courtlistener_v1"),
        help="Path to outcomes directory (default: data/outcomes/courtlistener_v1)",
    )

    parser.add_argument(
        "--max-threshold",
        type=float,
        default=5_000_000_000.00,
        help="Maximum threshold for filtering extreme values (default: 5,000,000,000.00)",
    )

    parser.add_argument(
        "--export-csv", type=Path, help="Optional: Export detailed data to CSV file"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.outcomes_dir.exists():
        print(f"Error: Outcomes directory not found: {args.outcomes_dir}")
        sys.exit(1)

    print("Corporate Speech Risk Dataset - Outcome Summary Statistics")
    print("=" * 60)
    print(f"Outcomes Directory: {args.outcomes_dir}")
    print(f"Max Threshold: {format_currency(args.max_threshold)}")

    try:
        # Load data
        entry_df, case_df = load_outcome_data(args.outcomes_dir)

        if len(entry_df) == 0:
            print("Error: No valid outcome data found")
            sys.exit(1)

        # Analysis 1: Full dataset (before filtering)
        print(f"\n{'='*60}")
        print("ANALYSIS 1: COMPLETE DATASET (All Values)")
        print(f"{'='*60}")

        entry_amounts_full = entry_df["final_judgement_real"].values
        case_amounts_full = case_df["mean_amount"].values

        # Entry-level analysis (full)
        entry_stats_full = compute_distribution_stats(entry_amounts_full)
        entry_percentiles_full = compute_percentile_stats(entry_amounts_full)
        entry_buckets_full = compute_percentile_bucket_support(entry_amounts_full)
        entry_zero_stats_full = analyze_zero_vs_nonzero(entry_amounts_full)

        print_summary_stats(
            entry_stats_full, "ENTRY-LEVEL STATISTICS (All Observations)"
        )
        print_percentile_stats(entry_percentiles_full, "ENTRY-LEVEL PERCENTILES")
        print_bucket_analysis(entry_buckets_full, "ENTRY-LEVEL BUCKET ANALYSIS")
        print_zero_nonzero_analysis(
            entry_zero_stats_full, "ENTRY-LEVEL ZERO/NON-ZERO ANALYSIS"
        )

        # Case-level analysis (full)
        case_stats_full = compute_distribution_stats(case_amounts_full)
        case_percentiles_full = compute_percentile_stats(case_amounts_full)
        case_buckets_full = compute_percentile_bucket_support(case_amounts_full)
        case_zero_stats_full = analyze_zero_vs_nonzero(case_amounts_full)

        print_summary_stats(case_stats_full, "CASE-LEVEL STATISTICS (Mean per Case)")
        print_percentile_stats(case_percentiles_full, "CASE-LEVEL PERCENTILES")
        print_bucket_analysis(case_buckets_full, "CASE-LEVEL BUCKET ANALYSIS")
        print_zero_nonzero_analysis(
            case_zero_stats_full, "CASE-LEVEL ZERO/NON-ZERO ANALYSIS"
        )

        # Analysis 2: Filtered dataset (excluding extreme values)
        print(f"\n{'='*60}")
        print(
            f"ANALYSIS 2: FILTERED DATASET (Excluding Values > {format_currency(args.max_threshold)})"
        )
        print(f"{'='*60}")

        # Filter datasets
        entry_df_filtered = filter_extreme_values(
            entry_df, "final_judgement_real", args.max_threshold
        )
        case_df_filtered = filter_extreme_values(
            case_df, "mean_amount", args.max_threshold
        )

        if len(entry_df_filtered) == 0:
            print("Warning: No entries remain after filtering")
        else:
            entry_amounts_filtered = entry_df_filtered["final_judgement_real"].values
            case_amounts_filtered = case_df_filtered["mean_amount"].values

            # Entry-level analysis (filtered)
            entry_stats_filtered = compute_distribution_stats(entry_amounts_filtered)
            entry_percentiles_filtered = compute_percentile_stats(
                entry_amounts_filtered
            )
            entry_buckets_filtered = compute_percentile_bucket_support(
                entry_amounts_filtered
            )
            entry_zero_stats_filtered = analyze_zero_vs_nonzero(entry_amounts_filtered)

            print_summary_stats(
                entry_stats_filtered, "ENTRY-LEVEL STATISTICS (Filtered)"
            )
            print_percentile_stats(
                entry_percentiles_filtered, "ENTRY-LEVEL PERCENTILES (Filtered)"
            )
            print_bucket_analysis(
                entry_buckets_filtered, "ENTRY-LEVEL BUCKET ANALYSIS (Filtered)"
            )
            print_zero_nonzero_analysis(
                entry_zero_stats_filtered,
                "ENTRY-LEVEL ZERO/NON-ZERO ANALYSIS (Filtered)",
            )

            # Case-level analysis (filtered)
            case_stats_filtered = compute_distribution_stats(case_amounts_filtered)
            case_percentiles_filtered = compute_percentile_stats(case_amounts_filtered)
            case_buckets_filtered = compute_percentile_bucket_support(
                case_amounts_filtered
            )
            case_zero_stats_filtered = analyze_zero_vs_nonzero(case_amounts_filtered)

            print_summary_stats(case_stats_filtered, "CASE-LEVEL STATISTICS (Filtered)")
            print_percentile_stats(
                case_percentiles_filtered, "CASE-LEVEL PERCENTILES (Filtered)"
            )
            print_bucket_analysis(
                case_buckets_filtered, "CASE-LEVEL BUCKET ANALYSIS (Filtered)"
            )
            print_zero_nonzero_analysis(
                case_zero_stats_filtered, "CASE-LEVEL ZERO/NON-ZERO ANALYSIS (Filtered)"
            )

        # Summary comparison
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")

        print(f"\nDataset Overview:")
        print(f"Total unique cases:           {len(case_df):,}")
        print(f"Total entries/observations:   {len(entry_df):,}")
        print(f"Avg entries per case:         {len(entry_df) / len(case_df):.1f}")

        print(f"\nValue Distribution (Full Dataset):")
        print(
            f"Entry-level mean:             {format_currency(entry_stats_full['mean'])}"
        )
        print(
            f"Case-level mean:              {format_currency(case_stats_full['mean'])}"
        )
        print(
            f"Highest single value:         {format_currency(entry_stats_full['max'])}"
        )
        print(
            f"Values above threshold:       {len(entry_df) - len(entry_df_filtered):,}"
        )

        # Export to CSV if requested
        if args.export_csv:
            print(f"\nExporting detailed data to: {args.export_csv}")

            # Combine all data for export
            export_df = entry_df.copy()
            export_df["case_entry_count"] = export_df["case_name"].map(
                case_df.set_index("case_name")["entry_count"]
            )
            export_df["case_mean_amount"] = export_df["case_name"].map(
                case_df.set_index("case_name")["mean_amount"]
            )
            export_df["case_unique_amounts"] = export_df["case_name"].map(
                case_df.set_index("case_name")["unique_amounts"]
            )
            export_df["is_filtered"] = (
                export_df["final_judgement_real"] <= args.max_threshold
            )

            export_df.to_csv(args.export_csv, index=False)
            print(f"Exported {len(export_df)} records to CSV")

        print(f"\nAnalysis completed successfully!")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
