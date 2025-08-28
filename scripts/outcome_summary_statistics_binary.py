#!/usr/bin/env python3
"""
outcome_summary_statistics_binary.py

Comprehensive summary statistics analyzer for binary case outcome results.
Analyzes the distribution of final_judgement_real values for binary classification,
providing detailed statistics by case and by individual observations/entries.

Usage:
    python scripts/outcome_summary_statistics_binary.py \
        --input-data data/enhanced_combined/final_clean_dataset_with_interpretable_features.jsonl \
        --kfold-dir data/final_stratified_kfold_splits_binary \
        --max-threshold 5000000000.00

Features:
- Complete binary distribution analysis (min, max, mean, median, std, skewness, kurtosis)
- Percentile analysis with binary boundary emphasis
- Binary support counts (Lower Risk vs Higher Risk)
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


def load_binary_outcome_data(
    input_file: Path, kfold_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Load all final_judgement_real values and compute binary classification.

    Args:
        input_file: Path to enhanced dataset JSONL file
        kfold_dir: Path to binary k-fold splits directory

    Returns:
        Tuple of (entry_level_df, case_level_df, binary_edge)
    """
    print(f"Loading binary outcome data from: {input_file}")
    print(f"Loading binary k-fold metadata from: {kfold_dir}")

    # Load k-fold metadata to get binary boundary
    metadata_file = kfold_dir / "per_fold_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        # Get binary boundary from fold 4 (final training fold)
        binary_edge = metadata["binning"]["fold_edges"]["fold_4"][0]
        print(f"✓ Binary boundary from k-fold: ${binary_edge:,.0f}")
    else:
        binary_edge = None
        print("⚠️  No k-fold metadata found, will compute binary boundary from data")

    entry_data = []
    case_data = defaultdict(list)

    processed_lines = 0
    skipped_lines = 0

    # Load the enhanced dataset
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line.strip())
                case_id = data.get("case_id_clean", f"case_{line_num}")
                final_amount = data.get("final_judgement_real")

                if final_amount is not None:
                    entry_data.append(
                        {
                            "case_id": case_id,
                            "final_judgement_real": float(final_amount),
                            "line_number": line_num,
                        }
                    )
                    case_data[case_id].append(float(final_amount))
                    processed_lines += 1

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                skipped_lines += 1
                continue

            if line_num % 5000 == 0:
                print(f"  Processed {line_num:,} records...")

    print(
        f"Processed {processed_lines} valid entries, skipped {skipped_lines} invalid lines"
    )

    # Create entry-level DataFrame
    entry_df = pd.DataFrame(entry_data)
    print(f"Loaded {len(entry_df)} total entries")

    # Create case-level DataFrame (aggregate by case)
    case_aggregates = []
    for case_id, amounts in case_data.items():
        case_aggregates.append(
            {
                "case_id": case_id,
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

    # Compute binary boundary if not available from k-fold
    if binary_edge is None:
        # Use median of case-level means as binary boundary
        binary_edge = case_df["mean_amount"].median()
        print(f"Computed binary boundary from data median: ${binary_edge:,.0f}")

    # Add binary labels
    entry_df["binary_class"] = np.where(
        entry_df["final_judgement_real"] < binary_edge, "Low (Green)", "High (Red)"
    )
    case_df["binary_class"] = np.where(
        case_df["mean_amount"] < binary_edge, "Low (Green)", "High (Red)"
    )

    # Print binary distribution
    entry_dist = entry_df["binary_class"].value_counts()
    case_dist = case_df["binary_class"].value_counts()
    print(f"\n✓ Binary Classification Results:")
    print(f"  Entry distribution: {dict(entry_dist)}")
    print(f"  Case distribution: {dict(case_dist)}")
    print(f"  Binary boundary: ${binary_edge:,.0f}")

    return entry_df, case_df, binary_edge


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

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_dict = {}

    for p in percentiles:
        key = f"p{p}"
        percentile_dict[key] = np.percentile(values, p)

    return percentile_dict


def compute_binary_boundary_analysis(
    values: np.ndarray, boundary: float
) -> Dict[str, Any]:
    """Compute analysis around the binary boundary."""
    if len(values) == 0:
        return {}

    lower_values = values[values < boundary]
    higher_values = values[values >= boundary]

    return {
        "boundary": boundary,
        "total_count": len(values),
        "lower_count": len(lower_values),
        "higher_count": len(higher_values),
        "lower_percentage": len(lower_values) / len(values) * 100,
        "higher_percentage": len(higher_values) / len(values) * 100,
        "lower_mean": np.mean(lower_values) if len(lower_values) > 0 else np.nan,
        "higher_mean": np.mean(higher_values) if len(higher_values) > 0 else np.nan,
        "lower_median": np.median(lower_values) if len(lower_values) > 0 else np.nan,
        "higher_median": np.median(higher_values) if len(higher_values) > 0 else np.nan,
        "lower_std": np.std(lower_values) if len(lower_values) > 0 else np.nan,
        "higher_std": np.std(higher_values) if len(higher_values) > 0 else np.nan,
        "separation_ratio": (
            np.mean(higher_values) / np.mean(lower_values)
            if len(lower_values) > 0
            and len(higher_values) > 0
            and np.mean(lower_values) > 0
            else np.nan
        ),
    }


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


def print_binary_summary_stats(stats: Dict[str, float], title: str) -> None:
    """Print formatted binary summary statistics."""
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
        "p50",
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
        "50th (Median)",
        "75th (Q3)",
        "90th",
        "95th",
        "99th",
        "100th (Max)",
    ]

    for key, label in zip(key_percentiles, labels):
        if key in percentiles:
            print(f"{label:<15}: {format_currency(percentiles[key])}")


def print_binary_boundary_analysis(boundary_stats: Dict[str, Any], title: str) -> None:
    """Print binary boundary analysis."""
    print(f"\n{title}")
    print("=" * len(title))

    boundary = boundary_stats.get("boundary", 0)
    print(f"Binary Boundary:     {format_currency(boundary)}")
    print(f"Total Cases:         {boundary_stats.get('total_count', 0):,}")
    print()

    print("LOWER RISK (GREEN) - Below Boundary:")
    print(
        f"  Count:             {boundary_stats.get('lower_count', 0):,} ({boundary_stats.get('lower_percentage', 0):.1f}%)"
    )
    print(
        f"  Mean:              {format_currency(boundary_stats.get('lower_mean', 0))}"
    )
    print(
        f"  Median:            {format_currency(boundary_stats.get('lower_median', 0))}"
    )
    print(f"  Std Dev:           {format_currency(boundary_stats.get('lower_std', 0))}")
    print()

    print("HIGHER RISK (RED) - At or Above Boundary:")
    print(
        f"  Count:             {boundary_stats.get('higher_count', 0):,} ({boundary_stats.get('higher_percentage', 0):.1f}%)"
    )
    print(
        f"  Mean:              {format_currency(boundary_stats.get('higher_mean', 0))}"
    )
    print(
        f"  Median:            {format_currency(boundary_stats.get('higher_median', 0))}"
    )
    print(
        f"  Std Dev:           {format_currency(boundary_stats.get('higher_std', 0))}"
    )
    print()

    separation_ratio = boundary_stats.get("separation_ratio", np.nan)
    if not pd.isna(separation_ratio):
        print(f"Risk Separation Ratio (Higher/Lower Mean): {separation_ratio:.2f}x")


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
    """Main binary analysis function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive summary statistics for binary case outcome results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input-data",
        type=Path,
        default=Path(
            "data/enhanced_combined/final_clean_dataset_with_interpretable_features.jsonl"
        ),
        help="Path to enhanced dataset JSONL file",
    )

    parser.add_argument(
        "--kfold-dir",
        type=Path,
        default=Path("data/final_stratified_kfold_splits_binary"),
        help="Path to binary k-fold splits directory",
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

    # Validate input files
    if not args.input_data.exists():
        print(f"Error: Input data file not found: {args.input_data}")
        sys.exit(1)

    if not args.kfold_dir.exists():
        print(f"Error: K-fold directory not found: {args.kfold_dir}")
        sys.exit(1)

    print("Corporate Speech Risk Dataset - Binary Outcome Summary Statistics")
    print("=" * 70)
    print(f"Input Data: {args.input_data}")
    print(f"K-fold Directory: {args.kfold_dir}")
    print(f"Max Threshold: {format_currency(args.max_threshold)}")

    try:
        # Load data
        entry_df, case_df, binary_edge = load_binary_outcome_data(
            args.input_data, args.kfold_dir
        )

        if len(entry_df) == 0:
            print("Error: No valid outcome data found")
            sys.exit(1)

        # Analysis 1: Full dataset (before filtering)
        print(f"\n{'='*70}")
        print("ANALYSIS 1: COMPLETE BINARY DATASET (All Values)")
        print(f"{'='*70}")

        entry_amounts_full = entry_df["final_judgement_real"].values
        case_amounts_full = case_df["mean_amount"].values

        # Entry-level analysis (full)
        entry_stats_full = compute_distribution_stats(entry_amounts_full)
        entry_percentiles_full = compute_percentile_stats(entry_amounts_full)
        entry_boundary_full = compute_binary_boundary_analysis(
            entry_amounts_full, binary_edge
        )
        entry_zero_stats_full = analyze_zero_vs_nonzero(entry_amounts_full)

        print_binary_summary_stats(
            entry_stats_full, "ENTRY-LEVEL STATISTICS (All Observations)"
        )
        print_percentile_stats(entry_percentiles_full, "ENTRY-LEVEL PERCENTILES")
        print_binary_boundary_analysis(
            entry_boundary_full, "ENTRY-LEVEL BINARY BOUNDARY ANALYSIS"
        )
        print_zero_nonzero_analysis(
            entry_zero_stats_full, "ENTRY-LEVEL ZERO/NON-ZERO ANALYSIS"
        )

        # Case-level analysis (full)
        case_stats_full = compute_distribution_stats(case_amounts_full)
        case_percentiles_full = compute_percentile_stats(case_amounts_full)
        case_boundary_full = compute_binary_boundary_analysis(
            case_amounts_full, binary_edge
        )
        case_zero_stats_full = analyze_zero_vs_nonzero(case_amounts_full)

        print_binary_summary_stats(
            case_stats_full, "CASE-LEVEL STATISTICS (Mean per Case)"
        )
        print_percentile_stats(case_percentiles_full, "CASE-LEVEL PERCENTILES")
        print_binary_boundary_analysis(
            case_boundary_full, "CASE-LEVEL BINARY BOUNDARY ANALYSIS"
        )
        print_zero_nonzero_analysis(
            case_zero_stats_full, "CASE-LEVEL ZERO/NON-ZERO ANALYSIS"
        )

        # Analysis 2: Filtered dataset (excluding extreme values)
        print(f"\n{'='*70}")
        print(
            f"ANALYSIS 2: FILTERED BINARY DATASET (Excluding Values > {format_currency(args.max_threshold)})"
        )
        print(f"{'='*70}")

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
            entry_boundary_filtered = compute_binary_boundary_analysis(
                entry_amounts_filtered, binary_edge
            )
            entry_zero_stats_filtered = analyze_zero_vs_nonzero(entry_amounts_filtered)

            print_binary_summary_stats(
                entry_stats_filtered, "ENTRY-LEVEL STATISTICS (Filtered)"
            )
            print_percentile_stats(
                entry_percentiles_filtered, "ENTRY-LEVEL PERCENTILES (Filtered)"
            )
            print_binary_boundary_analysis(
                entry_boundary_filtered,
                "ENTRY-LEVEL BINARY BOUNDARY ANALYSIS (Filtered)",
            )
            print_zero_nonzero_analysis(
                entry_zero_stats_filtered,
                "ENTRY-LEVEL ZERO/NON-ZERO ANALYSIS (Filtered)",
            )

            # Case-level analysis (filtered)
            case_stats_filtered = compute_distribution_stats(case_amounts_filtered)
            case_percentiles_filtered = compute_percentile_stats(case_amounts_filtered)
            case_boundary_filtered = compute_binary_boundary_analysis(
                case_amounts_filtered, binary_edge
            )
            case_zero_stats_filtered = analyze_zero_vs_nonzero(case_amounts_filtered)

            print_binary_summary_stats(
                case_stats_filtered, "CASE-LEVEL STATISTICS (Filtered)"
            )
            print_percentile_stats(
                case_percentiles_filtered, "CASE-LEVEL PERCENTILES (Filtered)"
            )
            print_binary_boundary_analysis(
                case_boundary_filtered, "CASE-LEVEL BINARY BOUNDARY ANALYSIS (Filtered)"
            )
            print_zero_nonzero_analysis(
                case_zero_stats_filtered, "CASE-LEVEL ZERO/NON-ZERO ANALYSIS (Filtered)"
            )

        # Analysis 3: Binary Class Comparison
        print(f"\n{'='*70}")
        print("ANALYSIS 3: BINARY CLASS COMPARISON")
        print(f"{'='*70}")

        # Entry-level by binary class
        entry_lower = entry_df[entry_df["binary_class"] == "Low (Green)"][
            "final_judgement_real"
        ].values
        entry_higher = entry_df[entry_df["binary_class"] == "High (Red)"][
            "final_judgement_real"
        ].values

        entry_lower_stats = compute_distribution_stats(entry_lower)
        entry_higher_stats = compute_distribution_stats(entry_higher)

        print_binary_summary_stats(entry_lower_stats, "ENTRY-LEVEL: LOWER RISK (GREEN)")
        print_binary_summary_stats(entry_higher_stats, "ENTRY-LEVEL: HIGHER RISK (RED)")

        # Case-level by binary class
        case_lower = case_df[case_df["binary_class"] == "Low (Green)"][
            "mean_amount"
        ].values
        case_higher = case_df[case_df["binary_class"] == "High (Red)"][
            "mean_amount"
        ].values

        case_lower_stats = compute_distribution_stats(case_lower)
        case_higher_stats = compute_distribution_stats(case_higher)

        print_binary_summary_stats(case_lower_stats, "CASE-LEVEL: LOWER RISK (GREEN)")
        print_binary_summary_stats(case_higher_stats, "CASE-LEVEL: HIGHER RISK (RED)")

        # Summary comparison
        print(f"\n{'='*70}")
        print("BINARY SUMMARY COMPARISON")
        print(f"{'='*70}")

        print(f"\nDataset Overview:")
        print(f"Total unique cases:           {len(case_df):,}")
        print(f"Total entries/observations:   {len(entry_df):,}")
        print(f"Avg entries per case:         {len(entry_df) / len(case_df):.1f}")
        print(f"Binary boundary:              {format_currency(binary_edge)}")

        print(f"\nBinary Distribution (Cases):")
        case_dist = case_df["binary_class"].value_counts()
        for class_name, count in case_dist.items():
            pct = count / len(case_df) * 100
            print(f"  {class_name}: {count:,} ({pct:.1f}%)")

        print(f"\nBinary Distribution (Entries):")
        entry_dist = entry_df["binary_class"].value_counts()
        for class_name, count in entry_dist.items():
            pct = count / len(entry_df) * 100
            print(f"  {class_name}: {count:,} ({pct:.1f}%)")

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
            print(f"\nExporting detailed binary data to: {args.export_csv}")

            # Combine all data for export
            export_df = entry_df.copy()
            export_df["case_entry_count"] = export_df["case_id"].map(
                case_df.set_index("case_id")["entry_count"]
            )
            export_df["case_mean_amount"] = export_df["case_id"].map(
                case_df.set_index("case_id")["mean_amount"]
            )
            export_df["case_binary_class"] = export_df["case_id"].map(
                case_df.set_index("case_id")["binary_class"]
            )
            export_df["is_filtered"] = (
                export_df["final_judgement_real"] <= args.max_threshold
            )
            export_df["binary_edge"] = binary_edge

            export_df.to_csv(args.export_csv, index=False)
            print(f"Exported {len(export_df)} records to CSV")

        print(f"\nBinary analysis completed successfully!")

    except Exception as e:
        print(f"\nError during binary analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
