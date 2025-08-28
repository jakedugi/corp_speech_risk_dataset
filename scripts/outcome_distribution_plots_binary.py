#!/usr/bin/env python3
"""
outcome_distribution_plots_binary.py

Creates visualization plots for the binary case outcome distribution results.
Generates histograms, box plots, and distribution comparisons for binary analysis.

Usage:
    python scripts/outcome_distribution_plots_binary.py \
        --input-data data/enhanced_combined/final_clean_dataset_with_interpretable_features.jsonl \
        --kfold-dir data/final_stratified_kfold_splits_binary \
        --output-dir plots/outcomes_binary \
        --max-threshold 5000000000.00

Author: Jake Dugan <jake.dugan@ed.ac.uk>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Binary colors
BINARY_COLORS = {
    "lower": "#2E7D32",  # Green - Lower Risk
    "higher": "#C62828",  # Red - Higher Risk
    "green": "#2E7D32",
    "red": "#C62828",
}


def load_binary_outcome_data(
    input_file: Path, kfold_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load binary outcome data from enhanced dataset and k-fold metadata."""
    print(f"Loading binary outcome data from: {input_file}")
    print(f"Loading binary k-fold metadata from: {kfold_dir}")

    # Load k-fold metadata to get binary boundary
    metadata_file = kfold_dir / "per_fold_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        # Get binary boundary from fold 3 (final training fold)
        binary_edge = metadata["binning"]["fold_edges"]["fold_4"][0]
        print(f"Binary boundary from k-fold: ${binary_edge:,.0f}")
    else:
        binary_edge = None
        print(
            "Warning: No k-fold metadata found, will compute binary boundary from data"
        )

    entry_data = []
    case_data = defaultdict(list)

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

            except (json.JSONDecodeError, ValueError, TypeError):
                continue

            if line_num % 5000 == 0:
                print(f"  Processed {line_num:,} records...")

    # Create DataFrames
    entry_df = pd.DataFrame(entry_data)

    # Create case-level aggregates
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
            }
        )

    case_df = pd.DataFrame(case_aggregates)

    # Compute binary boundary if not available from k-fold
    if binary_edge is None:
        # Use median of case-level means as binary boundary
        binary_edge = case_df["mean_amount"].median()
        print(f"Computed binary boundary from data median: ${binary_edge:,.0f}")

    # Add binary labels
    entry_df["binary_label"] = np.where(
        entry_df["final_judgement_real"] < binary_edge, "Low (Green)", "High (Red)"
    )
    case_df["binary_label"] = np.where(
        case_df["mean_amount"] < binary_edge, "Low (Green)", "High (Red)"
    )

    print(f"Loaded {len(entry_df)} entries from {len(case_df)} cases")
    print(f"Binary split at ${binary_edge:,.0f}")

    # Print binary distribution
    entry_dist = entry_df["binary_label"].value_counts()
    case_dist = case_df["binary_label"].value_counts()
    print(f"Entry distribution: {dict(entry_dist)}")
    print(f"Case distribution: {dict(case_dist)}")

    return entry_df, case_df, binary_edge


def create_binary_distribution_plots(
    entry_df: pd.DataFrame,
    case_df: pd.DataFrame,
    binary_edge: float,
    output_dir: Path,
    max_threshold: float,
):
    """Create comprehensive binary distribution plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter data
    entry_filtered = entry_df[entry_df["final_judgement_real"] <= max_threshold].copy()
    case_filtered = case_df[case_df["mean_amount"] <= max_threshold].copy()

    # 1. Binary distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Binary Distribution of Final Judgement Amounts", fontsize=16, fontweight="bold"
    )

    # Entry-level full data with binary boundary
    axes[0, 0].hist(
        entry_df[entry_df["binary_label"] == "Low (Green)"]["final_judgement_real"],
        bins=30,
        alpha=0.7,
        color=BINARY_COLORS["green"],
        label="Low (Green)",
        edgecolor="black",
    )
    axes[0, 0].hist(
        entry_df[entry_df["binary_label"] == "High (Red)"]["final_judgement_real"],
        bins=30,
        alpha=0.7,
        color=BINARY_COLORS["red"],
        label="High (Red)",
        edgecolor="black",
    )
    axes[0, 0].axvline(
        binary_edge,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Binary Boundary: ${binary_edge:,.0f}",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Entry-Level Binary Distribution (All Values)")
    axes[0, 0].set_xlabel("Final Judgement Amount ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Entry-level filtered data with binary boundary
    entry_lower_filt = entry_filtered[entry_filtered["binary_label"] == "Low (Green)"][
        "final_judgement_real"
    ]
    entry_higher_filt = entry_filtered[entry_filtered["binary_label"] == "High (Red)"][
        "final_judgement_real"
    ]

    axes[0, 1].hist(
        entry_lower_filt,
        bins=30,
        alpha=0.7,
        color=BINARY_COLORS["green"],
        label="Low (Green)",
        edgecolor="black",
    )
    axes[0, 1].hist(
        entry_higher_filt,
        bins=30,
        alpha=0.7,
        color=BINARY_COLORS["red"],
        label="High (Red)",
        edgecolor="black",
    )
    axes[0, 1].axvline(
        binary_edge,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Binary Boundary: ${binary_edge:,.0f}",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title(f"Entry-Level Binary Distribution (≤ ${max_threshold:,.0f})")
    axes[0, 1].set_xlabel("Final Judgement Amount ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Case-level full data with binary boundary
    axes[1, 0].hist(
        case_df[case_df["binary_label"] == "Low (Green)"]["mean_amount"],
        bins=20,
        alpha=0.7,
        color=BINARY_COLORS["green"],
        label="Low (Green)",
        edgecolor="black",
    )
    axes[1, 0].hist(
        case_df[case_df["binary_label"] == "High (Red)"]["mean_amount"],
        bins=20,
        alpha=0.7,
        color=BINARY_COLORS["red"],
        label="High (Red)",
        edgecolor="black",
    )
    axes[1, 0].axvline(
        binary_edge,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Binary Boundary: ${binary_edge:,.0f}",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Case-Level Binary Distribution (All Values)")
    axes[1, 0].set_xlabel("Mean Case Amount ($)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Case-level filtered data with binary boundary
    case_lower_filt = case_filtered[case_filtered["binary_label"] == "Low (Green)"][
        "mean_amount"
    ]
    case_higher_filt = case_filtered[case_filtered["binary_label"] == "High (Red)"][
        "mean_amount"
    ]

    axes[1, 1].hist(
        case_lower_filt,
        bins=20,
        alpha=0.7,
        color=BINARY_COLORS["green"],
        label="Low (Green)",
        edgecolor="black",
    )
    axes[1, 1].hist(
        case_higher_filt,
        bins=20,
        alpha=0.7,
        color=BINARY_COLORS["red"],
        label="High (Red)",
        edgecolor="black",
    )
    axes[1, 1].axvline(
        binary_edge,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Binary Boundary: ${binary_edge:,.0f}",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(f"Case-Level Binary Distribution (≤ ${max_threshold:,.0f})")
    axes[1, 1].set_xlabel("Mean Case Amount ($)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "binary_distribution_histograms.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Binary box plots comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Binary Distribution Comparison: Box Plots", fontsize=16, fontweight="bold"
    )

    # Entry-level binary box plot
    entry_lower = entry_df[entry_df["binary_label"] == "Low (Green)"][
        "final_judgement_real"
    ].values
    entry_higher = entry_df[entry_df["binary_label"] == "High (Red)"][
        "final_judgement_real"
    ].values

    box_data_entry = [entry_lower, entry_higher]
    box1 = axes[0].boxplot(
        box_data_entry, labels=["Low\n(Green)", "High\n(Red)"], patch_artist=True
    )
    box1["boxes"][0].set_facecolor(BINARY_COLORS["green"])
    box1["boxes"][1].set_facecolor(BINARY_COLORS["red"])
    axes[0].set_yscale("log")
    axes[0].set_title("Entry-Level Binary Amounts")
    axes[0].set_ylabel("Final Judgement Amount ($)")
    axes[0].grid(True, alpha=0.3)

    # Case-level binary box plot
    case_lower = case_df[case_df["binary_label"] == "Low (Green)"]["mean_amount"].values
    case_higher = case_df[case_df["binary_label"] == "High (Red)"]["mean_amount"].values

    box_data_case = [case_lower, case_higher]
    box2 = axes[1].boxplot(
        box_data_case, labels=["Low\n(Green)", "High\n(Red)"], patch_artist=True
    )
    box2["boxes"][0].set_facecolor(BINARY_COLORS["green"])
    box2["boxes"][1].set_facecolor(BINARY_COLORS["red"])
    axes[1].set_yscale("log")
    axes[1].set_title("Case-Level Binary Mean Amounts")
    axes[1].set_ylabel("Mean Case Amount ($)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "binary_distribution_boxplots.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Binary percentile analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Binary Percentile Analysis", fontsize=16, fontweight="bold")

    # Entry-level percentiles by binary class
    percentiles = range(0, 101, 5)
    entry_lower_pcts = [np.percentile(entry_lower, p) for p in percentiles]
    entry_higher_pcts = [np.percentile(entry_higher, p) for p in percentiles]

    axes[0].plot(
        percentiles,
        entry_lower_pcts,
        "o-",
        label="Low (Green)",
        linewidth=2,
        markersize=4,
        color=BINARY_COLORS["green"],
    )
    axes[0].plot(
        percentiles,
        entry_higher_pcts,
        "s-",
        label="High (Red)",
        linewidth=2,
        markersize=4,
        color=BINARY_COLORS["red"],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Entry-Level Binary Percentiles")
    axes[0].set_xlabel("Percentile")
    axes[0].set_ylabel("Amount ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Case-level percentiles by binary class
    case_lower_pcts = [np.percentile(case_lower, p) for p in percentiles]
    case_higher_pcts = [np.percentile(case_higher, p) for p in percentiles]

    axes[1].plot(
        percentiles,
        case_lower_pcts,
        "o-",
        label="Low (Green)",
        linewidth=2,
        markersize=4,
        color=BINARY_COLORS["green"],
    )
    axes[1].plot(
        percentiles,
        case_higher_pcts,
        "s-",
        label="High (Red)",
        linewidth=2,
        markersize=4,
        color=BINARY_COLORS["red"],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Case-Level Binary Percentiles")
    axes[1].set_xlabel("Percentile")
    axes[1].set_ylabel("Amount ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "binary_percentile_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Entries per case distribution by binary class
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Split by binary class
    case_lower_counts = case_df[case_df["binary_label"] == "Low (Green)"]["entry_count"]
    case_higher_counts = case_df[case_df["binary_label"] == "High (Red)"]["entry_count"]

    ax.hist(
        case_lower_counts,
        bins=30,
        alpha=0.7,
        color=BINARY_COLORS["green"],
        label="Low (Green)",
        edgecolor="black",
    )
    ax.hist(
        case_higher_counts,
        bins=30,
        alpha=0.7,
        color=BINARY_COLORS["red"],
        label="High (Red)",
        edgecolor="black",
    )
    ax.set_title(
        "Distribution of Entries per Case by Binary Class",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Entries per Case")
    ax.set_ylabel("Number of Cases")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text
    mean_lower = case_lower_counts.mean()
    mean_higher = case_higher_counts.mean()
    ax.axvline(
        mean_lower,
        color=BINARY_COLORS["green"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )
    ax.axvline(
        mean_higher,
        color=BINARY_COLORS["red"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )

    # Add text box with statistics
    stats_text = (
        f"Lower Risk Mean: {mean_lower:.1f}\nHigher Risk Mean: {mean_higher:.1f}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "binary_entries_per_case.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 5. Binary summary statistics comparison table plot
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("tight")
    ax.axis("off")

    # Create binary summary table data
    stats_data = []

    # Entry-level stats by binary class
    entry_lower_stats = entry_df[entry_df["binary_label"] == "Low (Green)"][
        "final_judgement_real"
    ]
    entry_higher_stats = entry_df[entry_df["binary_label"] == "High (Red)"][
        "final_judgement_real"
    ]

    stats_data.append(
        [
            "Entry Lower Risk (Green)",
            f"{len(entry_lower_stats):,}",
            f"${entry_lower_stats.mean():,.0f}",
            f"${entry_lower_stats.median():,.0f}",
            f"${entry_lower_stats.std():,.0f}",
            f"${entry_lower_stats.min():,.0f}",
            f"${entry_lower_stats.max():,.0f}",
        ]
    )

    stats_data.append(
        [
            "Entry Higher Risk (Red)",
            f"{len(entry_higher_stats):,}",
            f"${entry_higher_stats.mean():,.0f}",
            f"${entry_higher_stats.median():,.0f}",
            f"${entry_higher_stats.std():,.0f}",
            f"${entry_higher_stats.min():,.0f}",
            f"${entry_higher_stats.max():,.0f}",
        ]
    )

    # Case-level stats by binary class
    case_lower_stats = case_df[case_df["binary_label"] == "Low (Green)"]["mean_amount"]
    case_higher_stats = case_df[case_df["binary_label"] == "High (Red)"]["mean_amount"]

    stats_data.append(
        [
            "Case Lower Risk (Green)",
            f"{len(case_lower_stats):,}",
            f"${case_lower_stats.mean():,.0f}",
            f"${case_lower_stats.median():,.0f}",
            f"${case_lower_stats.std():,.0f}",
            f"${case_lower_stats.min():,.0f}",
            f"${case_lower_stats.max():,.0f}",
        ]
    )

    stats_data.append(
        [
            "Case Higher Risk (Red)",
            f"{len(case_higher_stats):,}",
            f"${case_higher_stats.mean():,.0f}",
            f"${case_higher_stats.median():,.0f}",
            f"${case_higher_stats.std():,.0f}",
            f"${case_higher_stats.min():,.0f}",
            f"${case_higher_stats.max():,.0f}",
        ]
    )

    # Create table
    table = ax.table(
        cellText=stats_data,
        colLabels=["Dataset", "Count", "Mean", "Median", "Std Dev", "Min", "Max"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style the table - header in green
    for i in range(7):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color rows by binary class
    table[(1, 0)].set_facecolor(BINARY_COLORS["green"])  # Lower risk row
    table[(1, 0)].set_text_props(weight="bold", color="white")
    table[(2, 0)].set_facecolor(BINARY_COLORS["red"])  # Higher risk row
    table[(2, 0)].set_text_props(weight="bold", color="white")
    table[(3, 0)].set_facecolor(BINARY_COLORS["green"])  # Lower risk row
    table[(3, 0)].set_text_props(weight="bold", color="white")
    table[(4, 0)].set_facecolor(BINARY_COLORS["red"])  # Higher risk row
    table[(4, 0)].set_text_props(weight="bold", color="white")

    ax.set_title(
        f"Binary Summary Statistics Comparison\nBinary Boundary: ${binary_edge:,.0f}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "binary_summary_statistics_table.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Generated 5 binary visualization plots in: {output_dir}")


def main():
    """Main binary visualization function."""
    parser = argparse.ArgumentParser(
        description="Create binary distribution plots for case outcome results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--output-dir",
        type=Path,
        default=Path("plots/outcomes_binary"),
        help="Output directory for plots (default: plots/outcomes_binary)",
    )

    parser.add_argument(
        "--max-threshold",
        type=float,
        default=5_000_000_000.00,
        help="Maximum threshold for filtering (default: 5,000,000,000.00)",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.input_data.exists():
        print(f"Error: Input data file not found: {args.input_data}")
        sys.exit(1)

    if not args.kfold_dir.exists():
        print(f"Error: K-fold directory not found: {args.kfold_dir}")
        sys.exit(1)

    print("Corporate Speech Risk Dataset - Binary Outcome Distribution Plots")
    print("=" * 60)
    print(f"Input Data: {args.input_data}")
    print(f"K-fold Directory: {args.kfold_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Threshold: ${args.max_threshold:,.2f}")

    try:
        # Load data
        entry_df, case_df, binary_edge = load_binary_outcome_data(
            args.input_data, args.kfold_dir
        )

        if len(entry_df) == 0:
            print("Error: No valid outcome data found")
            sys.exit(1)

        # Create plots
        create_binary_distribution_plots(
            entry_df, case_df, binary_edge, args.output_dir, args.max_threshold
        )

        print(f"\nBinary visualization completed successfully!")
        print(f"View plots in: {args.output_dir}")

    except Exception as e:
        print(f"\nError during binary visualization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
