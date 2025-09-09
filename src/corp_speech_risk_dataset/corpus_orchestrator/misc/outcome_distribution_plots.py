#!/usr/bin/env python3
"""
outcome_distribution_plots.py

Creates visualization plots for the case outcome imputation results distribution.
Generates histograms, box plots, and distribution comparisons for analysis.

Usage:
    python scripts/outcome_distribution_plots.py \
        --outcomes-dir data/outcomes/courtlistener_v1 \
        --output-dir plots/outcomes \
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
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_outcome_data(outcomes_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load outcome data (reuse from summary statistics script)."""
    print(f"Loading outcome data from: {outcomes_dir}")

    entry_data = []
    case_data = defaultdict(list)

    # Find all JSONL files
    jsonl_files = list(outcomes_dir.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files to process")

    for jsonl_path in jsonl_files:
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
                                    "final_judgement_real": float(final_amount),
                                }
                            )
                            case_data[case_name].append(float(final_amount))

                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue

        except Exception:
            continue

    # Create DataFrames
    entry_df = pd.DataFrame(entry_data)

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
            }
        )

    case_df = pd.DataFrame(case_aggregates)

    print(f"Loaded {len(entry_df)} entries from {len(case_df)} cases")
    return entry_df, case_df


def create_distribution_plots(
    entry_df: pd.DataFrame,
    case_df: pd.DataFrame,
    output_dir: Path,
    max_threshold: float,
):
    """Create comprehensive distribution plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter data
    entry_filtered = entry_df[entry_df["final_judgement_real"] <= max_threshold].copy()
    case_filtered = case_df[case_df["mean_amount"] <= max_threshold].copy()

    # 1. Log-scale histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Distribution of Final Judgement Amounts", fontsize=16, fontweight="bold"
    )

    # Entry-level full data
    axes[0, 0].hist(
        entry_df["final_judgement_real"],
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Entry-Level Distribution (All Values)")
    axes[0, 0].set_xlabel("Final Judgement Amount ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # Entry-level filtered data
    axes[0, 1].hist(
        entry_filtered["final_judgement_real"],
        bins=50,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title(f"Entry-Level Distribution (≤ ${max_threshold:,.0f})")
    axes[0, 1].set_xlabel("Final Judgement Amount ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Case-level full data
    axes[1, 0].hist(
        case_df["mean_amount"],
        bins=30,
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Case-Level Distribution (All Values)")
    axes[1, 0].set_xlabel("Mean Case Amount ($)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # Case-level filtered data
    axes[1, 1].hist(
        case_filtered["mean_amount"],
        bins=30,
        alpha=0.7,
        color="gold",
        edgecolor="black",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(f"Case-Level Distribution (≤ ${max_threshold:,.0f})")
    axes[1, 1].set_xlabel("Mean Case Amount ($)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "distribution_histograms.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Box plots comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Distribution Comparison: Box Plots", fontsize=16, fontweight="bold")

    # Entry-level box plot
    box_data_entry = [
        entry_df["final_judgement_real"].values,
        entry_filtered["final_judgement_real"].values,
    ]
    box1 = axes[0].boxplot(
        box_data_entry, labels=["All Values", "Filtered"], patch_artist=True
    )
    box1["boxes"][0].set_facecolor("skyblue")
    box1["boxes"][1].set_facecolor("lightcoral")
    axes[0].set_yscale("log")
    axes[0].set_title("Entry-Level Amounts")
    axes[0].set_ylabel("Final Judgement Amount ($)")
    axes[0].grid(True, alpha=0.3)

    # Case-level box plot
    box_data_case = [case_df["mean_amount"].values, case_filtered["mean_amount"].values]
    box2 = axes[1].boxplot(
        box_data_case, labels=["All Values", "Filtered"], patch_artist=True
    )
    box2["boxes"][0].set_facecolor("lightgreen")
    box2["boxes"][1].set_facecolor("gold")
    axes[1].set_yscale("log")
    axes[1].set_title("Case-Level Mean Amounts")
    axes[1].set_ylabel("Mean Case Amount ($)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Percentile analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Percentile Analysis", fontsize=16, fontweight="bold")

    # Entry-level percentiles
    percentiles = range(0, 101, 5)
    entry_pcts_all = [
        np.percentile(entry_df["final_judgement_real"], p) for p in percentiles
    ]
    entry_pcts_filt = [
        np.percentile(entry_filtered["final_judgement_real"], p) for p in percentiles
    ]

    axes[0].plot(
        percentiles, entry_pcts_all, "o-", label="All Values", linewidth=2, markersize=4
    )
    axes[0].plot(
        percentiles, entry_pcts_filt, "s-", label="Filtered", linewidth=2, markersize=4
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Entry-Level Percentiles")
    axes[0].set_xlabel("Percentile")
    axes[0].set_ylabel("Amount ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Case-level percentiles
    case_pcts_all = [np.percentile(case_df["mean_amount"], p) for p in percentiles]
    case_pcts_filt = [
        np.percentile(case_filtered["mean_amount"], p) for p in percentiles
    ]

    axes[1].plot(
        percentiles, case_pcts_all, "o-", label="All Values", linewidth=2, markersize=4
    )
    axes[1].plot(
        percentiles, case_pcts_filt, "s-", label="Filtered", linewidth=2, markersize=4
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Case-Level Percentiles")
    axes[1].set_xlabel("Percentile")
    axes[1].set_ylabel("Amount ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "percentile_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Entries per case distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(
        case_df["entry_count"], bins=30, alpha=0.7, color="purple", edgecolor="black"
    )
    ax.set_title("Distribution of Entries per Case", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Entries per Case")
    ax.set_ylabel("Number of Cases")
    ax.grid(True, alpha=0.3)

    # Add statistics text
    mean_entries = case_df["entry_count"].mean()
    median_entries = case_df["entry_count"].median()
    ax.axvline(
        mean_entries,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_entries:.1f}",
    )
    ax.axvline(
        median_entries,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_entries:.1f}",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "entries_per_case.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Summary statistics comparison table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    # Create summary table data
    stats_data = []

    # Entry-level stats
    entry_all = entry_df["final_judgement_real"]
    entry_filt = entry_filtered["final_judgement_real"]

    stats_data.append(
        [
            "Entry-Level (All)",
            f"{len(entry_all):,}",
            f"${entry_all.mean():,.0f}",
            f"${entry_all.median():,.0f}",
            f"${entry_all.std():,.0f}",
            f"${entry_all.min():,.0f}",
            f"${entry_all.max():,.0f}",
        ]
    )

    stats_data.append(
        [
            "Entry-Level (Filtered)",
            f"{len(entry_filt):,}",
            f"${entry_filt.mean():,.0f}",
            f"${entry_filt.median():,.0f}",
            f"${entry_filt.std():,.0f}",
            f"${entry_filt.min():,.0f}",
            f"${entry_filt.max():,.0f}",
        ]
    )

    # Case-level stats
    case_all = case_df["mean_amount"]
    case_filt = case_filtered["mean_amount"]

    stats_data.append(
        [
            "Case-Level (All)",
            f"{len(case_all):,}",
            f"${case_all.mean():,.0f}",
            f"${case_all.median():,.0f}",
            f"${case_all.std():,.0f}",
            f"${case_all.min():,.0f}",
            f"${case_all.max():,.0f}",
        ]
    )

    stats_data.append(
        [
            "Case-Level (Filtered)",
            f"{len(case_filt):,}",
            f"${case_filt.mean():,.0f}",
            f"${case_filt.median():,.0f}",
            f"${case_filt.std():,.0f}",
            f"${case_filt.min():,.0f}",
            f"${case_filt.max():,.0f}",
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
    table.scale(1.2, 1.5)

    # Style the table
    table[(0, 0)].set_facecolor("#4CAF50")
    table[(0, 1)].set_facecolor("#4CAF50")
    table[(0, 2)].set_facecolor("#4CAF50")
    table[(0, 3)].set_facecolor("#4CAF50")
    table[(0, 4)].set_facecolor("#4CAF50")
    table[(0, 5)].set_facecolor("#4CAF50")
    table[(0, 6)].set_facecolor("#4CAF50")

    ax.set_title(
        "Summary Statistics Comparison", fontsize=16, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "summary_statistics_table.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Generated 5 visualization plots in: {output_dir}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Create distribution plots for case outcome results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--outcomes-dir",
        type=Path,
        default=Path("data/outcomes/courtlistener_v1"),
        help="Path to outcomes directory (default: data/outcomes/courtlistener_v1)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/outcomes"),
        help="Output directory for plots (default: plots/outcomes)",
    )

    parser.add_argument(
        "--max-threshold",
        type=float,
        default=5_000_000_000.00,
        help="Maximum threshold for filtering (default: 5,000,000,000.00)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.outcomes_dir.exists():
        print(f"Error: Outcomes directory not found: {args.outcomes_dir}")
        sys.exit(1)

    print("Corporate Speech Risk Dataset - Outcome Distribution Plots")
    print("=" * 60)
    print(f"Outcomes Directory: {args.outcomes_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Threshold: ${args.max_threshold:,.2f}")

    try:
        # Load data
        entry_df, case_df = load_outcome_data(args.outcomes_dir)

        if len(entry_df) == 0:
            print("Error: No valid outcome data found")
            sys.exit(1)

        # Create plots
        create_distribution_plots(
            entry_df, case_df, args.output_dir, args.max_threshold
        )

        print(f"\nVisualization completed successfully!")
        print(f"View plots in: {args.output_dir}")

    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
