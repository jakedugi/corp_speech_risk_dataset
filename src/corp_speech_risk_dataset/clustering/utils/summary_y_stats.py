#!/usr/bin/env python3
"""
summary_y_stats.py

CLI to compute rich summary statistics for a numeric field in a metadata JSON file.

Features:
  - Positional argument for metadata JSON path
  - Optional --field to specify which numeric column to summarize (default: final_judgement_real)
  - Optional --output-format {text,json}
  - Reports total entries, non-null count, missing count/rate,
    mean, median, std dev, min/25%/50%/75%/max, skewness, kurtosis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compute_summary_stats(df: pd.DataFrame, field: str) -> dict:
    """Compute summary stats for a numeric series in a DataFrame."""
    if field not in df:
        raise KeyError(f"Field '{field}' not found in metadata.")
    # Coerce to numeric, convert non-coercible to NaN
    y = pd.to_numeric(df[field], errors="coerce")
    total = len(y)
    non_null = int(y.count())
    missing = total - non_null
    missing_rate = missing / total if total > 0 else np.nan

    stats = {
        "total_entries": total,
        "count_non_null": non_null,
        "missing_count": missing,
        "missing_rate": missing_rate,
    }

    if non_null > 0:
        desc = y.describe()  # count, mean, std, min, 25%, 50%, 75%, max
        stats.update(
            {
                "mean": desc["mean"],
                "std_dev": desc["std"],
                "min": desc["min"],
                "25%": desc["25%"],
                "50%": desc["50%"],
                "75%": desc["75%"],
                "max": desc["max"],
                "skewness": y.skew(),
                "kurtosis": y.kurt(),
            }
        )
    else:
        # Fill numeric stats with NaN if no valid values
        for key in [
            "mean",
            "std_dev",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "skewness",
            "kurtosis",
        ]:
            stats[key] = np.nan

    return stats


def main():
    p = argparse.ArgumentParser(
        description="Compute summary stats for a numeric field in metadata JSON"
    )
    p.add_argument(
        "metadata",
        type=Path,
        help="Path to metadata JSON (list of dicts)",
    )
    p.add_argument(
        "--field",
        "-f",
        default="final_judgement_real",
        help="Name of numeric field to summarize (default: final_judgement_real)",
    )
    p.add_argument(
        "--output-format",
        "-o",
        choices=["text", "json"],
        default="text",
        help="Format of the output (default: text)",
    )
    args = p.parse_args()

    try:
        records = json.loads(args.metadata.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading metadata JSON: {e}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    try:
        stats = compute_summary_stats(df, args.field)
    except KeyError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if args.output_format == "json":
        # Ensure JSON is serializable
        out = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in stats.items()
        }
        print(json.dumps(out, indent=2))
    else:
        # Pretty-text output
        print(f"\nSummary statistics for '{args.field}':\n")
        print(f"  Total entries      : {stats['total_entries']}")
        print(f"  Non-null count     : {stats['count_non_null']}")
        print(f"  Missing count      : {stats['missing_count']}")
        print(f"  Missing rate       : {stats['missing_rate']:.4f}\n")

        if stats["count_non_null"] > 0:
            print("  Descriptive stats for non-missing values:")
            print(f"    Mean             : {stats['mean']:.4f}")
            print(f"    Std dev          : {stats['std_dev']:.4f}")
            print(f"    Min              : {stats['min']:.4f}")
            print(f"    25%              : {stats['25%']:.4f}")
            print(f"    Median (50%)     : {stats['50%']:.4f}")
            print(f"    75%              : {stats['75%']:.4f}")
            print(f"    Max              : {stats['max']:.4f}")
            print(f"    Skewness         : {stats['skewness']:.4f}")
            print(f"    Kurtosis         : {stats['kurtosis']:.4f}")
        else:
            print("  No non-missing values to summarize.\n")


if __name__ == "__main__":
    main()
