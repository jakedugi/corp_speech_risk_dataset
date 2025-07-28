#!/usr/bin/env python3
"""
summary_y_stats.py

CLI to compute rich summary statistics for a numeric field across different data structures.

Features:
  - Positional argument for metadata JSON/JSONL path or directory
  - Optional --field to specify which numeric column to summarize (default: final_judgement_real)
  - Optional --output-format {text,json}
  - Reports total entries, non-null count, missing count/rate,
    mean, median, std dev, min/25%/50%/75%/max, skewness, kurtosis
  - Supports: JSON files, JSONL files, and directory scanning
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd


def load_data_from_source(source: Path) -> List[Dict[str, Any]]:
    """
    Load data from various sources: JSON file, JSONL file, or directory.

    Args:
        source: Path to JSON file, JSONL file, or directory

    Returns:
        List of dictionaries containing the data
    """
    records = []

    if source.is_file():
        try:
            if source.suffix.lower() == ".jsonl":
                # Handle JSONL files
                with open(source, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(
                                f"Warning: Invalid JSON on line {line_num} in {source}: {e}",
                                file=sys.stderr,
                            )
            else:
                # Handle JSON files
                content = source.read_text(encoding="utf-8")
                data = json.loads(content)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)
                else:
                    raise ValueError(
                        f"JSON file must contain an object or array, got {type(data)}"
                    )
        except Exception as e:
            print(f"Error reading file {source}: {e}", file=sys.stderr)

    elif source.is_dir():
        # Handle directory - recursively find JSON/JSONL files
        json_files = list(source.rglob("*.json")) + list(source.rglob("*.jsonl"))
        print(f"Found {len(json_files)} JSON/JSONL files in {source}", file=sys.stderr)

        for file_path in json_files:
            file_records = load_data_from_source(file_path)
            records.extend(file_records)

    else:
        raise FileNotFoundError(f"Source {source} does not exist")

    return records


def compute_summary_stats(df: pd.DataFrame, field: str) -> Dict[str, Union[int, float]]:
    """Compute summary stats for a numeric series in a DataFrame."""
    if field not in df:
        raise KeyError(f"Field '{field}' not found in metadata.")

    # Coerce to numeric, convert non-coercible to NaN
    y = pd.to_numeric(df[field], errors="coerce")
    assert isinstance(y, pd.Series), "Expected pandas Series"
    total = len(y)
    non_null = int(y.count())
    missing = total - non_null
    missing_rate = missing / total if total > 0 else np.nan

    stats: Dict[str, Union[int, float]] = {
        "total_entries": total,
        "count_non_null": non_null,
        "missing_count": missing,
        "missing_rate": missing_rate,
    }

    if non_null > 0:
        desc = y.describe()  # count, mean, std, min, 25%, 50%, 75%, max
        stats.update(
            {
                "mean": float(desc["mean"]),
                "std_dev": float(desc["std"]),
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
                "skewness": float(y.skew()),
                "kurtosis": float(y.kurt()),
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
        description="Compute summary stats for a numeric field across JSON/JSONL files or directories"
    )
    p.add_argument(
        "source",
        type=Path,
        help="Path to JSON file, JSONL file, or directory containing JSON/JSONL files",
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
        records = load_data_from_source(args.source)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("No data found to analyze", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.source}", file=sys.stderr)
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
        print(
            f"\nSummary statistics for '{args.field}' across {len(records)} records:\n"
        )
        print(f"  Total entries      : {stats['total_entries']}")
        print(f"  Non-null count     : {stats['count_non_null']}")
        print(f"  Missing count      : {stats['missing_count']}")
        print(f"  Missing rate       : {stats['missing_rate']:.4f}\n")

        if stats["count_non_null"] > 0:
            print("  Descriptive stats for non-missing values:")
            print(f"    Mean             : ${stats['mean']:,.2f}")
            print(f"    Std dev          : ${stats['std_dev']:,.2f}")
            print(f"    Min              : ${stats['min']:,.2f}")
            print(f"    25%              : ${stats['25%']:,.2f}")
            print(f"    Median (50%)     : ${stats['50%']:,.2f}")
            print(f"    75%              : ${stats['75%']:,.2f}")
            print(f"    Max              : ${stats['max']:,.2f}")
            print(f"    Skewness         : {stats['skewness']:.4f}")
            print(f"    Kurtosis         : {stats['kurtosis']:.4f}")
        else:
            print("  No non-missing values to summarize.\n")


if __name__ == "__main__":
    main()
