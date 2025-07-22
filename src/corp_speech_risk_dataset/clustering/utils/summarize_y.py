#!/usr/bin/env python3
import json
import pandas as pd
import argparse
from pathlib import Path


def compute_summary_stats(meta_path: Path):
    # Load metadata.json
    with meta_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    # Build DataFrame and extract the target column
    df = pd.DataFrame(records)
    y = df["final_judgement_real"]

    # Compute summary statistics
    summary = {
        "count_non_null": y.count(),
        "missing_count": int(y.isna().sum()),
        "missing_rate": float(y.isna().mean()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std_dev": float(y.std()),
        "skewness": float(y.skew()),
        "kurtosis": float(y.kurtosis()),
    }

    # Print results
    print("\nSummary statistics for 'final_judgement_real':")
    for k, v in summary.items():
        print(f"  {k:15s}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute summary stats for final_judgement_real in metadata.json"
    )
    parser.add_argument(
        "metadata", type=Path, help="Path to data/clustering/metadata.json"
    )
    args = parser.parse_args()
    compute_summary_stats(args.metadata)
