#!/usr/bin/env python3
"""
Visualize embedding similarity in 2D space, color-coded by outcome buckets.

This script:
1. Loads JSONL data with 'fused_emb' and 'final_judgement_real' fields
2. Projects embeddings to 2D using UMAP (unsupervised - no clustering bias)
3. Colors points by outcome percentile buckets (low/med/high/missing)
4. Filters by speakers and outcome thresholds
5. Generates interactive HTML visualization showing pure similarity patterns

Usage:
    python scripts/visualize_similarity_by_outcomes.py \
        --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" \
        --output similarity_by_outcomes.html \
        --max-threshold 15500000000 \
        --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA"
"""

import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import umap


def load_jsonl_files(pattern: str) -> List[Dict[str, Any]]:
    """Load data from JSONL files matching the pattern."""
    files = glob.glob(pattern) if "*" in pattern else [pattern]

    all_data = []
    for file_path in files:
        print(f"Loading {file_path}...")
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                if "fused_emb" in data and "final_judgement_real" in data:
                    all_data.append(data)

    print(f"Loaded {len(all_data)} records from {len(files)} files")
    return all_data


def filter_by_threshold_and_speakers(
    data: List[Dict[str, Any]],
    max_threshold: float = None,
    exclude_speakers: List[str] = None,
    exclude_missing: bool = False,
) -> List[Dict[str, Any]]:
    """Filter by outcome threshold, exclude specific speakers, and optionally exclude missing outcomes."""
    if max_threshold is None and not exclude_speakers and not exclude_missing:
        return data

    filtered = []
    excluded_speakers = set(exclude_speakers or [])

    for record in data:
        # Filter by speaker
        speaker = record.get("speaker", "")
        if speaker in excluded_speakers:
            continue

        # Filter by missing outcomes
        amount = record.get("final_judgement_real")
        if exclude_missing and amount is None:
            continue

        # Filter by threshold
        if max_threshold is not None and amount is not None and amount > max_threshold:
            continue

        filtered.append(record)

    original_count = len(data)
    speaker_filter_info = (
        f" (excluded speakers: {exclude_speakers})" if exclude_speakers else ""
    )
    threshold_info = f" (max threshold: ${max_threshold:,.0f})" if max_threshold else ""
    missing_filter_info = " (excluded missing outcomes)" if exclude_missing else ""

    print(
        f"Filtered {original_count} -> {len(filtered)} records{speaker_filter_info}{threshold_info}{missing_filter_info}"
    )
    return filtered


def create_outcome_buckets(amounts: np.ndarray) -> tuple[np.ndarray, dict]:
    """Create tercile buckets (33rd/67th percentiles) for outcome amounts."""
    valid_amounts = amounts[~np.isnan(amounts)]

    if len(valid_amounts) < 3:
        # Not enough data for terciles
        return np.full(len(amounts), "insufficient_data"), {}

    q33, q67 = np.percentile(valid_amounts, [33.33, 66.67])

    buckets = np.where(
        np.isnan(amounts),
        "missing",
        np.select([amounts < q33, amounts < q67], ["low", "medium"], "high"),
    )

    bucket_info = {
        "low": {
            "range": f"${valid_amounts.min():,.2f} - ${q33:,.2f}",
            "percentile": "0-33rd",
        },
        "medium": {"range": f"${q33:,.2f} - ${q67:,.2f}", "percentile": "33-67th"},
        "high": {
            "range": f"${q67:,.2f} - ${valid_amounts.max():,.2f}",
            "percentile": "67-100th",
        },
        "missing": {"range": "No outcome data", "percentile": "N/A"},
    }

    return buckets, bucket_info


def analyze_outcome_distribution(data: List[Dict[str, Any]]) -> None:
    """Print analysis of outcome value distribution."""
    amounts = []
    missing_count = 0

    for record in data:
        amount = record.get("final_judgement_real")
        if amount is None:
            missing_count += 1
        else:
            amounts.append(amount)

    if amounts:
        amounts = np.array(amounts)
        print(f"\nOutcome Distribution Analysis:")
        print(f"Total records: {len(data)}")
        print(f"Missing values: {missing_count}")
        print(f"Valid amounts: {len(amounts)}")
        print(f"Min: ${amounts.min():,.2f}")
        print(f"Max: ${amounts.max():,.2f}")
        print(f"Median: ${np.median(amounts):,.2f}")
        print(f"33rd percentile: ${np.percentile(amounts, 33.33):,.2f}")
        print(f"67th percentile: ${np.percentile(amounts, 66.67):,.2f}")


def create_similarity_visualization(
    data: List[Dict[str, Any]], output_path: str = "similarity_by_outcomes.html"
) -> str:
    """Create 2D similarity visualization colored by outcome buckets."""

    print("\n" + "=" * 60)
    print("CREATING SIMILARITY VISUALIZATION")
    print("=" * 60)

    # Extract embeddings and outcomes
    embeddings = np.array([d["fused_emb"] for d in data], dtype=np.float32)
    amounts = np.array([d.get("final_judgement_real") for d in data], dtype=np.float64)

    # Handle missing values
    amounts = np.where([a is None for a in amounts], np.nan, amounts)

    print(f"Processing {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")

    # Create outcome buckets
    buckets, bucket_info = create_outcome_buckets(amounts)

    # Print bucket statistics
    unique_buckets, bucket_counts = np.unique(buckets, return_counts=True)
    print(f"\nBucket Distribution:")
    for bucket, count in zip(unique_buckets, bucket_counts):
        pct = 100 * count / len(buckets)
        info = bucket_info.get(bucket, {})
        range_str = info.get("range", "Unknown")
        percentile_str = info.get("percentile", "Unknown")
        print(f"{bucket:>8}: {count:>6} ({pct:5.1f}%) | {percentile_str} | {range_str}")

    # Normalize embeddings for better UMAP performance
    print("\nNormalizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Create 2D projection with UMAP (unsupervised - pure similarity)
    print("Projecting to 2D with UMAP (unsupervised)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=True,
    )

    coords_2d = reducer.fit_transform(embeddings_scaled)
    print(f"2D projection complete: {coords_2d.shape}")

    # Create DataFrame for visualization
    df = pd.DataFrame(
        {
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "outcome_bucket": buckets,
            "outcome_amount": amounts,
            "text": [
                (
                    d.get("text", "")[:100] + "..."
                    if len(d.get("text", "")) > 100
                    else d.get("text", "")
                )
                for d in data
            ],
            "speaker": [d.get("speaker", "") for d in data],
            "doc_id": [d.get("doc_id", "") for d in data],
            "score": [d.get("score", 0.0) for d in data],
        }
    )

    # Create interactive scatter plot
    print("Creating interactive visualization...")

    # Define colors for buckets
    color_map = {
        "low": "#2E86AB",  # Blue
        "medium": "#A23B72",  # Purple
        "high": "#F18F01",  # Orange
        "missing": "#C73E1D",  # Red
        "insufficient_data": "#808080",  # Gray
    }

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="outcome_bucket",
        color_discrete_map=color_map,
        hover_data={
            "text": True,
            "speaker": True,
            "doc_id": True,
            "outcome_amount": ":.2f",
            "score": ":.3f",
            "x": False,
            "y": False,
        },
        labels={
            "outcome_bucket": "Outcome Bucket",
            "x": "UMAP Dimension 1",
            "y": "UMAP Dimension 2",
        },
        title="Corporate Speech Embeddings: Similarity Clustering by Outcome Buckets",
        width=1200,
        height=800,
    )

    # Customize layout
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    # Add bucket info as annotations
    annotation_text = "<br>".join(
        [
            f"<b>{bucket.title()}</b>: {info['percentile']} percentile"
            for bucket, info in bucket_info.items()
            if bucket in unique_buckets and bucket != "insufficient_data"
        ]
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    # Save visualization
    output_path = Path(output_path)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Visualization saved to: {output_path.absolute()}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL file(s) with fused_emb and final_judgement_real (supports globs)",
    )
    parser.add_argument(
        "--output",
        default="similarity_by_outcomes.html",
        help="Output HTML file for visualization",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=None,
        help="Exclude outcomes above this dollar amount",
    )
    parser.add_argument(
        "--exclude-speakers",
        type=str,
        default="Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA",
        help="Comma-separated list of speakers to exclude",
    )
    parser.add_argument(
        "--exclude-missing",
        action="store_true",
        help="Exclude records with missing outcome data",
    )

    args = parser.parse_args()

    # Parse excluded speakers
    exclude_speakers = (
        [s.strip() for s in args.exclude_speakers.split(",") if s.strip()]
        if args.exclude_speakers
        else []
    )

    # Load and process data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data = load_jsonl_files(args.input)

    if not data:
        print("ERROR: No data loaded!")
        return

    # Analyze distribution before filtering
    analyze_outcome_distribution(data)

    # Apply filters
    data = filter_by_threshold_and_speakers(
        data, args.max_threshold, exclude_speakers, args.exclude_missing
    )

    if not data:
        print("ERROR: No data remaining after filtering!")
        return

    # Analyze distribution after filtering
    print("\nAfter filtering:")
    analyze_outcome_distribution(data)

    # Create visualization
    output_path = create_similarity_visualization(data, args.output)
    print(
        f"\n🎉 Complete! Open {output_path} in your browser to explore the similarity patterns."
    )


if __name__ == "__main__":
    main()
