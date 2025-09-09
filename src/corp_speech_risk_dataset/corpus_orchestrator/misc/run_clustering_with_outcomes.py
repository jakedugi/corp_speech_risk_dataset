#!/usr/bin/env python3
"""
Script to run clustering pipeline on outcome data with fused embeddings.

This script:
1. Loads JSONL data with 'fused_emb' and 'final_judgement_real' fields
2. Converts to the format expected by ClusterPipeline
3. Runs clustering with percentile bucket supervision
4. Generates interactive HTML visualization

Usage:
    python scripts/run_clustering_with_outcomes.py \
        --input data/outcomes/courtlistener_v1/0:15-cv-62604_flsd/doc_4101961_text_stage9.jsonl \
        --output clusters_outcomes.html \
        --min-cluster-size 5 \
        --threshold 1000000

    # For multiple files:
    python scripts/run_clustering_with_outcomes.py \
        --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" \
        --output clusters_all_outcomes.html \
        --min-cluster-size 10 \
        --threshold 5000000
"""

import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import tempfile

import numpy as np
from corp_speech_risk_dataset.clustering.pipeline import ClusterPipeline


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


def filter_by_threshold(
    data: List[Dict[str, Any]], threshold: float = None
) -> List[Dict[str, Any]]:
    """Filter out records above threshold if specified."""
    if threshold is None:
        return data

    filtered = []
    for record in data:
        amount = record.get("final_judgement_real")
        if amount is None or amount <= threshold:
            filtered.append(record)

    print(
        f"Filtered {len(data)} -> {len(filtered)} records (threshold: ${threshold:,.0f})"
    )
    return filtered


def prepare_pipeline_data(data: List[Dict[str, Any]]) -> tuple[str, str]:
    """
    Convert JSONL data to ClusterPipeline format.

    Returns:
        Tuple of (vector_file_path, metadata_file_path)
    """
    # Extract embeddings and create metadata
    embeddings = []
    metadata = []

    for i, record in enumerate(data):
        # Extract fused embedding
        fused_emb = record["fused_emb"]
        embeddings.append(fused_emb)

        # Create metadata entry in expected format
        meta_entry = {
            "text": record.get("text", f"Record {i}"),
            "sp_ids": record.get("sp_ids", []),  # Required by decoder
            "doc_id": record.get("doc_id", f"doc_{i}"),
            "final_judgement_real": record.get("final_judgement_real"),
            "speaker": record.get("speaker", ""),
            "score": record.get("score", 0.0),
            "_src": record.get("_src", ""),
        }
        metadata.append(meta_entry)

    # Convert to numpy and save temporarily
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Create temporary files
    temp_dir = Path(tempfile.mkdtemp())
    vec_path = temp_dir / "embeddings.npy"
    meta_path = temp_dir / "metadata.json"

    # Save data
    np.save(vec_path, embeddings_array)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prepared data: {embeddings_array.shape} embeddings")
    print(f"Vector file: {vec_path}")
    print(f"Metadata file: {meta_path}")

    return str(vec_path), str(meta_path)


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
        print(f"25th percentile: ${np.percentile(amounts, 25):,.2f}")
        print(f"75th percentile: ${np.percentile(amounts, 75):,.2f}")
        print(f"33rd percentile: ${np.percentile(amounts, 33.3):,.2f}")
        print(f"67th percentile: ${np.percentile(amounts, 66.6):,.2f}")


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
        default="clusters_outcomes.html",
        help="Output HTML file for visualization",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Exclude outcomes above this dollar amount",
    )
    parser.add_argument(
        "--supervision",
        choices=["categorical", "continuous"],
        default="categorical",
        help="UMAP supervision mode",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU acceleration if available"
    )

    args = parser.parse_args()

    # Load and process data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data = load_jsonl_files(args.input)

    if not data:
        print("ERROR: No data loaded!")
        return

    # Analyze distribution
    analyze_outcome_distribution(data)

    # Apply threshold filter
    if args.threshold:
        data = filter_by_threshold(data, args.threshold)

    # Prepare data for pipeline
    print("\n" + "=" * 60)
    print("PREPARING PIPELINE DATA")
    print("=" * 60)
    vec_path, meta_path = prepare_pipeline_data(data)

    # Run clustering pipeline
    print("\n" + "=" * 60)
    print("RUNNING CLUSTERING PIPELINE")
    print("=" * 60)

    pipeline = ClusterPipeline(
        vec_path=vec_path,
        meta_path=meta_path,
        use_gpu=args.gpu,
        min_cluster_size=args.min_cluster_size,
        supervision_mode=args.supervision,
    )

    print("[1/4] Building FAISS index...")
    pipeline.build()

    print("[2/4] Clustering with HDBSCAN...")
    labels = pipeline.cluster()
    n_clusters = len(set(labels) - {-1})
    print(f"[✓] Found {n_clusters} clusters (noise label = -1)")

    print("[3/4] Reducing dimensions for visualization...")
    coords = pipeline.reduce()

    # Calculate clustering metrics
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    if n_clusters > 1:
        sil = silhouette_score(coords, labels, metric="euclidean")
        ari = adjusted_rand_score(pipeline.buckets, labels)
        print(f"[✓] Silhouette Score: {sil:.3f}")
        print(f"[✓] Adjusted Rand Index: {ari:.3f}")

    print("[4/4] Creating visualization...")
    output_path = pipeline.visualise(args.output)
    print(f"[✓] Visualization saved to: {output_path}")

    # Print bucket statistics
    print("\n" + "=" * 60)
    print("BUCKET STATISTICS")
    print("=" * 60)
    from collections import Counter

    bucket_counts = Counter(pipeline.buckets)
    for bucket, count in bucket_counts.items():
        pct = 100 * count / len(pipeline.buckets)
        print(f"{bucket:>8}: {count:>6} ({pct:5.1f}%)")

    # Cleanup temporary files
    Path(vec_path).unlink()
    Path(meta_path).unlink()
    Path(vec_path).parent.rmdir()

    print(f"\n🎉 Complete! Open {output_path} in your browser to explore the clusters.")


if __name__ == "__main__":
    main()
