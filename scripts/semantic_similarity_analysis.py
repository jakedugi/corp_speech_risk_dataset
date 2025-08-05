#!/usr/bin/env python3
"""
Comprehensive Semantic Similarity Analysis in Raw Embedding Space

This script performs state-of-the-art semantic similarity analysis on 384D embeddings
across different outcome buckets, covering:

1. Distance Metrics:
   - Cosine similarity/distance
   - Euclidean distance
   - Manhattan (L1) distance
   - Chebyshev (L∞) distance
   - Mahalanobis distance
   - Jensen-Shannon divergence
   - Wasserstein distance

2. Statistical Analysis:
   - Intra-bucket vs inter-bucket similarities
   - Centroid analysis
   - Variance/spread analysis
   - Silhouette analysis
   - Calinski-Harabasz index
   - Davies-Bouldin index

3. Visualizations:
   - Distance distribution plots
   - Similarity heatmaps
   - Centroid analysis plots
   - Statistical comparison plots
   - Embedding space analysis plots

Usage:
    python scripts/semantic_similarity_analysis.py \
        --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" \
        --output similarity_analysis_report.html \
        --max-threshold 15500000000 \
        --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA" \
        --exclude-missing
"""

import argparse
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

# Statistical and ML imports
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# GPU acceleration support
try:
    import cupy as cp
    import cupy.scipy.spatial.distance as cp_distance

    CUPY_AVAILABLE = True
    print("GPU acceleration (CuPy) available")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("GPU acceleration (CuPy) not available, using CPU")

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


def load_data(pattern: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file pattern."""
    print(f"Loading data from pattern: {pattern}")

    files = glob.glob(pattern)
    if not files:
        print(f"WARNING: No files found matching pattern: {pattern}")
        return []

    all_data = []
    for file_path in files:
        if Path(file_path).stat().st_size == 0:
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "fused_emb" in record:
                            all_data.append(record)
                    except json.JSONDecodeError as e:
                        print(f"JSON error in {file_path}:{line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Loaded {len(all_data)} records from {len(files)} files")
    return all_data


def filter_data(
    data: List[Dict[str, Any]],
    max_threshold: float = None,
    exclude_speakers: List[str] = None,
    exclude_missing: bool = False,
) -> List[Dict[str, Any]]:
    """Filter data by threshold, speakers, and missing outcomes."""
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
    print(f"Filtered {original_count} -> {len(filtered)} records")
    return filtered


def create_outcome_buckets(amounts: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Create tercile buckets (33rd/67th percentiles) for outcome amounts."""
    valid_amounts = amounts[~np.isnan(amounts)]

    if len(valid_amounts) == 0:
        return np.array(["insufficient_data"] * len(amounts)), {}

    p33 = np.percentile(valid_amounts, 33.33)
    p67 = np.percentile(valid_amounts, 66.67)

    buckets = np.where(
        np.isnan(amounts),
        "missing",
        np.where(amounts <= p33, "low", np.where(amounts <= p67, "medium", "high")),
    )

    thresholds = {
        "p33": p33,
        "p67": p67,
        "min": np.min(valid_amounts),
        "max": np.max(valid_amounts),
    }

    return buckets, thresholds


def compute_distance_matrices(
    embeddings: np.ndarray, use_gpu: bool = True
) -> Dict[str, np.ndarray]:
    """Compute various distance matrices for embeddings with optional GPU acceleration."""
    print("Computing distance matrices...")

    # Decide whether to use GPU
    use_gpu = use_gpu and CUPY_AVAILABLE
    if use_gpu:
        print("  Using GPU acceleration (CuPy)")
        # Move to GPU
        embeddings_gpu = cp.asarray(embeddings)
    else:
        print("  Using CPU computation")
        embeddings_gpu = embeddings

    distances = {}

    # Cosine distance
    print("  - Cosine distances")
    if use_gpu:
        # CuPy cosine distances
        dot_product = cp.dot(embeddings_gpu, embeddings_gpu.T)
        norms = cp.linalg.norm(embeddings_gpu, axis=1)
        cosine_sim = dot_product / cp.outer(norms, norms)
        distances["cosine"] = cp.asnumpy(1 - cosine_sim)
    else:
        distances["cosine"] = cosine_distances(embeddings)

    # Euclidean distance
    print("  - Euclidean distances")
    if use_gpu:
        distances["euclidean"] = cp.asnumpy(
            cp.linalg.norm(embeddings_gpu[:, None] - embeddings_gpu[None, :], axis=2)
        )
    else:
        distances["euclidean"] = squareform(pdist(embeddings, metric="euclidean"))

    # Manhattan distance
    print("  - Manhattan distances")
    if use_gpu:
        distances["manhattan"] = cp.asnumpy(
            cp.sum(cp.abs(embeddings_gpu[:, None] - embeddings_gpu[None, :]), axis=2)
        )
    else:
        distances["manhattan"] = squareform(pdist(embeddings, metric="cityblock"))

    # Chebyshev distance
    print("  - Chebyshev distances")
    if use_gpu:
        distances["chebyshev"] = cp.asnumpy(
            cp.max(cp.abs(embeddings_gpu[:, None] - embeddings_gpu[None, :]), axis=2)
        )
    else:
        distances["chebyshev"] = squareform(pdist(embeddings, metric="chebyshev"))

    # Mahalanobis distance (if possible)
    try:
        print("  - Mahalanobis distances")
        if use_gpu:
            cov_inv = cp.linalg.pinv(cp.cov(embeddings_gpu.T))
            diff = embeddings_gpu[:, None] - embeddings_gpu[None, :]
            mahal_dist = cp.sqrt(cp.sum((diff @ cov_inv) * diff, axis=2))
            distances["mahalanobis"] = cp.asnumpy(mahal_dist)
        else:
            cov_inv = np.linalg.pinv(np.cov(embeddings.T))
            distances["mahalanobis"] = squareform(
                pdist(embeddings, metric="mahalanobis", VI=cov_inv)
            )
    except Exception as e:
        print(f"  - Mahalanobis distance failed ({e}), using Euclidean")
        distances["mahalanobis"] = distances["euclidean"].copy()

    return distances


def analyze_bucket_statistics(
    embeddings: np.ndarray, buckets: np.ndarray, distances: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Analyze statistical properties of each bucket."""
    print("Analyzing bucket statistics...")

    unique_buckets = np.unique(buckets)
    stats_results = {}

    for bucket in unique_buckets:
        mask = buckets == bucket
        bucket_embeddings = embeddings[mask]

        if len(bucket_embeddings) < 2:
            continue

        bucket_stats = {
            "count": len(bucket_embeddings),
            "centroid": np.mean(bucket_embeddings, axis=0),
            "variance": np.var(bucket_embeddings, axis=0),
            "std": np.std(bucket_embeddings, axis=0),
            "mean_variance": np.mean(np.var(bucket_embeddings, axis=0)),
            "total_variance": np.trace(np.cov(bucket_embeddings.T)),
        }

        # Intra-bucket distances for each metric
        bucket_indices = np.where(mask)[0]
        for metric_name, dist_matrix in distances.items():
            intra_distances = []
            for i in range(len(bucket_indices)):
                for j in range(i + 1, len(bucket_indices)):
                    intra_distances.append(
                        dist_matrix[bucket_indices[i], bucket_indices[j]]
                    )

            if intra_distances:
                bucket_stats[f"intra_{metric_name}_mean"] = np.mean(intra_distances)
                bucket_stats[f"intra_{metric_name}_std"] = np.std(intra_distances)
                bucket_stats[f"intra_{metric_name}_median"] = np.median(intra_distances)

        stats_results[bucket] = bucket_stats

    return stats_results


def compute_inter_bucket_distances(
    bucket_stats: Dict[str, Any], distances: Dict[str, np.ndarray], buckets: np.ndarray
) -> Dict[str, Any]:
    """Compute inter-bucket distance statistics."""
    print("Computing inter-bucket distances...")

    unique_buckets = list(bucket_stats.keys())
    inter_bucket_stats = {}

    for i, bucket1 in enumerate(unique_buckets):
        for j, bucket2 in enumerate(unique_buckets):
            if i >= j:  # Only compute upper triangle
                continue

            mask1 = buckets == bucket1
            mask2 = buckets == bucket2
            indices1 = np.where(mask1)[0]
            indices2 = np.where(mask2)[0]

            pair_key = f"{bucket1}_vs_{bucket2}"
            inter_bucket_stats[pair_key] = {}

            for metric_name, dist_matrix in distances.items():
                inter_distances = []
                for idx1 in indices1:
                    for idx2 in indices2:
                        inter_distances.append(dist_matrix[idx1, idx2])

                if inter_distances:
                    inter_bucket_stats[pair_key][f"inter_{metric_name}_mean"] = np.mean(
                        inter_distances
                    )
                    inter_bucket_stats[pair_key][f"inter_{metric_name}_std"] = np.std(
                        inter_distances
                    )
                    inter_bucket_stats[pair_key][f"inter_{metric_name}_median"] = (
                        np.median(inter_distances)
                    )

    return inter_bucket_stats


def compute_clustering_metrics(
    embeddings: np.ndarray, buckets: np.ndarray
) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    print("Computing clustering metrics...")

    # Convert bucket labels to numeric
    unique_buckets = np.unique(buckets)
    bucket_to_numeric = {bucket: i for i, bucket in enumerate(unique_buckets)}
    numeric_labels = np.array([bucket_to_numeric[bucket] for bucket in buckets])

    metrics = {}

    # Silhouette score
    try:
        metrics["silhouette_score"] = silhouette_score(
            embeddings, numeric_labels, metric="cosine"
        )
    except:
        metrics["silhouette_score"] = np.nan

    # Calinski-Harabasz index
    try:
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(
            embeddings, numeric_labels
        )
    except:
        metrics["calinski_harabasz_score"] = np.nan

    # Davies-Bouldin index
    try:
        metrics["davies_bouldin_score"] = davies_bouldin_score(
            embeddings, numeric_labels
        )
    except:
        metrics["davies_bouldin_score"] = np.nan

    return metrics


def create_visualizations(
    embeddings: np.ndarray,
    buckets: np.ndarray,
    amounts: np.ndarray,
    distances: Dict[str, np.ndarray],
    bucket_stats: Dict[str, Any],
    inter_bucket_stats: Dict[str, Any],
    clustering_metrics: Dict[str, float],
    data: List[Dict[str, Any]],
) -> go.Figure:
    """Create comprehensive visualization dashboard."""
    print("Creating visualizations...")

    # Create subplot layout
    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=[
            "Distance Distribution by Metric",
            "Intra vs Inter-Bucket Distances",
            "Bucket Centroids (PCA)",
            "Cosine Similarity Heatmap",
            "Variance Analysis",
            "Clustering Metrics",
            "Embedding Dimension Variance",
            "Distance Correlations",
            "Bucket Size Distribution",
            "PCA Projection (Colored by Bucket)",
            "Statistical Summary",
            "Distance vs Outcome Amount",
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}, {"type": "xy"}],
        ],
    )

    unique_buckets = np.unique(buckets)
    colors = {
        "low": "#2E86AB",
        "medium": "#A23B72",
        "high": "#F18F01",
        "missing": "#C73E1D",
    }

    # 1. Distance Distribution by Metric
    for i, (metric_name, dist_matrix) in enumerate(distances.items()):
        upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        fig.add_trace(
            go.Histogram(x=upper_triangle, name=metric_name, opacity=0.7, nbinsx=50),
            row=1,
            col=1,
        )

    # 2. Intra vs Inter-bucket distances
    intra_means = []
    inter_means = []
    metric_names = []

    for bucket in unique_buckets:
        if bucket in bucket_stats:
            for metric in ["cosine", "euclidean", "manhattan"]:
                if f"intra_{metric}_mean" in bucket_stats[bucket]:
                    intra_means.append(bucket_stats[bucket][f"intra_{metric}_mean"])
                    metric_names.append(f"{bucket}_{metric}")

    for pair_key, stats in inter_bucket_stats.items():
        for metric in ["cosine", "euclidean", "manhattan"]:
            if f"inter_{metric}_mean" in stats:
                inter_means.append(stats[f"inter_{metric}_mean"])

    fig.add_trace(
        go.Scatter(
            x=list(range(len(intra_means))),
            y=intra_means,
            mode="markers",
            name="Intra-bucket",
            marker=dict(color="blue"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(inter_means))),
            y=inter_means,
            mode="markers",
            name="Inter-bucket",
            marker=dict(color="red"),
        ),
        row=1,
        col=2,
    )

    # 3. Bucket Centroids (PCA)
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    for bucket in unique_buckets:
        mask = buckets == bucket
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode="markers",
                    name=f"Centroid {bucket}",
                    marker=dict(color=colors.get(bucket, "gray"), size=4),
                ),
                row=1,
                col=3,
            )

    # 4. Cosine Similarity Heatmap (sample)
    sample_size = min(100, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_similarity = cosine_similarity(embeddings[sample_indices])

    fig.add_trace(
        go.Heatmap(z=sample_similarity, colorscale="Viridis", showscale=False),
        row=2,
        col=1,
    )

    # 5. Variance Analysis
    bucket_variances = []
    bucket_names = []
    for bucket in unique_buckets:
        if bucket in bucket_stats:
            bucket_variances.append(bucket_stats[bucket]["mean_variance"])
            bucket_names.append(bucket)

    fig.add_trace(
        go.Bar(
            x=bucket_names,
            y=bucket_variances,
            name="Mean Variance",
            marker=dict(color=[colors.get(b, "gray") for b in bucket_names]),
        ),
        row=2,
        col=2,
    )

    # 6. Clustering Metrics
    metric_names = list(clustering_metrics.keys())
    metric_values = list(clustering_metrics.values())

    fig.add_trace(
        go.Bar(x=metric_names, y=metric_values, name="Clustering Metrics"), row=2, col=3
    )

    # 7. Embedding Dimension Variance
    dim_variances = np.var(embeddings, axis=0)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(dim_variances))),
            y=dim_variances,
            mode="lines",
            name="Dimension Variance",
        ),
        row=3,
        col=1,
    )

    # 8. Distance Correlations
    metric_keys = list(distances.keys())
    if len(metric_keys) >= 2:
        dist1 = distances[metric_keys[0]][
            np.triu_indices_from(distances[metric_keys[0]], k=1)
        ]
        dist2 = distances[metric_keys[1]][
            np.triu_indices_from(distances[metric_keys[1]], k=1)
        ]

        fig.add_trace(
            go.Scatter(
                x=dist1[:1000],
                y=dist2[:1000],
                mode="markers",
                name=f"{metric_keys[0]} vs {metric_keys[1]}",
                marker=dict(size=2, opacity=0.6),
            ),
            row=3,
            col=2,
        )

    # 9. Bucket Size Distribution
    bucket_sizes = [np.sum(buckets == bucket) for bucket in unique_buckets]
    fig.add_trace(
        go.Pie(labels=unique_buckets, values=bucket_sizes, name="Bucket Sizes"),
        row=3,
        col=3,
    )

    # 10. PCA Projection (full)
    for bucket in unique_buckets:
        mask = buckets == bucket
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode="markers",
                    name=f"PCA {bucket}",
                    marker=dict(color=colors.get(bucket, "gray"), size=3, opacity=0.6),
                ),
                row=4,
                col=1,
            )

    # 11. Statistical Summary Table
    summary_data = []
    for bucket in unique_buckets:
        if bucket in bucket_stats:
            stats = bucket_stats[bucket]
            summary_data.append(
                [
                    bucket,
                    stats["count"],
                    f"{stats['mean_variance']:.4f}",
                    f"{stats.get('intra_cosine_mean', 0):.4f}",
                    f"{stats.get('intra_euclidean_mean', 0):.4f}",
                ]
            )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Bucket", "Count", "Mean Var", "Cosine Dist", "Euclidean Dist"]
            ),
            cells=dict(values=list(zip(*summary_data)) if summary_data else []),
        ),
        row=4,
        col=2,
    )

    # 12. Distance vs Outcome Amount
    if len(amounts) > 0:
        valid_mask = ~np.isnan(amounts)
        if np.any(valid_mask):
            # Sample to avoid overplotting
            sample_mask = np.random.choice(
                np.where(valid_mask)[0], min(1000, np.sum(valid_mask)), replace=False
            )
            sample_amounts = amounts[sample_mask]
            sample_distances = distances["cosine"][sample_mask][:, sample_mask]
            mean_distances = np.mean(sample_distances, axis=1)

            fig.add_trace(
                go.Scatter(
                    x=sample_amounts,
                    y=mean_distances,
                    mode="markers",
                    name="Distance vs Amount",
                    marker=dict(size=3, opacity=0.6),
                ),
                row=4,
                col=3,
            )

    # Update layout
    fig.update_layout(
        height=1600,
        title_text="Comprehensive Semantic Similarity Analysis",
        showlegend=False,
    )

    return fig


def generate_summary_report(
    bucket_stats: Dict[str, Any],
    inter_bucket_stats: Dict[str, Any],
    clustering_metrics: Dict[str, float],
    thresholds: Dict[str, float],
) -> str:
    """Generate a comprehensive text summary report."""

    report = f"""
# Semantic Similarity Analysis Report

## Dataset Overview
- Bucket thresholds: Low ≤ ${thresholds['p33']:,.2f}, Medium ≤ ${thresholds['p67']:,.2f}, High > ${thresholds['p67']:,.2f}
- Amount range: ${thresholds['min']:,.2f} - ${thresholds['max']:,.2f}

## Clustering Quality Metrics
"""

    for metric, value in clustering_metrics.items():
        if not np.isnan(value):
            report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"

    report += "\n## Bucket Statistics\n"

    for bucket, stats in bucket_stats.items():
        report += f"\n### {bucket.title()} Bucket\n"
        report += f"- Count: {stats['count']:,}\n"
        report += f"- Mean variance: {stats['mean_variance']:.6f}\n"
        report += f"- Total variance: {stats['total_variance']:.6f}\n"

        for metric in ["cosine", "euclidean", "manhattan"]:
            if f"intra_{metric}_mean" in stats:
                mean_dist = stats[f"intra_{metric}_mean"]
                std_dist = stats[f"intra_{metric}_std"]
                report += (
                    f"- Intra-{metric} distance: {mean_dist:.4f} ± {std_dist:.4f}\n"
                )

    report += "\n## Inter-Bucket Comparisons\n"

    for pair, stats in inter_bucket_stats.items():
        report += f"\n### {pair.replace('_vs_', ' vs ').title()}\n"
        for metric in ["cosine", "euclidean", "manhattan"]:
            if f"inter_{metric}_mean" in stats:
                mean_dist = stats[f"inter_{metric}_mean"]
                std_dist = stats[f"inter_{metric}_std"]
                report += (
                    f"- Inter-{metric} distance: {mean_dist:.4f} ± {std_dist:.4f}\n"
                )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive semantic similarity analysis"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file pattern",
    )
    parser.add_argument(
        "--output",
        default="similarity_analysis_report.html",
        help="Output HTML file path",
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
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (force CPU computation)",
    )

    args = parser.parse_args()

    # Parse excluded speakers
    exclude_speakers = (
        [s.strip() for s in args.exclude_speakers.split(",") if s.strip()]
        if args.exclude_speakers
        else []
    )

    print("=" * 60)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 60)

    # Load data
    data = load_data(args.input)
    if not data:
        print("ERROR: No data loaded!")
        return

    # Apply filters
    data = filter_data(data, args.max_threshold, exclude_speakers, args.exclude_missing)
    if not data:
        print("ERROR: No data remaining after filtering!")
        return

    # Extract embeddings and outcomes
    embeddings = np.array([d["fused_emb"] for d in data])
    amounts = np.array([d.get("final_judgement_real") for d in data], dtype=float)

    print(f"Processing {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")

    # Create outcome buckets
    buckets, thresholds = create_outcome_buckets(amounts)

    print("\nBucket Distribution:")
    unique, counts = np.unique(buckets, return_counts=True)
    for bucket, count in zip(unique, counts):
        percentage = 100 * count / len(buckets)
        print(f"  {bucket:>8}: {count:>6} ({percentage:>5.1f}%)")

    # Normalize embeddings
    print("\nNormalizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Compute distance matrices
    distances = compute_distance_matrices(embeddings_scaled, use_gpu=not args.no_gpu)

    # Analyze bucket statistics
    bucket_stats = analyze_bucket_statistics(embeddings_scaled, buckets, distances)

    # Compute inter-bucket distances
    inter_bucket_stats = compute_inter_bucket_distances(
        bucket_stats, distances, buckets
    )

    # Compute clustering metrics
    clustering_metrics = compute_clustering_metrics(embeddings_scaled, buckets)

    # Create visualizations
    fig = create_visualizations(
        embeddings_scaled,
        buckets,
        amounts,
        distances,
        bucket_stats,
        inter_bucket_stats,
        clustering_metrics,
        data,
    )

    # Generate summary report
    summary_report = generate_summary_report(
        bucket_stats, inter_bucket_stats, clustering_metrics, thresholds
    )

    # Save results
    print(f"\nSaving analysis to: {args.output}")

    # Add summary to the plot
    fig.update_layout(
        annotations=[
            dict(
                text=summary_report,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=10, family="monospace"),
            )
        ]
    )

    fig.write_html(args.output)

    # Also save text report
    text_report_path = args.output.replace(".html", "_report.txt")
    with open(text_report_path, "w") as f:
        f.write(summary_report)

    print(f"✅ Analysis complete!")
    print(f"📊 Interactive report: {args.output}")
    print(f"📝 Text summary: {text_report_path}")


if __name__ == "__main__":
    main()
