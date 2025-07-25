"""Command‑line entry‑point for the reversible clustering workflow."""

import argparse
from pathlib import Path
import numpy as np

from corp_speech_risk_dataset.clustering.pipeline import ClusterPipeline


def _parse_args():
    p = argparse.ArgumentParser(
        description="Reversible clustering of BPE + WL vectors using FAISS → HDBSCAN → UMAP",
    )
    p.add_argument(
        "--vec", required=True, help="Path to *.npy* file of concatenated vectors"
    )
    p.add_argument(
        "--meta",
        required=True,
        help="Path to JSON metadata file containing at least 'text' and 'sp_ids' per entry",
    )
    p.add_argument("--out", default="clusters.html", help="Output HTML scatter plot")
    p.add_argument("--gpu", action="store_true", help="Use GPU (CUDA) if available")
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument(
        "--supervision",
        dest="supervision_mode",  # map flag into args.supervision_mode  [oai_citation:0‡omz-software.com](https://omz-software.com/editorial/docs/library/argparse.html?utm_source=chatgpt.com)
        choices=[
            "categorical",
            "continuous",
        ],  # restricts to allowed modes  [oai_citation:1‡Stack Overflow](https://stackoverflow.com/questions/40324356/python-argparse-choices-with-a-default-choice?utm_source=chatgpt.com)
        default="categorical",  # default remains the same  [oai_citation:2‡Python documentation](https://docs.python.org/3/library/argparse.html?utm_source=chatgpt.com)
        help="UMAP supervision mode: categorical (default) or continuous with jitter",
    )
    return p.parse_args()


def main():  # pragma: no cover
    args = _parse_args()
    pipe = ClusterPipeline(
        vec_path=args.vec,
        meta_path=args.meta,
        use_gpu=args.gpu,
        min_cluster_size=args.min_cluster_size,
        supervision_mode=args.supervision_mode,
    )
    print("[1/3] Building FAISS index …", flush=True)
    pipe.build()
    print("[2/3] Clustering with HDBSCAN …", flush=True)
    n_clusters = len(set(pipe.cluster()) - {-1})
    print(f"[✓] Found {n_clusters} clusters (noise label = −1)")
    # ── 3) Quantitative diagnostics ─────────────────────────────────────────
    # reduce first so we can compute silhouette on the final 2-D coords
    coords = pipe.reduce()
    from sklearn.metrics import (
        silhouette_score,
        silhouette_samples,
        adjusted_rand_score,
    )

    # 1) Standard mean Silhouette
    sil = silhouette_score(coords, pipe.clusterer.labels_, metric="euclidean")
    # 2) Per-sample Silhouette values
    sil_vals = silhouette_samples(coords, pipe.clusterer.labels_, metric="euclidean")
    # 3) Macro-average: mean Silhouette per cluster, then mean across clusters
    macro_sil = np.mean(
        [
            sil_vals[pipe.clusterer.labels_ == c].mean()
            for c in np.unique(pipe.clusterer.labels_)
        ]
    )  # implements per-cluster then across-cluster averaging  [oai_citation:5‡GitHub](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/cluster/_unsupervised.py?utm_source=chatgpt.com) [oai_citation:6‡Stack Overflow](https://stackoverflow.com/questions/59919627/how-to-calculate-the-silhouette-score-for-each-cluster-separately-in-python?utm_source=chatgpt.com)
    # 4) Adjusted Rand Index
    ari = adjusted_rand_score(pipe.buckets, pipe.clusterer.labels_)

    # Print all three metrics
    print(f"[✓] Silhouette (mean)           : {sil:.3f}")
    print(f"[✓] Silhouette (macro-averaged) : {macro_sil:.3f}")
    print(f"[✓] ARI                         : {ari:.3f}")

    print("[4/4] Writing HTML scatter …", flush=True)
    out_path = pipe.visualise(args.out)
    print(f"[✓] Done → {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
