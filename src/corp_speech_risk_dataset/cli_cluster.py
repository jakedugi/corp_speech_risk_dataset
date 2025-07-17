"""Command‑line entry‑point for the reversible clustering workflow."""

import argparse
from pathlib import Path

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
    return p.parse_args()


def main():  # pragma: no cover
    args = _parse_args()
    pipe = ClusterPipeline(
        vec_path=args.vec,
        meta_path=args.meta,
        use_gpu=args.gpu,
        min_cluster_size=args.min_cluster_size,
    )
    print("[1/3] Building FAISS index …", flush=True)
    pipe.build()
    print("[2/3] Clustering with HDBSCAN …", flush=True)
    n_clusters = len(set(pipe.cluster()) - {-1})
    print(f"[✓] Found {n_clusters} clusters (noise label = −1)")
    print("[3/3] Reducing to 2‑D and writing HTML …", flush=True)
    out_path = pipe.visualise(args.out)
    print(f"[✓] Done → {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
