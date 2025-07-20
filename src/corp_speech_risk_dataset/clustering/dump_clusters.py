#!/usr/bin/env python3
""" For use after the clustering pipeline has been run. In order to get the cluster labels for each document, we need to dump the cluster labels to a JSON file.
it’s very handy for:
	•	Automating downstream analysis or plotting (e.g. grouping sentences by risk-level in Python/R).
	•	Hyperparameter sweeps: you can quickly diff two runs’ JSONs to see how cluster assignments shifted.
	•	Audit logs: regulators often want a raw data dump, not just visuals.
"""
from pathlib import Path
import argparse
from corp_speech_risk_dataset.clustering.pipeline import ClusterPipeline


def main():
    p = argparse.ArgumentParser(description="Dump idx→cluster mapping to JSON")
    p.add_argument(
        "--vec", required=True, help="Path to *.npy* file of concatenated vectors"
    )
    p.add_argument("--meta", required=True, help="Path to metadata.json")
    p.add_argument(
        "--out", default="cluster_labels.json", help="Output JSON mapping file"
    )
    args = p.parse_args()

    pipe = ClusterPipeline(
        vec_path=Path(args.vec),
        meta_path=Path(args.meta),
        use_gpu=False,  # or True if you want
        min_cluster_size=50,  # or your chosen value
    )
    pipe.build()
    pipe.cluster()
    pipe.save_labels(Path(args.out))
    print(f"→ Wrote idx→cluster mapping to {args.out}")


if __name__ == "__main__":
    main()
