#!/usr/bin/env python3
"""
Simple clustering script for outcome data.

Usage:
    python scripts/cluster_outcomes_simple.py \
        data/outcomes/courtlistener_v1/0:15-cv-62604_flsd/doc_4101961_text_stage9.jsonl
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict


# Temporary workaround - create the data files in the expected format
def quick_cluster(jsonl_path: str, output_html: str = "quick_clusters.html"):
    """Quick clustering of a single JSONL file."""

    # Load data
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            if "fused_emb" in record and "final_judgement_real" in record:
                data.append(record)

    print(f"Loaded {len(data)} records")

    if len(data) < 2:
        print("Need at least 2 records for clustering")
        return

    # Prepare data
    embeddings = np.array([d["fused_emb"] for d in data], dtype=np.float32)
    metadata = []

    for i, record in enumerate(data):
        meta_entry = {
            "text": record.get("text", f"Record {i}"),
            "sp_ids": record.get("sp_ids", []),
            "doc_id": record.get("doc_id", f"doc_{i}"),
            "final_judgement_real": record.get("final_judgement_real"),
            "speaker": record.get("speaker", ""),
        }
        metadata.append(meta_entry)

    # Save temporarily
    temp_vec = "temp_embeddings.npy"
    temp_meta = "temp_metadata.json"

    np.save(temp_vec, embeddings)
    with open(temp_meta, "w") as f:
        json.dump(metadata, f)

    # Import and run pipeline
    from corp_speech_risk_dataset.clustering.pipeline import ClusterPipeline

    pipeline = ClusterPipeline(
        vec_path=temp_vec,
        meta_path=temp_meta,
        use_gpu=False,
        min_cluster_size=2,  # Small for testing
        supervision_mode="categorical",
    )

    print("Building index...")
    pipeline.build()

    print("Clustering...")
    labels = pipeline.cluster()
    n_clusters = len(set(labels) - {-1})
    print(f"Found {n_clusters} clusters")

    print("Creating visualization...")
    output_path = pipeline.visualise(output_html)
    print(f"Saved to: {output_path}")

    # Cleanup
    Path(temp_vec).unlink()
    Path(temp_meta).unlink()

    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/cluster_outcomes_simple.py <jsonl_file>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    if not Path(jsonl_path).exists():
        print(f"File not found: {jsonl_path}")
        sys.exit(1)

    quick_cluster(jsonl_path)
