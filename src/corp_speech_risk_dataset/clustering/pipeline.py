"""High‑level orchestrator that glues *FaissIndex*, *HDBSCANClusterer*,
*DimReducer*, and *Visualizer* together.

This keeps the public API surface small while respecting separation of concerns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection

from ..encoding.tokenizer import decode_sp_ids  # existing util
from .faiss_index import FaissIndex
from .hdbscan_clusterer import HDBSCANClusterer
from .dim_reducer import DimReducer
from .visualize import Visualizer


class ClusterPipeline:
    """Run the full reversible clustering workflow in three phases."""

    def __init__(
        self,
        *,
        vec_path: str | Path,
        meta_path: str | Path,
        use_gpu: bool = False,
        min_cluster_size: int = 50,
    ):
        self.vec_path = Path(vec_path)
        self.meta_path = Path(meta_path)
        self.use_gpu = use_gpu
        self.min_cluster_size = min_cluster_size

        raw = np.load(self.vec_path, mmap_mode="r").astype(np.float32)
        # L2-normalize (so Euclidean ≃ Cosine on unit sphere)
        normalized = normalize(raw, norm="l2", axis=1)
        # ↓—— add this: random-project to 256 dims for faster neighbor search
        rp = GaussianRandomProjection(n_components=256, random_state=42)
        self.cluster_vectors = rp.fit_transform(normalized)
        # keep full-dim for index/visualization
        self.vectors = normalized
        self.meta: List[Dict] = json.loads(Path(self.meta_path).read_text())

        if self.vectors.shape[0] != len(self.meta):
            raise ValueError("Vector / metadata length mismatch")

        self.index = FaissIndex(dim=self.vectors.shape[1], use_gpu=use_gpu)
        self.clusterer = HDBSCANClusterer(min_cluster_size=min_cluster_size)
        self.reducer = DimReducer()
        self.visualizer = Visualizer()

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Add all vectors to the FAISS index (in‑place)."""
        t0 = time.perf_counter()
        self.index.add(self.vectors)
        t1 = time.perf_counter()
        print(f"Faiss build time: {t1 - t0:.1f}s")

    # ------------------------------------------------------------------
    def cluster(self) -> np.ndarray:
        """Run HDBSCAN on the full vector set and return labels."""
        t0 = time.perf_counter()

        labels = self.clusterer.fit(self.cluster_vectors)

        t1 = time.perf_counter()
        print(f"HDBSCAN clustering time: {t1 - t0:.1f}s")
        return labels

    # ------------------------------------------------------------------
    def reduce(self) -> np.ndarray:
        """Run 2‑D UMAP (for visualisation only)."""
        t0 = time.perf_counter()
        coords = self.reducer.fit_transform(self.vectors)
        t1 = time.perf_counter()
        print(f"UMAP reduction time for visualization: {t1 - t0:.1f}s")
        return coords

    # ------------------------------------------------------------------
    def dataframe(self) -> pd.DataFrame:
        """Return a *tidy* DataFrame with coords, cluster id, and sentence text."""
        coords = self.reduce()
        labels = self.clusterer.labels_
        sentences = [m["text"] for m in self.meta]
        decoded = [decode_sp_ids(m["sp_ids"]) for m in self.meta]

        # keep an immutable key for exact round-trip
        doc_ids = [m.get("doc_id", str(i)) for i, m in enumerate(self.meta)]
        indices = np.arange(len(self.meta))  # stable positional key

        df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "cluster": labels,
                "sentence": sentences,
                "decoded": decoded,
                "doc_id": doc_ids,
                "idx": indices,
            }
        )
        return df

    # OPTIONAL: persist mapping → jsonl
    def save_labels(self, out_path: str | Path = "cluster_labels.json") -> None:
        t0 = time.perf_counter()
        mapping = {
            "idx": list(range(len(self.meta))),
            "cluster": self.clusterer.labels_.tolist(),
        }
        Path(out_path).write_text(json.dumps(mapping))
        t1 = time.perf_counter()
        print(f"Labels saved in {t1 - t0:.1f}s")

    # ------------------------------------------------------------------
    def visualise(self, out_html: str | Path = "clusters.html") -> Path:
        df = self.dataframe()
        return self.visualizer.scatter(df, out_html=out_html)
