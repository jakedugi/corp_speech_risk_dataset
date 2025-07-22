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
from numpy import nanpercentile
import random

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
        supervision_mode: str = "categorical",
    ):
        self.vec_path = Path(vec_path)
        self.meta_path = Path(meta_path)
        self.use_gpu = use_gpu
        self.min_cluster_size = min_cluster_size

        # ── 1) Load metadata first (fixes AttributeError) ───────────────────────
        self.meta: List[Dict] = json.loads(Path(self.meta_path).read_text())

        raw = np.load(self.vec_path, mmap_mode="r").astype(np.float32)
        # L2-normalize (so Euclidean ≃ Cosine on unit sphere)
        normalized = normalize(raw, norm="l2", axis=1)
        # ↓—— add this: random-project to 256 dims for faster neighbor search
        rp = GaussianRandomProjection(n_components=256, random_state=42)
        self.cluster_vectors = rp.fit_transform(normalized)
        # keep full-dim for index/visualization
        self.vectors = normalized
        # compute ordinal buckets + explicit missing category
        y = np.array([m.get("final_judgement_real") for m in self.meta], dtype=object)
        is_missing = [
            v is None or (isinstance(v, float) and np.isnan(v)) for v in y
        ]  # flag nulls  [oai_citation:2‡letsdatascience.com](https://letsdatascience.com/handling-missing-values/?utm_source=chatgpt.com)
        vals = np.array(
            [float(v) if not m else np.nan for v, m in zip(y, is_missing)],
            dtype=np.float64,
        )
        q33, q66 = nanpercentile(
            vals[~np.isnan(vals)], [33.3, 66.6]
        )  # ignore NaNs  [oai_citation:3‡GitHub](https://github.com/lmcinnes/umap/issues/135?utm_source=chatgpt.com)
        buckets = np.where(
            is_missing,
            "missing",
            np.select([vals < q33, vals < q66], ["low", "med"], "high"),
        )

        # ── Encode string buckets into integer codes for UMAP ──────────────
        # This numeric array satisfies sklearn.check_array while keeping strings
        categories, bucket_codes = np.unique(buckets, return_inverse=True)
        # bucket_codes is an integer array [0=len(meta)-1], values in {0,1,2,3}
        # we'll use `buckets` (strings) later for coloring, but pass bucket_codes to UMAP

        # decide supervision target for UMAP
        if supervision_mode == "continuous":
            # drop missing y entirely and jitter
            mask = ~np.isnan(vals)
            vals_clean = vals[mask]
            noise = np.random.normal(
                scale=1e-6 * np.nanstd(vals_clean), size=vals_clean.shape
            )
            target = (
                vals_clean + noise
            )  # continuous numeric target  [oai_citation:2‡PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9403402/?utm_source=chatgpt.com)
            # also drop corresponding rows from vectors & meta
            self.vectors = self.vectors[mask]
            self.cluster_vectors = self.cluster_vectors[mask]
            self.meta = [m for (m, ok) in zip(self.meta, mask) if ok]
            buckets = buckets[mask]
            is_missing = np.array(is_missing)[mask]
        else:
            # categorical: use integer codes (not strings) as y for UMAP
            target = bucket_codes  # array of ints  [0..n_categories)  [oai_citation:4‡UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/supervised.html?utm_source=chatgpt.com)

        # save for plotting & diagnostics
        self.buckets = buckets
        self.is_missing = np.array(is_missing)

        if self.vectors.shape[0] != len(self.meta):
            raise ValueError("Vector / metadata length mismatch")

        self.index = FaissIndex(dim=self.vectors.shape[1], use_gpu=use_gpu)
        self.clusterer = HDBSCANClusterer(min_cluster_size=min_cluster_size)
        # supervised UMAP with categorical buckets (no numeric NaNs)
        self.buckets = buckets  # save for plotting & diagnostics
        self.is_missing = np.array(is_missing)
        # ── 4) Supervised UMAP ─────────────────────────────────────────────
        # categorical: string labels; continuous: jittered floats
        tm = "categorical" if supervision_mode == "categorical" else "l2"
        self.reducer = DimReducer(
            n_neighbors=50,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            target=target,  # now numeric: jittered floats or bucket_codes
            target_weight=0.7,
            target_metric=tm,
        )
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
        # Build the DataFrame and inject bucket / missing / raw-y columns
        df = self.dataframe()
        df["bucket"] = (
            self.buckets
        )  # low/med/high/missing  [ pandas.DataFrame.assign docs turn0search16 ]
        df["is_missing"] = self.is_missing  # boolean flag for NaNs
        df["final_judgement_real"] = [
            m.get("final_judgement_real", np.nan) for m in self.meta
        ]
        return self.visualizer.scatter(df, out_html=out_html)
