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

        # make is_missing available on self before resampling
        self.is_missing = np.array(is_missing)

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

        # ── QUICK OVERSAMPLING with small Gaussian noise ────────────────────
        rs = np.random.RandomState(42)
        counts = np.bincount(bucket_codes)
        max_count = counts.max()
        bucket_codes_orig = bucket_codes.copy()
        resampled_idx = []
        for cls in np.unique(bucket_codes_orig):

            idx_cls = np.where(bucket_codes_orig == cls)[0]
            n_needed = max_count - len(idx_cls)
            # sample with replacement and add slight noise to jitter
            choice = rs.choice(idx_cls, n_needed, replace=True)
            resampled_idx.append(np.concatenate([idx_cls, choice]))
        resampled_idx = np.concatenate(resampled_idx)  # final index array
        bucket_codes = bucket_codes[resampled_idx]  # balanced labels
        self.vectors = self.vectors[resampled_idx]  # balanced vectors
        self.cluster_vectors = self.cluster_vectors[resampled_idx]
        buckets = buckets[resampled_idx]
        self.meta = [self.meta[i] for i in resampled_idx]
        # after reshaping self.vectors and self.cluster_vectors:
        dup_mask = np.zeros(len(resampled_idx), bool)
        # mark the last n_needed entries in each class block as duplicates
        start = 0
        for cls in np.unique(bucket_codes_orig):
            idx_cls = np.where(bucket_codes_orig == cls)[0]
            n_needed = max_count - len(idx_cls)
            dup_mask[start + len(idx_cls) : start + len(idx_cls) + n_needed] = True
            start += len(idx_cls) + n_needed
        # # apply noise only to duplicates
        noise = rs.normal(
            scale=1e-3 * np.linalg.norm(self.vectors, axis=1).mean(),
            size=self.vectors.shape,
        )
        self.vectors[dup_mask] += noise[dup_mask]
        # now recompute is_missing to match our oversampled meta
        is_missing = [
            (m.get("final_judgement_real") is None)
            or (
                isinstance(m.get("final_judgement_real"), float)
                and np.isnan(m.get("final_judgement_real"))
            )
            for m in self.meta
        ]
        self.is_missing = np.array(is_missing)
        target = bucket_codes

        # ── 2) JITTER to break exact duplicates ──────────────────────
        from sklearn.neighbors import KDTree

        rs_j = np.random.RandomState(123)
        tree = KDTree(self.vectors)
        dists, _ = tree.query(self.vectors, k=2)
        avg_nndist = dists[:, 1].mean()
        jitter_scale = 0.001 * avg_nndist
        print(f"avg_nndist={avg_nndist:.4f}, jitter_scale={jitter_scale:.6f}")
        self.vectors += rs_j.normal(
            scale=jitter_scale, size=self.vectors.shape
        )  # avoids UMAP neighbor‐graph collapse  [oai_citation:0‡gemfury.com](https://gemfury.com/emaballarin/python%3Aumap-learn/-/content/parametric_umap.py)  [oai_citation:1‡GitHub](https://github.com/lmcinnes/umap/issues/771?utm_source=chatgpt.com)

        # ── 3) CLUSTER-BASED UNDERSAMPLING of the majority class ─────
        from sklearn.cluster import KMeans

        # Identify the largest class code
        maj_code = np.bincount(target).argmax()
        min_count = np.bincount(target).min()
        maj_idx = np.where(target == maj_code)[0]
        # Cluster majority into balanced groups
        n_clusters = int(np.floor(len(maj_idx) / min_count)) or 1
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        maj_labels = kmeans.fit_predict(self.vectors[maj_idx])
        undersample_idx = []
        for c in range(n_clusters):
            members = maj_idx[maj_labels == c]
            # only sample if sufficient members
            if len(members) >= min_count:
                undersample_idx.append(rs.choice(members, min_count, replace=False))
            else:
                undersample_idx.append(members)
        undersample_idx = np.concatenate(undersample_idx)
        # Keep *all* minority + balanced subset of majority
        keep_idx = np.where(target != maj_code)[0].tolist() + undersample_idx.tolist()
        keep_idx = np.array(keep_idx, dtype=int)
        self.vectors = self.vectors[keep_idx]
        self.cluster_vectors = self.cluster_vectors[keep_idx]
        target = target[keep_idx]
        self.meta = [self.meta[i] for i in keep_idx]
        # now `target` is balanced by cluster among the majority  [oai_citation:2‡GitHub](https://github.com/farshidrayhancv/CUSBoost?utm_source=chatgpt.com) [oai_citation:3‡arXiv](https://arxiv.org/abs/1712.04356?utm_source=chatgpt.com)

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
        self.clusterer = HDBSCANClusterer(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
        )
        # supervised UMAP with categorical buckets (no numeric NaNs)
        self.buckets = buckets  # save for plotting & diagnostics
        self.is_missing = np.array(is_missing)
        # ── 4) Supervised UMAP (using target_weight only) ────────────────
        tm = "categorical" if supervision_mode == "categorical" else "l2"
        self.reducer = DimReducer(
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            target=target,
            spread=0.5,
            target_weight=0.9,  # bias toward bucket labels
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
