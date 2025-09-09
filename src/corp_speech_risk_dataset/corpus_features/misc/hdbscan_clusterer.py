"""Density‑based clustering wrapper (HDBSCAN)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import hdbscan  # type: ignore


class HDBSCANClusterer:
    """Convenience wrapper around *hdbscan.HDBSCAN* using Euclidean on normalized data (cosine proxy)."""

    def __init__(self, *, min_cluster_size: int = 50, metric: str = "euclidean"):
        self.min_cluster_size = int(min_cluster_size)
        self.metric = metric
        self._model: Optional[hdbscan.HDBSCAN] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN and return cluster labels."""
        # 1) Use all CPU cores for neighbor search (faster on M1)
        # 2) Enable approximate MST building (optional speed boost)
        self._model = hdbscan.HDBSCAN(
            metric=self.metric,
            min_cluster_size=self.min_cluster_size,
            core_dist_n_jobs=-1,  # full parallelism  [oai_citation:3‡Read the Docs](https://media.readthedocs.org/pdf/hdbscan/0.8.6/hdbscan.pdf?utm_source=chatgpt.com)
            approx_min_span_tree=True,  # approximate MST via NN-Descent  [oai_citation:4‡Read the Docs](https://media.readthedocs.org/pdf/hdbscan/0.8.6/hdbscan.pdf?utm_source=chatgpt.com)
            leaf_size=20,  # shallower BallTree → faster queries
        )
        self._model.fit(vectors)
        self.labels_ = self._model.labels_.astype(int)
        return self.labels_

    # Expose properties for downstream access
    @property
    def probabilities_(self):
        if self._model is None:
            raise RuntimeError("fit() must be called first")
        return self._model.probabilities_

    @property
    def outlier_scores_(self):
        if self._model is None:
            raise RuntimeError("fit() must be called first")
        return self._model.outlier_scores_
