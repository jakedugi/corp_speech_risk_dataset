"""UMAP projection helper for 2‑D visualisation (optional)."""

from __future__ import annotations

import umap  # type: ignore
import numpy as np


class DimReducer:
    """Project high‑dim vectors to 2‑D using UMAP."""

    def __init__(
        self,
        *,
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ):
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(vectors)
