"""UMAP projection helper for 2‑D visualisation (optional)."""

from __future__ import annotations

import umap  # type: ignore
import numpy as np


class DimReducer:
    """Project high-dim vectors to 2-D using UMAP (supervised or unsupervised)."""

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        metric: str = "cosine",
        random_state: int = 42,
        target: np.ndarray | None = None,
        target_weight: float = 0.9,
        target_metric: str | None = None,
        spread: float = 1.0,
        set_op_mix_ratio: float = 0.5,
        local_connectivity: int = 1,
    ):
        """
        Args:
          target: 1-D array of labels (numeric or categorical).
          target_weight: float in [0,1], tradeoff between data and target.
          target_metric: 'l2', 'l1', or 'categorical' (None → unsupervised).
        """
        self._target = target
        # now UMAP will use exactly the metric you specify, no guessing
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            target_metric=target_metric,
            target_weight=target_weight,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            spread=spread,
        )

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform supervised (if target provided) or unsupervised UMAP.
        Identical y-values incur zero distance; jitter breaks ties  [oai_citation:8‡dr.ntu.edu.sg](https://dr.ntu.edu.sg/bitstream/10356/170673/2/Emadeldeen_PhD_thesis.pdf?utm_source=chatgpt.com).
        """
        # we pass y to fit_transform; UMAP handles NaNs as unlabelled  [oai_citation:9‡umap-learn.readthedocs.io](https://umap-learn.readthedocs.io/en/latest/_sources/supervised.rst.txt?utm_source=chatgpt.com) [oai_citation:10‡umap-learn.readthedocs.io](https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html?utm_source=chatgpt.com)
        return self.reducer.fit_transform(vectors, y=self._target)
