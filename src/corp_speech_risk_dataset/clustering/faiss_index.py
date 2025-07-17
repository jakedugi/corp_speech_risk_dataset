"""Thin wrapper around FAISS for exact inner‑profaissduct search.

The class keeps the *original* float32 vectors untouched so the pipeline remains
loss‑less and reversible.  GPU acceleration is optional; when enabled it
leverages FAISS' CUDA back‑end, otherwise Apple Accelerate is used implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "faiss‑cpu or faiss‑gpu must be installed before using FaissIndex"
    ) from exc


class FaissIndex:
    """Exact (IndexFlatIP) similarity search wrapper."""

    _CPU_INDEX_CLS = faiss.IndexFlatIP

    def __init__(self, dim: int, *, use_gpu: bool = False, gpu_id: int = 0):
        self.dim = dim
        self.use_gpu = bool(use_gpu)
        self.gpu_id = int(gpu_id)

        index = self._CPU_INDEX_CLS(dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
        self._index = index

    # ------------------------------------------------------------------
    # Building / updating
    # ------------------------------------------------------------------
    def add(self, vectors: np.ndarray) -> None:
        """Add *float32* vectors to the index (in‑place)."""
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("Vector array shape mismatch with index dimension")
        self._index.add(vectors)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Return *(similarities, indices)* for the *k* nearest neighbours."""
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        sims, idxs = self._index.search(query, k)
        return sims, idxs

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: str | Path, *, use_gpu: bool | None = None) -> "FaissIndex":
        path = Path(path)
        index = faiss.read_index(str(path))
        dim = index.d
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.use_gpu = bool(use_gpu)
        obj.gpu_id = 0
        obj._index = index
        # NOTE: CPU/GPU transfer must be done *after* reading
        if use_gpu:
            res = faiss.StandardGpuResources()
            obj._index = faiss.index_cpu_to_gpu(res, 0, index)
        return obj
