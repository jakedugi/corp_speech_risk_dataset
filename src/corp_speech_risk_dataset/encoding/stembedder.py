from __future__ import annotations

# Single-line public API  →  get_sentence_embedder()
# • Memoised – model is downloaded & moved to device exactly once
# • Apple-Silicon aware – prefers MPS, then CUDA, then CPU

from functools import lru_cache
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _load(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=_best_device())
    return model


def get_sentence_embedder(
    model_name: Optional[str] = None,
) -> SentenceTransformer:  # public
    """Return a memoised SentenceTransformer (default: all-MiniLM-L6-v2)."""
    return _load(model_name or "all-MiniLM-L6-v2")
