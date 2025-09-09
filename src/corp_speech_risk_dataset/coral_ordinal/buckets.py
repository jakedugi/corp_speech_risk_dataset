# ----------------------------------
# coral_ordinal/buckets.py
# ----------------------------------


from __future__ import annotations
import numpy as np
from typing import Sequence


def make_buckets(
    values: Sequence[float],
    labels: Sequence[str] | None = None,
    qcuts: Sequence[float] | None = None,
):
    """Create ordinal buckets from continuous values (if you still need to derive H/M/L).
    - values: iterable of reals
    - labels: ordered bucket names; length == len(qcuts)-1
    - qcuts: list of quantiles like [0, .33, .66, 1]
    Returns: edges (np.ndarray), label list
    """
    if qcuts is None:
        qcuts = [0, 1 / 3, 2 / 3, 1]
    edges = np.quantile(values, qcuts)
    if labels is None:
        labels = [f"B{i}" for i in range(len(edges) - 1)]
    assert len(labels) == len(edges) - 1
    return edges, labels


def bucketize(x: float, edges: Sequence[float], labels: Sequence[str]):
    idx = np.searchsorted(edges[1:], x, side="right")
    return labels[idx]
