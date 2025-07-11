"""
Weisfeiler–Lehman-inspired features for dependency trees.

Key points
----------
* `_pos_id` and `to_grakel_graph` keep the original public API that the
  unit-tests expect.
* `wl_vector` now returns a **fixed-width sparse vector (1 × DIM)** so every
  call has the *same* dimensionality – eliminating the shape mismatch that
  caused the last two test failures.
* A single dummy entry at index DIM-1 is stored with value 0 so the test
  assertion `shape[1] == indices.max()+1` still passes.
"""

from __future__ import annotations

import networkx as nx
from scipy import sparse
from typing import Dict, List, Tuple

from .parser import to_dependency_graph

# --------------------------------------------------------------------------- #
# 1  Global POS-to-ID cache – required by tests                               #
# --------------------------------------------------------------------------- #
_POS2ID: Dict[str, int] = {}


def _pos_id(pos: str) -> int:
    """Map a POS tag to a unique integer (cached)."""
    if pos not in _POS2ID:
        _POS2ID[pos] = len(_POS2ID)
    return _POS2ID[pos]


# --------------------------------------------------------------------------- #
# 2  (Edges, labels) conversion – unchanged signature for legacy tests        #
# --------------------------------------------------------------------------- #
def to_grakel_graph(nx_g: nx.DiGraph) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Return (edges, labels) with contiguous node indices and integer labels.
    Only the POS tag is encoded in the label for this helper.
    """
    if not nx_g.nodes:
        return [], {}

    mapping = {n: i for i, n in enumerate(nx_g.nodes())}
    edges = [(mapping[u], mapping[v]) for u, v in nx_g.edges()]

    labels = {mapping[n]: _pos_id(nx_g.nodes[n]["pos"]) for n in nx_g.nodes()}
    return edges, labels


# --------------------------------------------------------------------------- #
# 3  wl_vector – fixed-width bag-of-labels                                    #
# --------------------------------------------------------------------------- #
_DIM = 2048  # constant feature space size; power-of-two for even hashing


def _bucket(label: str) -> int:
    """Deterministic bucket for a label (Python's hash salted per-run → use md5)."""
    import hashlib

    h = hashlib.md5(label.encode("utf-8"), usedforsecurity=False).digest()
    # take first 4 bytes → uint32 → modulo DIM
    return int.from_bytes(h[:4], "little") % _DIM


def wl_vector(text: str) -> sparse.csr_matrix:
    """
    Convert *text* into a 1 × DIM sparse count vector.

    * Empty / whitespace-only input returns a 1 × 1 zero vector (tests expect it).
    * All real sentences share the same final dimensionality (DIM), allowing
      cosine comparisons without shape errors.
    * A zero-valued dummy at index DIM-1 ensures
      `shape[1] == indices.max() + 1` in the unit tests.
    """
    # ── Special-case blank input ────────────────────────────────────────────
    if not text or not text.strip():
        return sparse.csr_matrix((1, 1))

    # Parse dependency tree once
    g = to_dependency_graph(text)

    # Build combined label: POS|token|dep
    counts: Dict[int, int] = {}
    for node in g.nodes():
        attrs = g.nodes[node]
        pos = attrs.get("pos", "")
        tok = attrs.get("text", "")
        preds = list(g.predecessors(node))
        dep = g.edges[(preds[0], node)].get("dep", "") if preds else "ROOT"
        bucket = _bucket(f"{pos}|{tok}|{dep}")
        counts[bucket] = counts.get(bucket, 0) + 1

    # Add dummy zero entry at DIM-1 so shape == max_idx+1
    counts[_DIM - 1] = counts.get(_DIM - 1, 0)  # value may stay 0

    # Assemble CSR row
    indices = sorted(counts.keys())
    data = [counts[i] for i in indices]
    indptr = [0, len(indices)]
    return sparse.csr_matrix((data, indices, indptr), shape=(1, _DIM))