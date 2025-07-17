"""
Domain module for corporate speech risk feature calculations.

This module contains pure business logic for computing risk features
from text, following Domain-Driven Design principles with no external
dependencies.
"""

from __future__ import annotations

import hashlib
from typing import Dict
from scipy import sparse


# --------------------------------------------------------------------------- #
# Risk Feature Constants                                                      #
# --------------------------------------------------------------------------- #
_DIM = 2048  # constant feature space size; power-of-two for even hashing


def _bucket(label: str) -> int:
    """
    Deterministic bucket for a label using MD5 hashing.

    This is core business logic for risk feature calculation,
    providing consistent bucketing across different runs.

    Args:
        label: The label string to bucket

    Returns:
        Integer bucket index in range [0, _DIM)
    """
    h = hashlib.md5(label.encode("utf-8"), usedforsecurity=False).digest()
    # take first 4 bytes → uint32 → modulo DIM
    return int.from_bytes(h[:4], "little") % _DIM


def calculate_risk_vector_counts(
    pos_tokens_deps: list[tuple[str, str, str]]
) -> Dict[int, int]:
    """
    Calculate risk feature counts from POS/token/dependency triples.

    This is pure business logic for risk calculation that operates on
    structured data without external parsing dependencies.

    Args:
        pos_tokens_deps: List of (pos_tag, token, dependency) tuples

    Returns:
        Dictionary mapping bucket indices to their counts
    """
    counts: Dict[int, int] = {}

    for pos, token, dep in pos_tokens_deps:
        bucket = _bucket(f"{pos}|{token}|{dep}")
        counts[bucket] = counts.get(bucket, 0) + 1

    # Add dummy zero entry at DIM-1 so shape == max_idx+1
    counts[_DIM - 1] = counts.get(_DIM - 1, 0)  # value may stay 0

    return counts


def counts_to_sparse_vector(counts: Dict[int, int]) -> sparse.csr_matrix:
    """
    Convert risk feature counts to a sparse vector representation.

    Args:
        counts: Dictionary mapping bucket indices to counts

    Returns:
        1 × DIM sparse CSR matrix
    """
    if not counts:
        return sparse.csr_matrix((1, 1))

    # Assemble CSR row
    indices = sorted(counts.keys())
    data = [counts[i] for i in indices]
    indptr = [0, len(indices)]
    return sparse.csr_matrix((data, indices, indptr), shape=(1, _DIM))


def get_feature_dimension() -> int:
    """
    Get the fixed feature space dimension.

    Returns:
        The dimension of the risk feature space
    """
    return _DIM
