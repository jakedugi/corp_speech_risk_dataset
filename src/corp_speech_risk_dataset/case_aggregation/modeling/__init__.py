"""Case-level modeling from quote-level risk predictions.

This subpackage provides utilities to aggregate per-quote predicted risk labels
and positional features into per-case feature vectors, and to train simple,
efficient models (scikit-learn) to predict case-level outcomes.

Modules:
- utils: File IO and case-id parsing helpers
- dataset: Thresholding, aggregation, dataset construction
- models: Lightweight model registry and evaluation utilities
- cli: Command-line interface for end-to-end training/evaluation

Design:
- KISS: small, composable functions with clear responsibilities
- Efficient: avoids heavy dependencies; uses Polars for tabular ops
- Extensible: feature aggregation is pluggable; thresholds are configurable
"""

from __future__ import annotations

__all__ = [
    "utils",
    "dataset",
    "models",
]
