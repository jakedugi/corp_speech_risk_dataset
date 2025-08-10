"""
fully_interpretable
===================

Interpretable baseline models that mirror the `coral_ordinal` data ingestion
interface but avoid any fused/learned embeddings. These models rely on
fully-explainable features such as TF‑IDF over raw text and optional scalar
priors derived from `raw_features`.

Exposed entry points:
- `fully_interpretable.cli`: Command‑line interface for training/evaluating
  Ridge (linear), Multinomial Naive Bayes, and Decision Tree baselines.
- `fully_interpretable.pipeline`: Dataset loader and sklearn pipelines.
"""

from __future__ import annotations

__all__ = [
    "pipeline",
]
