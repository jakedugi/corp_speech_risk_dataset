"""Case aggregation utilities package.

This package provides positional feature extraction for per-quote data
against raw case dockets (stage1.jsonl) and simple threshold helpers
for case-level aggregation experiments.

Modules:
- utils: Shared helpers for tokenization and path parsing.
- positional_features: Core logic to map quotes to docket and case positions.
- thresholds: Thresholding utilities for experiments (e.g., first half by tokens).
- cli: Command-line interface for batch processing quotes directories.

Design goals:
- Keep it simple and modular (KISS) for fast first inference runs.
- Use only standard library for extraction.
- Emphasize accuracy; avoid fuzzy matches that inflate false positives.
"""

__all__ = [
    "utils",
    "positional_features",
    "thresholds",
]
