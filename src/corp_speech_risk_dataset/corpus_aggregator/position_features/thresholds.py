"""Threshold helpers for case-level evaluation.

Utilities to decide whether a quote should be included under different
holdout strategies:
- By docket order: keep only quotes from the first half/third of dockets
- By global token quota: keep only quotes whose start index is below a token
  budget (e.g., 2500 tokens)

Boundary handling rule:
- If a quote overlaps the threshold boundary (i.e., any part of the quote's
  tokens would fall beyond the threshold), exclude it. This ensures we never
  include partially-truncated quotes in thresholded subsets.
"""

from __future__ import annotations

from typing import Dict


def include_by_docket_fraction(
    quote: Dict, total_dockets: int, fraction: float
) -> bool:
    """Include quote if entirely within the first `fraction` of dockets.

    Excludes the quote if the docket_number exceeds the cut or if metadata is
    missing.
    """
    if not total_dockets or "docket_number" not in quote:
        return False
    cutoff = max(0, int(total_dockets * fraction))
    return quote["docket_number"] <= max(1, cutoff)


def include_by_global_token_budget(quote: Dict, token_budget: int) -> bool:
    """Include quote if its token span is entirely below `token_budget`.

    A quote with starting token `global_token_start` and length `num_tokens` must
    satisfy: global_token_start + num_tokens <= token_budget.
    """
    start = quote.get("global_token_start")
    length = quote.get("num_tokens")
    if start is None or length is None:
        return False
    return (start + length) <= token_budget
