"""
Corpus extractors module for quote and outcome extraction.

This module provides functionality to extract quotes and outcomes from
normalized documents, with stable ID schemes and deterministic spans.
"""

from .quote_extractor import QuoteExtractor
from .first_pass import FirstPassExtractor
from .attribution import Attributor
from .rerank import SemanticReranker
from .final_pass_filter import filter_speakers, filter_heuristics
from .base_extractor import BaseExtractor
from .case_outcome_imputer import (
    scan_stage1,
    impute_for_case,
    AmountSelector,
    ManualAmountSelector,
)

# from .extract_cash_amounts_stage1 import extract_cash_amounts_stage1  # Function not found
# from .final_evaluate import evaluate_outcomes  # Functions not found

__all__ = [
    "QuoteExtractor",
    "FirstPassExtractor",
    "Attributor",
    "SemanticReranker",
    "filter_speakers",
    "filter_heuristics",
    "BaseExtractor",
    "scan_stage1",
    "impute_for_case",
    "AmountSelector",
    "ManualAmountSelector",
    # "extract_cash_amounts_stage1",  # Function not found    # "evaluate_outcomes",  # Functions not found
]
