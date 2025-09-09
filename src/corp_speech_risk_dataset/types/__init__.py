"""
corpus-types: Authoritative schemas, IDs, and validators for the corpus pipeline.

This module provides:
- Pydantic models for data contracts (Doc, Quote, Outcome, etc.)
- Deterministic ID generation functions
- JSON Schema validation and export
- CLI tools for data validation
"""

from .schemas.models import (
    Doc,
    Quote,
    Outcome,
    QuoteFeatures,
    CaseVector,
    Prediction,
    CasePrediction,
    Meta,
    Span,
    SchemaVersion,
    APIConfig,
    QuoteCandidate,
    QuoteRow,
)
from .ids.generate import doc_id, quote_id, case_id

__version__ = "0.1.0"
__all__ = [
    # Core models
    "Doc",
    "Quote",
    "Outcome",
    "QuoteFeatures",
    "CaseVector",
    "Prediction",
    "CasePrediction",
    "Meta",
    "Span",
    "SchemaVersion",
    # Legacy models
    "APIConfig",
    "QuoteCandidate",
    "QuoteRow",
    # ID functions
    "doc_id",
    "quote_id",
    "case_id",
]
