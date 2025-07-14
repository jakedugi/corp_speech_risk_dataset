"""
Domain Layer - Corporate Speech Risk Dataset

This layer contains pure business objects, value types, and business logic
with NO external dependencies. This is the innermost layer of the Clean
Architecture concentric circles.

Key Principles:
- Pure business logic only
- No frameworks, databases, or external services
- Value objects and entities
- Domain services and business rules

Contents:
- quote_candidate.py: Core business entity for quote processing
- base_types.py: Value objects and types
- risk_features.py: Risk calculation business logic

Dependencies: NONE (this is the dependency-free core)
"""

from .quote_candidate import *
from .base_types import *
from .risk_features import *

__all__ = [
    # Quote entities
    "QuoteCandidate",

    # Risk calculations
    "calculate_risk_vector_counts",
    "counts_to_sparse_vector",
    "get_feature_dimension",

    # Base types (if any)
]
