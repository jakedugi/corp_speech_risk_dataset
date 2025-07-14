"""
Application Layer - Corporate Speech Risk Dataset

This layer contains use-cases and services that orchestrate domain logic.
It depends only on the domain layer and defines interfaces for adapters.

Key Principles:
- Use-case implementations
- Application services
- Interfaces for adapters (ports)
- Orchestration of domain objects
- No framework details

Contents:
- quote_extraction_pipeline.py: Core quote extraction use-case
- quote_extraction_config.py: Configuration for extraction
- courtlistener_orchestrator.py: Legal data orchestration
- Various run_*.py: Use-case runners

Dependencies: domain/ only
Dependents: adapters/, infrastructure/
"""

from .quote_extraction_pipeline import *
from .quote_extraction_config import *

__all__ = [
    "QuoteExtractionPipeline",
    "QuoteExtractionConfig",
    # Add other use-case classes as needed
]
