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
- courtlistener_orchestrator.py: Legal data orchestration
- Various run_*.py: Use-case runners

Dependencies: domain/ only
Dependents: adapters/, infrastructure/
"""

# Configuration is now centralized in orchestrators/
# from ..orchestrators.quote_extraction_config import *

__all__ = [
    # Add use-case classes as needed
]
