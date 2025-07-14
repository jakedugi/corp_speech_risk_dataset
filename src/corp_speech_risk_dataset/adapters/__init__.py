"""
Adapters Layer - Corporate Speech Risk Dataset

This layer translates between the outside world and the application.
It implements the interfaces defined by the application layer.

Key Principles:
- Implements application layer interfaces
- Translates external formats to/from domain objects
- Contains both inbound and outbound adapters
- No business logic (only translation)

Structure:
- inbound/: User interfaces (CLI, web controllers)
- outbound/: External system interfaces (databases, APIs, file systems)

Dependencies: domain/, application/
Dependents: infrastructure/
"""

# Import adapters as needed
# from .inbound import *
# from .outbound import *

__all__ = [
    # Will be populated with adapter interfaces
]
