"""
Shared Layer - Corporate Speech Risk Dataset

Cross-cutting concerns that can be used by any layer.
These are utilities that don't contain business logic.

Key Principles:
- No business logic
- Pure utilities and helpers
- Configuration management
- Logging infrastructure
- Common constants

Contents:
- logging_utils.py: Logging configuration and utilities
- config.py: Application configuration
- constants.py: Application constants
- discovery.py: Discovery utilities
- stage_writer.py: Stage writing utilities

Dependencies: None (pure utilities)
Dependents: All layers may use shared utilities
"""

# Shared utilities will be imported here when needed
__all__ = [
    # Logging, config, constants, etc.
]
