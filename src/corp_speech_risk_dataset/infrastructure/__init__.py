"""
Infrastructure Layer - Corporate Speech Risk Dataset

This is the outermost layer containing framework and external system details.
It implements the adapters and provides concrete implementations.

Key Principles:
- Framework-specific code
- Database implementations
- HTTP clients and servers
- File system operations
- Third-party library integrations

Contents:
- api/: HTTP client implementations
- http_utils.py: HTTP utility functions
- file_io.py: File system operations
- nlp.py: NLP framework integrations
- nltk_setup.py: NLTK configuration
- resources/: External data dependencies

Dependencies: domain/, application/, adapters/
"""

# Infrastructure implementations will be imported here when needed
__all__ = [
    # HTTP clients, file operations, etc.
]
