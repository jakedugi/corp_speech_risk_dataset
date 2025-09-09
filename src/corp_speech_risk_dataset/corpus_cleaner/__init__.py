"""
Text cleaning and normalization utilities for the corpus.

This module provides deterministic text normalization with offset mapping
to preserve span information across transformations.
"""

from .cleaner import TextCleaner

__all__ = ["TextCleaner"]
