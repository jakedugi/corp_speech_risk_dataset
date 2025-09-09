"""
Very thin wrapper that adapts RSSLoader output to the standard
BaseExtractor → Cleaner → QuoteExtractor stages.

Almost everything is inherited; we only need to plug in the new loader.
"""

from __future__ import annotations

from .loader import DocumentLoader  # unchanged
from .rss_loader import RSSLoader
from .base_extractor import BaseExtractor


class RSSExtractor(BaseExtractor):
    LOADER_CLS = RSSLoader  # ← single line that wires us in
