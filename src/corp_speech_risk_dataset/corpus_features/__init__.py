"""
Corpus Features Module

This module provides feature extraction and encoding functionality for quotes.
It takes Quote JSONL files as input and produces QuoteFeatures JSONL files as output.

Main components:
- Feature encoders (embeddings, interpretable features, etc.)
- Feature validation and quality checks
- CLI interface for feature extraction
- Feature registry for version management
"""

from .features_case_agnostic import (
    extract_evidential_lexicons,
    extract_sentiment,
    embed_text,
    fuse_features,
)
from .binary_pipeline import BinaryFeaturePipeline
from .validation_pipeline import UnifiedFeatureValidator
from .tertile_pipeline import TertileFeaturePipeline
from .encoders.legal_bert_embedder import LegalBertEmbedder
from .encoders.graphembedder import GraphEmbedder
from .registry import FeatureRegistry, registry

__all__ = [
    "extract_evidential_lexicons",
    "extract_sentiment",
    "embed_text",
    "fuse_features",
    "BinaryFeaturePipeline",
    "UnifiedFeatureValidator",
    "TertileFeaturePipeline",
    "LegalBertEmbedder",
    "GraphEmbedder",
    "FeatureRegistry",
    "registry",
]
