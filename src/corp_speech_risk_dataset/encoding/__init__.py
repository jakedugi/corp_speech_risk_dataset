"""
Encoding subpackage: lossless tokenization, dependency parsing, WL feature extraction, Legal-BERT embeddings, and similarity utilities for reversible, auditable NLP pipelines.
"""

from .legal_bert_embedder import (
    LegalBertEmbedder,
    get_legal_bert_embedder,
    get_legal_bert_embeddings,
)

__all__ = ["LegalBertEmbedder", "get_legal_bert_embedder", "get_legal_bert_embeddings"]
