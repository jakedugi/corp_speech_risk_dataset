"""
Abstract base classes and ports for the extraction domain.
Defines the core interfaces that all extractors must implement.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List
from ..models.quote_candidate import QuoteCandidate


class QuoteExtractor(ABC):
    """Abstract base class for all quote extraction implementations."""

    @abstractmethod
    def extract(self, doc_text: str) -> Iterator[QuoteCandidate]:
        """
        Extract quote candidates from document text.

        Args:
            doc_text: The full text of the document to process

        Yields:
            QuoteCandidate objects for each potential quote found
        """
        pass


class QuoteAttributor(ABC):
    """Abstract base class for quote attribution implementations."""

    @abstractmethod
    def filter(self, candidates: List[QuoteCandidate]) -> Iterator[QuoteCandidate]:
        """
        Filter and attribute speakers to quote candidates.

        Args:
            candidates: List of quote candidates to process

        Yields:
            QuoteCandidate objects with speaker attribution
        """
        pass


class QuoteReranker(ABC):
    """Abstract base class for quote reranking implementations."""

    @abstractmethod
    def rerank(self, candidates: List[QuoteCandidate]) -> Iterator[QuoteCandidate]:
        """
        Rerank quote candidates based on semantic similarity or other criteria.

        Args:
            candidates: List of attributed quote candidates

        Yields:
            QuoteCandidate objects meeting the ranking threshold
        """
        pass
