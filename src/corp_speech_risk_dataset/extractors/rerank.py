"""
The SemanticReranker, which scores quote candidates based on their semantic
similarity to a set of seed quotes.
"""

from sentence_transformers import SentenceTransformer, util
from typing import List, Iterator
from ..models.quote_candidate import QuoteCandidate
from ..domain.ports import QuoteReranker


class SemanticReranker(QuoteReranker):
    """
    Uses a SentenceTransformer model to encode quotes and score them against
    seed examples, filtering out those below a given threshold.
    """

    def __init__(self, seed_quotes: List[str], threshold: float = 0.55):
        """
        Initializes the reranker with seed quotes and a similarity threshold.
        """
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.seed_vec = self.model.encode(seed_quotes, normalize_embeddings=True)
        self.thresh = threshold

    def rerank(self, candidates: List[QuoteCandidate]) -> Iterator[QuoteCandidate]:
        """
        Reranks and filters candidates based on semantic similarity.

        Args:
            candidates: A list of attributed QuoteCandidate objects.

        Yields:
            QuoteCandidate objects that meet the similarity threshold, with their
            score updated.
        """
        for qc in candidates:
            emb = self.model.encode(qc.quote, normalize_embeddings=True)
            score = float(util.cos_sim(emb, self.seed_vec).max())
            if score >= self.thresh:
                qc.score = score
                yield qc
