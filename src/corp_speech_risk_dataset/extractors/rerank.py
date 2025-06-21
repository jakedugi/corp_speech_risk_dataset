from sentence_transformers import SentenceTransformer, util
from .models import QuoteCandidate
from typing import List, Iterator

class SemanticReranker:
    def __init__(self, seed_quotes: List[str], threshold: float = .55):
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.seed_vec = self.model.encode(seed_quotes, normalize_embeddings=True)
        self.thresh = threshold

    def rerank(self, candidates: List[QuoteCandidate]) -> Iterator[QuoteCandidate]:
        for qc in candidates:
            emb = self.model.encode(qc.quote, normalize_embeddings=True)
            score = float(util.cos_sim(emb, self.seed_vec).max())
            if score >= self.thresh:
                qc.score = score
                yield qc 