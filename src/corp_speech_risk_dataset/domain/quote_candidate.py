"""
A candidate quote, containing the text, its surrounding context, and any metadata
gathered during the extraction process.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class QuoteCandidate:
    """
    Represents a potential quote. This object is passed through the pipeline,
    where it is enriched with speaker, score, and other metadata.
    """
    quote: str
    context: str
    urls: List[str] = field(default_factory=list)
    speaker: Optional[str] = None
    score: float = 0.0

    def to_dict(self) -> dict:
        """Serializes the object to a dictionary."""
        return {
            "text": self.quote,
            "speaker": self.speaker,
            "score": self.score,
            "urls": self.urls,
            "context": self.context,
        }
