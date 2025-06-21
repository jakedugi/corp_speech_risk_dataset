"""Quote model for extracted quotes."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Quote:
    """Represents an extracted quote from a document."""
    
    quote: str
    speaker: Optional[str] = None
    score: float = 0.0
    urls: List[str] = None
    
    def __post_init__(self):
        if self.urls is None:
            self.urls = []
    
    def dict(self):
        """Convert to dictionary for serialization."""
        return {
            "quote": self.quote,
            "speaker": self.speaker,
            "score": self.score,
            "urls": self.urls
        } 