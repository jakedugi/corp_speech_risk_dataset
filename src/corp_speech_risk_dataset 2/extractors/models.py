from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Document:
    doc_id: str
    text: str
    source_path: str

@dataclass
class QuoteCandidate:
    quote: str
    context: str
    urls: List[str] = field(default_factory=list)
    speaker: Optional[str] = None
    score: Optional[float] = None 