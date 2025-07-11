from pydantic import BaseModel
from typing import List, Tuple, Optional

class QuoteRow(BaseModel):
    """
    Pydantic model for a single quote row, preserving all original metadata and new encoding outputs.
    This ensures lossless, auditable, and extensible data for downstream processing.
    """
    # Original metadata fields
    doc_id: str
    stage: int
    text: str
    context: Optional[str] = None
    speaker: Optional[str] = None
    score: Optional[float] = None
    urls: List[str] = []
    _src: Optional[str] = None

    # Encoding outputs
    sp_ids: List[int]                      # SentencePiece IDs (loss-less)
    deps: List[Tuple[int, int, str]]       # Dependency edges (head, child, label)
    wl_indices: List[int]                  # CSR indices of WL feature vector
    wl_counts: List[int]                   # CSR data of WL feature vector 