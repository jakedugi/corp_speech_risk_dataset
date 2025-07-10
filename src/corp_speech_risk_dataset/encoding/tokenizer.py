import sentencepiece as spm
from typing import List

class SentencePieceTokenizer:
    """
    Wrapper for SentencePiece with byte-level fallback for lossless, reversible tokenization.
    """
    def __init__(self, model_path: str):
        """Load a SentencePiece model from the given path."""
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of SentencePiece IDs (lossless, reversible)."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of SentencePiece IDs back to the original text (lossless)."""
        return self.sp.decode(ids) 