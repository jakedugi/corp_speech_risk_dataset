from pathlib import Path
from typing import List, Tuple, Optional
from transformers import GPT2TokenizerFast
from loguru import logger


# decode for sentencepiece ids in the clustering step
def decode_sp_ids(sp_ids: list[int]) -> str:
    """Decode a list of BPE token IDs back to text."""
    return _SHARED_SP_TOKENIZER.decode(sp_ids)


class SentencePieceTokenizer:
    """
    Byte-level BPE tokenizer using GPT-2's vocabulary.

    Provides deterministic, lossless tokenization with zero OOV tokens.
    Maintains backward compatibility with the existing pipeline interface.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Load GPT-2 tokenizer.

        Args:
            model_path: Unused, kept for backward compatibility
        """
        if model_path is not None:
            logger.warning("model_path parameter ignored - using GPT-2 tokenizer")

        # use the shared GPT-2 byte-level BPE tokenizer
        self.tokenizer = _SHARED_SP_TOKENIZER
        logger.info("Using shared GPT-2 byte-level BPE tokenizer")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using byte-level BPE.

        Args:
            text: Input text to tokenize

        Returns:
            List of token IDs (always reversible, no OOV)
        """
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text (guaranteed lossless)
        """
        return self.tokenizer.decode(ids)

    def encode_with_flag(self, text: str) -> Tuple[List[int], bool, List[str]]:
        """
        Encode text with fallback information (now always False).

        Maintained for backward compatibility with existing pipeline.
        Byte-level BPE never needs fallback since all bytes are covered.

        Args:
            text: Input text to tokenize

        Returns:
            Tuple of (token_ids, used_fallback, fallback_chars)
            - used_fallback: Always False (no OOV possible)
            - fallback_chars: Always empty list
        """
        ids = self.encode(text)
        return ids, False, []


# ——— Shared tokenizer instance ———
# one shared GPT-2 byte-level BPE tokenizer for all use
_SHARED_SP_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
logger.info("Loaded GPT-2 byte-level BPE tokenizer (50,257 tokens) once at startup")
