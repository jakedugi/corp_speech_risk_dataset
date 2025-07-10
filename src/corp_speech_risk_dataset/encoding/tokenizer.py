from pathlib import Path
from typing import List, Optional

import sentencepiece as spm
from bpemb import BPEmb          # new
import json

_VS = 32000
_LANG = "en"
# First look inside the repo (useful on CI or after `git clean`)
_LOCAL_MODEL = Path(__file__).parents[3] / f"data/models/en.wiki.bpe.vs{_VS}.model"

class SentencePieceTokenizer:
    def __init__(self, model_path: Optional[str] = None):
        """
        Load a SentencePiece model.
        Order of preference  
            1. explicit *model_path* argument  
            2. data/models/en.wiki.bpe.vs32000.model inside the repo  
            3. BPEmb cache (auto-downloads on first use)
        """
        cand = Path(model_path) if model_path else _LOCAL_MODEL

        if not cand.is_file():                   # fall-back to BPEmb
            # ensure we download a *cased* model (preserve uppercase)
            bp = BPEmb(lang=_LANG, vs=_VS)
            cand = Path(bp.model_file)

        self.sp = spm.SentencePieceProcessor(model_file=str(cand))
        self._BYTE_OFFSET = 0x110000
        # Persist the byte fallback map for auditability
        mapping_path = cand.parent / "byte_fallback_map.json"
        if not mapping_path.exists():
            with open(mapping_path, "w", encoding="utf-8") as f:
                char_map = {i + self._BYTE_OFFSET: chr(i) for i in range(0x110000)}
                json.dump(char_map, f, ensure_ascii=False)

    def encode(self, text: str) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if 0 in ids:                       # SentencePiece would emit <unk>
            return [ord(ch) + self._BYTE_OFFSET for ch in text]
        return ids

    def decode(self, ids: List[int]) -> str:
        if ids and ids[0] >= self._BYTE_OFFSET:     # byte-fallback path
            return "".join(chr(i - self._BYTE_OFFSET) for i in ids)
        return self.sp.decode(ids)

    def encode_with_flag(self, text: str):
        """
        Returns (ids, used_fallback, fallback_chars), where:
          – used_fallback: True if we saw any <unk> at token-level
          – fallback_chars: list of the unique chars that triggered fallback
        """
        ids = self.sp.encode(text, out_type=int)
        if 0 not in ids:
            return ids, False, []
        # genuine OOV → fall back per-character
        fallback_chars = list(dict.fromkeys(text))  # unique in order
        ids = [ord(ch) + self._BYTE_OFFSET for ch in text]
        self._record_fallback_mappings(text)
        return ids, True, fallback_chars

    def _record_fallback_mappings(self, text: str):
        """
        For each character in text, ensure its fallback ID is in our JSON map.
        This writes only the entries we actually need—so we never touch
        surrogate code points outside real BMP chars.
        """
        map_path = Path(__file__).parents[3] / "data" / "models" / "byte_fallback_map.json"
        # load existing (or start fresh)
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                char_map = json.load(f)
        else:
            char_map = {}
        # add missing entries
        changed = False
        for ch in text:
            fid = ord(ch) + self._BYTE_OFFSET
            if str(fid) not in char_map:
                char_map[str(fid)] = ch
                changed = True
        if changed:
            map_path.parent.mkdir(parents=True, exist_ok=True)
            # ensure_ascii=False here will escape any surrogates properly
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(char_map, f, ensure_ascii=False)