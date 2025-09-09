"""Shared utilities for case aggregation.

This module contains helpers for:
- Robust whitespace tokenization and character→token index mapping
- Path parsing and discovery of case docket files
- Lightweight text normalization strategies

We do not depend on external libraries to keep first-run latency low.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from bisect import bisect_left
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WHITESPACE_RE = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    """Collapse all whitespace to single spaces without trimming punctuation.

    This keeps character alignment close to original while tolerating differences
    in newlines and multiple spaces.
    """
    return WHITESPACE_RE.sub(" ", text)


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text by whitespace.

    For the experiments described, simple whitespace tokenization is sufficient
    and fast. It aligns with downstream needs (relative ordering by tokens).
    """
    if not text:
        return []
    # Normalize only whitespace to maintain token boundaries across inputs
    return normalize_spaces(text).strip().split()


def char_to_token_index(text: str, start_char: int) -> int:
    """Map a character offset to a token index using whitespace tokenization.

    Strategy: count tokens in the prefix up to start_char. Uses the same
    normalization as `simple_tokenize` to minimize discrepancies.

    Args:
        text: Full text to index into
        start_char: Character offset (0-based) into `text`

    Returns:
        int: 0-based token index
    """
    safe_start = max(0, min(len(text), start_char))
    prefix = text[:safe_start]
    return len(simple_tokenize(prefix))


def find_stage1_jsonl_files(case_dir: str) -> List[Tuple[int, str]]:
    """Discover and order `stage1.jsonl` docket files within a case directory.

    The expected structure is `.../entries/entry_XXXX_documents/doc_*_stage1.jsonl`.
    We sort by the numeric entry id extracted from `entry_XXXX_documents`.

    Args:
        case_dir: Absolute path to a case root containing an `entries/` directory.

    Returns:
        List of tuples `(entry_id, file_path)` sorted ascending by `entry_id`.
    """
    pattern = os.path.join(case_dir, "entries", "*", "doc_*_stage1.jsonl")
    files = glob(pattern)
    results: List[Tuple[int, str]] = []
    entry_re = re.compile(r"entry_(\d+)_documents")
    for fpath in files:
        m = entry_re.search(fpath)
        if not m:
            continue
        entry_id = int(m.group(1))
        results.append((entry_id, fpath))
    results.sort(key=lambda x: x[0])
    return results


@dataclass(frozen=True)
class DocketIndex:
    """Holds ordered docket metadata for a case.

    Attributes:
        entry_ids: Ordered entry ids (ascending)
        file_paths: Corresponding file paths
        docket_numbers: Map from doc_id -> 1-based docket index
        doc_texts: Map from doc_id -> full docket text
        ordered_doc_ids: List of doc_ids in docket order (1..N)
        full_text: Concatenated case text joined with a single space between dockets
        docket_prefix_chars: List of starting char offsets in full_text for each docket
    """

    entry_ids: List[int]
    file_paths: List[str]
    docket_numbers: Dict[str, int]
    doc_texts: Dict[str, str]
    ordered_doc_ids: List[str]
    full_text: str
    docket_prefix_chars: List[int]
    # Performance metadata (precomputed)
    doc_token_starts: Dict[str, List[int]]
    docket_prefix_tokens: List[int]


def build_docket_index(case_dir: str) -> DocketIndex:
    """Load docket texts and construct ordered indices for a case.

    Assumes each `stage1.jsonl` contains a single JSON object with keys
    `doc_id` and `text`.
    """
    # Optional fast JSON
    try:  # orjson accelerates large JSON decode if available
        import orjson as _json

        def _loads(s: str) -> Dict:
            return _json.loads(
                s if isinstance(s, (bytes, bytearray)) else s.encode("utf-8")
            )

    except Exception:  # pragma: no cover - fallback to stdlib
        import json as _json

        def _loads(s: str) -> Dict:
            return _json.loads(s)

    ordered = find_stage1_jsonl_files(case_dir)
    if not ordered:
        raise FileNotFoundError(f"No stage1.jsonl files under {case_dir}/entries")

    entry_ids: List[int] = []
    file_paths: List[str] = []
    docket_numbers: Dict[str, int] = {}
    doc_texts: Dict[str, str] = {}
    ordered_doc_ids: List[str] = []
    doc_token_starts: Dict[str, List[int]] = {}

    for idx, (entry_id, fpath) in enumerate(ordered, start=1):
        with open(fpath, "r", encoding="utf-8") as fp:
            line = fp.readline().strip()
            if not line:
                continue
            data = _loads(line)
            doc_id = data.get("doc_id")
            text = data.get("text", "")
            if not doc_id:
                continue
            entry_ids.append(entry_id)
            file_paths.append(fpath)
            docket_numbers[doc_id] = idx
            doc_texts[doc_id] = text
            ordered_doc_ids.append(doc_id)
            # Precompute token start character offsets for fast char→token mapping
            # Tokens are maximal \S+ spans; boundaries are whitespace transitions.
            token_starts = [m.start() for m in re.finditer(r"\S+", text)]
            doc_token_starts[doc_id] = token_starts

    # Build full_text and per-docket starting char offsets
    full_parts: List[str] = []
    docket_prefix_chars: List[int] = []
    docket_prefix_tokens: List[int] = []
    running_chars = 0
    running_tokens = 0
    for doc_id in ordered_doc_ids:
        docket_prefix_chars.append(running_chars)
        docket_prefix_tokens.append(running_tokens)
        part = doc_texts.get(doc_id, "")
        full_parts.append(part)
        # +1 for the joining space that will be inserted between dockets
        running_chars += len(part) + 1
        running_tokens += len(doc_token_starts.get(doc_id, ()))
    full_text = " ".join(full_parts)

    # Construct index with token metadata
    idx = DocketIndex(
        entry_ids=entry_ids,
        file_paths=file_paths,
        docket_numbers=docket_numbers,
        doc_texts=doc_texts,
        ordered_doc_ids=ordered_doc_ids,
        full_text=full_text,
        docket_prefix_chars=docket_prefix_chars,
        doc_token_starts=doc_token_starts,
        docket_prefix_tokens=docket_prefix_tokens,
    )
    return idx


def compute_global_char_offset(
    docket_index: DocketIndex, doc_id: str, local_char_offset: int
) -> int:
    """Convert a local docket character offset to a global case character offset.

    The global text is constructed as a single-space join between docket texts.
    Thus global offset = prefix_chars[docket_number-1] + local_char_offset + num_join_spaces_before.
    Here num_join_spaces_before equals docket_number-1, but we already folded
    those into prefix_chars when constructing it (prefix computed cumulatively as
    len(part)+1 per docket), so we just add local_char_offset to the prefix.
    """
    if doc_id not in docket_index.docket_numbers:
        raise KeyError(f"Unknown doc_id: {doc_id}")
    docket_number = docket_index.docket_numbers[doc_id]
    prefix = docket_index.docket_prefix_chars[docket_number - 1]
    return prefix + local_char_offset


def fast_char_to_token_index(token_starts: List[int], start_char: int) -> int:
    """Map a character offset to token index using precomputed token start offsets.

    The token index equals the number of tokens with start < start_char.
    We use bisect_left to get O(log T) mapping.
    """
    if not token_starts:
        return 0
    if start_char <= token_starts[0]:
        return 0
    return bisect_left(token_starts, start_char)
