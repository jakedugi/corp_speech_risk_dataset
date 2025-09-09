"""Positional feature extraction for quotes against case dockets.

Core functionality:
- Index a case's ordered dockets (stage1.jsonl)
- Find each quote's location using robust, conservative matching
- Compute docket_number, docket_token_start, and global_token_start
- Respect threshold boundary rules (exclude quotes whose tokens are cut off)

False positive budget: keep <1% by requiring exact/ignorecase/normalized-space
matches, optionally constrained by provided context. Ambiguous or low-confidence
matches are skipped and annotated with an error.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from .utils import (
    DocketIndex,
    build_docket_index,
    char_to_token_index,
    compute_global_char_offset,
    normalize_spaces,
    simple_tokenize,
    fast_char_to_token_index,
)


MATCH_FLAGS = re.IGNORECASE


@dataclass
class QuotePosition:
    """Computed position features for a single quote.

    Attributes:
        docket_number: 1-based order of the docket in the case
        docket_token_start: 0-based token index within the docket text
        global_token_start: 0-based token index within the concatenated case text
        docket_char_start: Character start within docket (for diagnostics)
        global_char_start: Character start within case (for diagnostics)
        num_tokens: Number of tokens in the quote text (for thresholds)
    """

    docket_number: int
    docket_token_start: int
    global_token_start: int
    docket_char_start: int
    global_char_start: int
    num_tokens: int


def _try_find(haystack: str, needle: str) -> int:
    """Conservative multi-strategy search returning start index or -1.

    Strategies in order:
    1) Exact substring
    2) Case-insensitive regex of the escaped needle
    3) Normalized-spacing, lowercase substring (maps back approximately)
    """
    if not needle:
        return -1

    # 1) Exact
    idx = haystack.find(needle)
    if idx != -1:
        return idx

    # 2) Ignore case
    m = re.search(re.escape(needle), haystack, MATCH_FLAGS)
    if m:
        return m.start()

    # 3) Normalize spaces + lowercase
    norm_h = normalize_spaces(haystack).lower()
    norm_n = normalize_spaces(needle).lower()
    norm_idx = norm_h.find(norm_n)
    if norm_idx != -1:
        # Best-effort mapping: find the first non-space boundary after collapsing
        # spaces. We approximate by scanning original haystack to find a substring
        # that matches ignoring case and with flexible whitespace.
        # Build a regex that treats any whitespace in needle as flexible \s+
        flex = re.sub(r"\s+", r"\\s+", re.escape(needle.strip()))
        m2 = re.search(flex, haystack, MATCH_FLAGS)
        if m2:
            return m2.start()
    return -1


def _locate_quote_in_docket(
    docket_text: str, quote_text: str, context: Optional[str]
) -> Optional[int]:
    """Locate the starting char index of the quote within the docket text.

    Primary search is by the quote text alone. If that fails, use context to
    restrict the search window and retry.
    """
    qt = (quote_text or "").strip()
    if qt:
        direct = _try_find(docket_text, qt)
        if direct != -1:
            return direct

    # Fallback: use context to narrow search
    if context:
        ctx_start = _try_find(docket_text, context.strip())
        if ctx_start != -1:
            # Try to find the quote within the context span
            rel = context.find(qt) if qt else -1
            if rel != -1:
                return ctx_start + rel
            # Search a local window around ctx_start
            window = docket_text[
                max(0, ctx_start - 500) : ctx_start + len(context) + 500
            ]
            rel2 = _try_find(window, qt)
            if rel2 != -1:
                return max(0, ctx_start - 500) + rel2

    return -1


def compute_quote_position(
    docket_index: DocketIndex, quote: Dict[str, Any]
) -> Tuple[Optional[QuotePosition], Optional[str]]:
    """Compute positional features for a single quote dict.

    The quote dict must include `doc_id` and `text`. Optionally `context` helps
    matching accuracy. Returns (position, error) where either may be None.
    """
    doc_id_raw = quote.get("doc_id")
    if not isinstance(doc_id_raw, str):
        return None, "Docket not found for doc_id"
    doc_id: str = doc_id_raw
    quote_text_raw = quote.get("text", "")
    if not isinstance(quote_text_raw, str) or not quote_text_raw:
        return None, "Empty quote text"
    quote_text: str = quote_text_raw
    context_raw = quote.get("context")
    context: Optional[str] = context_raw if isinstance(context_raw, str) else None
    if doc_id not in docket_index.docket_numbers:
        return None, "Docket not found for doc_id"

    docket_text: str = docket_index.doc_texts.get(doc_id, "")
    start_char = _locate_quote_in_docket(docket_text, quote_text, context)
    if start_char is None or start_char < 0:
        return None, "Low confidence match"

    docket_number = docket_index.docket_numbers[doc_id]
    global_char = compute_global_char_offset(docket_index, doc_id, start_char)
    # Prefer fast O(log T) mapping using precomputed starts if available
    doc_token_starts = getattr(docket_index, "doc_token_starts", {}).get(doc_id)
    if doc_token_starts is not None:
        docket_tok = fast_char_to_token_index(doc_token_starts, start_char)
        # Global token = tokens before this docket + docket_tok
        prefix_tokens = getattr(docket_index, "docket_prefix_tokens", [0])
        docket_number = docket_index.docket_numbers[doc_id]
        global_tok = prefix_tokens[docket_number - 1] + docket_tok
    else:
        docket_tok = char_to_token_index(docket_text, start_char)
        global_tok = char_to_token_index(docket_index.full_text, global_char)
    num_tokens = len(simple_tokenize(quote_text))

    return (
        QuotePosition(
            docket_number=docket_number,
            docket_token_start=docket_tok,
            global_token_start=global_tok,
            docket_char_start=start_char,
            global_char_start=global_char,
            num_tokens=num_tokens,
        ),
        None,
    )


def append_positional_features(
    case_dir: str, quotes: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    """Append positional features to a list of quote dicts for a single case.

    Adds the following fields on success:
    - docket_number
    - docket_token_start
    - global_token_start
    - docket_char_start (diagnostics)
    - global_char_start (diagnostics)
    - num_tokens (quote length)

    On failure, adds `feature_error` and leaves other fields untouched.
    """
    docket_index = build_docket_index(case_dir)
    out: List[Dict[str, Any]] = []
    for q in quotes:
        q_typed: Dict[str, Any] = dict(q)
        pos, err = compute_quote_position(docket_index, q_typed)
        if err:
            q_typed["feature_error"] = err
        else:
            assert pos is not None
            q_typed["docket_number"] = pos.docket_number
            q_typed["docket_token_start"] = pos.docket_token_start
            q_typed["global_token_start"] = pos.global_token_start
            q_typed["docket_char_start"] = pos.docket_char_start
            q_typed["global_char_start"] = pos.global_char_start
            q_typed["num_tokens"] = pos.num_tokens
        out.append(q_typed)
    return out


def load_jsonl(path: str) -> List[Dict[str, object]]:
    """Load a JSONL file containing quote dicts.

    If the file contains a single line JSON array, this function also handles it.
    """
    # Optional fast JSON
    try:
        import orjson as _json

        def _loads(s: bytes) -> object:
            return _json.loads(s)

    except Exception:  # pragma: no cover
        import json as _json

        def _loads(s: bytes) -> object:
            return _json.loads(s.decode("utf-8"))

    records: List[Dict[str, object]] = []
    with open(path, "rb") as fp:
        data = fp.read().strip()
        if not data:
            return records
        # Try JSON array first
        try:
            obj = _loads(data)
            if isinstance(obj, list):
                typed: List[Dict[str, Any]] = []
                for x in obj:
                    if isinstance(x, dict):
                        typed.append(cast(Dict[str, Any], x))
                return typed
        except Exception:
            pass
        # Fallback: JSONL
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            item = _loads(line)
            if isinstance(item, dict):
                records.append(cast(Dict[str, Any], item))
    return records


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    # Optional fast JSON
    try:
        import orjson as _json

        def _dumps(obj: Dict[str, Any]) -> bytes:
            return _json.dumps(obj, option=_json.OPT_APPEND_NEWLINE)

    except Exception:  # pragma: no cover
        import json as _json

        def _dumps(obj: Dict[str, Any]) -> bytes:
            return (_json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        for r in rows:
            fp.write(_dumps(r))
