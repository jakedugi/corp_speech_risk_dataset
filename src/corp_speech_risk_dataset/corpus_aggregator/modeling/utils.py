"""Streamlined utilities for case-level modeling.

Core functionality:
- Fast loading of quote JSONL files
- Case ID extraction from paths
- Data normalization for modeling
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import polars as pl

try:
    import orjson as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data)

    def _loads_str(data: str) -> Any:
        # orjson only accepts bytes; encode
        return _json.loads(data.encode("utf-8"))

except Exception:  # pragma: no cover
    import json as _json  # type: ignore

    def _loads_bytes(data: bytes) -> Any:  # type: ignore
        return _json.loads(data.decode("utf-8"))

    def _loads_str(data: str) -> Any:  # type: ignore
        return _json.loads(data)


# Pattern to extract case IDs from file paths
CASE_ID_RE = re.compile(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/")


def extract_case_id_from_src(src: str) -> Optional[str]:
    """Extract case ID from source path."""
    if not src:
        return None
    m = CASE_ID_RE.search(src)
    return m.group(1) if m else None


def load_quotes_dir(
    quotes_dir: str, pattern: str = "**/*.jsonl"
) -> List[Dict[str, Any]]:
    """Load quote dicts from a directory of JSONL files.

    - Supports recursive globbing using `pattern` relative to `quotes_dir`.
    - Each file may be JSONL or a single-line JSON array; handled by `load_jsonl`.
    - Adds `case_id` derived from `_src` when possible.
    """
    quotes_dir_abs = os.path.abspath(quotes_dir)
    file_pattern = os.path.join(quotes_dir_abs, pattern)
    files = [f for f in glob(file_pattern, recursive=True) if os.path.isfile(f)]
    logger.info("Scanning quotes files", count=len(files), root=quotes_dir_abs)
    rows: List[Dict[str, Any]] = []
    for fpath in files:
        try:
            items = _load_jsonl_fast(fpath)
        except Exception:  # pragma: no cover - defensive I/O handling
            logger.exception("Failed to load quotes file", path=fpath)
            continue
        for r in items:
            # Derive case_id from _src if not present
            if "case_id" not in r:
                src = r.get("_src") or r.get("src") or ""
                cid = extract_case_id_from_src(src) if isinstance(src, str) else None
                if cid:
                    r["case_id"] = cid
            rows.append(r)
    logger.info("Loaded quotes", num_rows=len(rows))
    return rows


def _load_jsonl_fast(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file or a single-line JSON array using orjson when available."""
    with open(path, "rb") as fp:
        data = fp.read().strip()
        if not data:
            return []
        # Try array first
        try:
            obj = _loads_bytes(data)
            if isinstance(obj, list):
                out: List[Dict[str, Any]] = []
                for x in obj:
                    if isinstance(x, dict):
                        out.append(x)  # type: ignore[arg-type]
                return out
        except Exception:
            pass
        # Fallback: iterate lines
        out2: List[Dict[str, Any]] = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _loads_bytes(line)
            except Exception:
                # As a fallback, try str loader (handles stray encodings)
                obj = _loads_str(line.decode("utf-8", errors="ignore"))
            if isinstance(obj, dict):
                out2.append(obj)  # type: ignore[arg-type]
        return out2


def select_relevant_columns(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    """Build a Polars DataFrame with the relevant columns for modeling.

    Ensures expected dtypes and fills missing columns with nulls.
    """
    if not rows:
        return pl.DataFrame([])

    # Normalize nested probability map if present
    normalized: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        probs = rr.get("coral_class_probs")
        if isinstance(probs, dict):
            low_val = probs.get("low")
            med_val = probs.get("medium")
            high_val = probs.get("high")
            if rr.get("coral_prob_low") is None and isinstance(low_val, (int, float)):
                rr["coral_prob_low"] = float(low_val)
            if rr.get("coral_prob_medium") is None and isinstance(
                med_val, (int, float)
            ):
                rr["coral_prob_medium"] = float(med_val)
            if rr.get("coral_prob_high") is None and isinstance(high_val, (int, float)):
                rr["coral_prob_high"] = float(high_val)
        # Normalize bucket string
        if isinstance(rr.get("coral_pred_bucket"), str):
            rr["coral_pred_bucket"] = str(rr["coral_pred_bucket"]).lower()
        normalized.append(rr)

    # Explicit dtype hints for list/object columns to avoid schema inference mismatches
    df = pl.DataFrame(normalized, strict=False)
    # If present, coerce coral_scores to List[Float64]
    if "coral_scores" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("coral_scores").is_not_null())
            .then(pl.col("coral_scores").cast(pl.List(pl.Float64), strict=False))
            .otherwise(pl.lit(None, dtype=pl.List(pl.Float64)))
            .alias("coral_scores")
        )
    # Ensure expected columns exist
    expected: List[Tuple[str, pl.PolarsDataType]] = [
        ("case_id", pl.Utf8),
        ("doc_id", pl.Utf8),
        ("docket_number", pl.Int64),
        ("docket_token_start", pl.Int64),
        ("global_token_start", pl.Int64),
        ("docket_char_start", pl.Int64),
        ("global_char_start", pl.Int64),
        ("num_tokens", pl.Int64),
        ("final_judgement_real", pl.Float64),
        ("coral_pred_bucket", pl.Utf8),
        ("coral_pred_class", pl.Int64),
        ("coral_confidence", pl.Float64),
        ("coral_prob_low", pl.Float64),
        ("coral_prob_medium", pl.Float64),
        ("coral_prob_high", pl.Float64),
        ("coral_model_threshold", pl.Float64),
    ]
    for col, dtype in expected:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
        else:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))

    return df
