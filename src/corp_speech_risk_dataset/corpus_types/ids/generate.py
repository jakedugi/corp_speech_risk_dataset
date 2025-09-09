"""
Deterministic ID generation utilities for corpus-types.

This module provides functions for generating consistent, deterministic IDs
for documents, quotes, and cases using BLAKE3 hashing for optimal performance
and collision resistance.
"""

from typing import Optional
from urllib.parse import urlparse

try:
    from blake3 import blake3
except ImportError:
    # Fallback to hashlib.sha256 if blake3 is not available
    import hashlib

    def blake3(data):
        return hashlib.sha256(data)


def _b3(content: str) -> str:
    """
    Generate a BLAKE3 hash of the content, returning a URL-safe base64 string.

    Args:
        content: String content to hash

    Returns:
        URL-safe base64 encoded hash (22 characters for 128-bit hash)
    """
    import base64

    hash_bytes = blake3(content.encode("utf-8")).digest()[:16]  # 128-bit hash
    return base64.urlsafe_b64encode(hash_bytes).decode("ascii").rstrip("=")


def doc_id(source_uri: str, retrieved_at_iso: str, court: Optional[str] = None) -> str:
    """
    Generate a deterministic document ID from source URI and retrieval time.

    Design principles:
    - Content-addressed: IDs are BLAKE3 hashes of normalized, stable tuples
    - Include namespace fields to avoid collisions across sources
    - Use short prefix to indicate entity type

    Args:
        source_uri: The source URI of the document
        retrieved_at_iso: ISO timestamp string (e.g., "2024-01-01T00:00:00Z")
        court: Optional court identifier for additional uniqueness

    Returns:
        Deterministic document ID with 'doc_' prefix
    """
    if not source_uri:
        raise ValueError("source_uri cannot be empty")

    # Normalize URI for consistency
    parsed = urlparse(source_uri)
    normalized_uri = parsed._replace(fragment="").geturl()  # Remove fragments

    # Create stable key: doc|uri|timestamp|court
    key = "|".join(["doc", normalized_uri, retrieved_at_iso, court or ""])
    return f"doc_{_b3(key)}"


def quote_id(doc_id: str, start: int, end: int, text_norm: str) -> str:
    """
    Generate a deterministic quote ID from document ID, span, and normalized text.

    The span + normalized text anchors the quote uniquely within its document.
    Text normalization (lowercase, collapse whitespace) should be done by caller.

    Args:
        doc_id: Document ID this quote belongs to
        start: Start character position in document
        end: End character position in document (exclusive)
        text_norm: Normalized quote text content

    Returns:
        Deterministic quote ID with 'q_' prefix
    """
    if not doc_id:
        raise ValueError("doc_id cannot be empty")
    if start < 0 or end < start:
        raise ValueError("Invalid span coordinates")
    if not text_norm or not text_norm.strip():
        raise ValueError("text_norm cannot be empty")

    # Create stable key: quote|doc_id|start|end|normalized_text
    key = "|".join(["quote", doc_id, str(start), str(end), text_norm.strip()])
    return f"q_{_b3(key)}"


def case_id(court: str, docket: str) -> str:
    """
    Generate a deterministic case ID from court and docket information.

    Args:
        court: Court identifier (e.g., 'scotus', 'ca1', 'nysupct')
        docket: Docket number string

    Returns:
        Deterministic case ID with 'case_' prefix
    """
    if not court or not court.strip():
        raise ValueError("court cannot be empty")
    if not docket or not docket.strip():
        raise ValueError("docket cannot be empty")

    # Create stable key: case|court|docket
    key = "|".join(["case", court.strip().lower(), docket.strip()])
    return f"case_{_b3(key)}"


def validate_id_format(id_str: str, expected_prefix: str) -> bool:
    """
    Validate that an ID string has the correct format and prefix.

    Args:
        id_str: ID string to validate
        expected_prefix: Expected prefix (e.g., 'doc_', 'q_', 'case_')

    Returns:
        True if format is valid
    """
    if not id_str or not isinstance(id_str, str):
        return False

    if not id_str.startswith(expected_prefix):
        return False

    # Check that the hash part looks like a base64url string
    hash_part = id_str[len(expected_prefix) :]
    try:
        import string

        # Should be valid base64url characters (no padding)
        valid_chars = string.ascii_letters + string.digits + "-_"
        if not all(c in valid_chars for c in hash_part):
            return False
        return len(hash_part) == 22  # BLAKE3 128-bit hash encoded length
    except:
        return False


def extract_namespace(id_str: str) -> Optional[str]:
    """
    Extract the namespace from an ID string.

    Args:
        id_str: ID string to parse

    Returns:
        Namespace prefix or None if invalid
    """
    if "_" not in id_str:
        return None

    prefix = id_str.split("_", 1)[0]
    if prefix in ["doc", "q", "case"]:
        return prefix

    return None


# Legacy function aliases for backward compatibility
def generate_doc_id(source_uri: str, retrieved_at: Optional[str] = None) -> str:
    """Legacy alias for doc_id."""
    return doc_id(source_uri, retrieved_at or "", None)


def generate_quote_id(doc_id: str, span_start: int, span_end: int, text: str) -> str:
    """Legacy alias for quote_id."""
    return quote_id(doc_id, span_start, span_end, text)


def generate_case_id(court: str, docket_number: str, year: Optional[int] = None) -> str:
    """Legacy alias for case_id."""
    return case_id(court, docket_number)
