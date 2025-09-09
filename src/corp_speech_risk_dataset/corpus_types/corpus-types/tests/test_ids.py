"""
Tests for ID generation utilities in corpus-types.
"""

import pytest
import sys
from pathlib import Path

# Add the main project src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from corp_speech_risk_dataset.types.ids.generate import (
    doc_id,
    quote_id,
    case_id,
    validate_id_format,
    extract_namespace,
    generate_doc_id,
    generate_quote_id,
    generate_case_id,
)


class TestDocId:
    """Test document ID generation."""

    def test_doc_id_basic(self):
        """Test basic doc ID generation."""
        result = doc_id("https://example.com/case1", "2024-01-01T00:00:00Z", "N.D.Cal")
        assert result.startswith("doc_")
        assert validate_id_format(result, "doc_")
        assert extract_namespace(result) == "doc"

    def test_doc_id_deterministic(self):
        """Test that same inputs produce same doc ID."""
        uri = "https://example.com/case1"
        timestamp = "2024-01-01T00:00:00Z"
        court = "N.D.Cal"

        result1 = doc_id(uri, timestamp, court)
        result2 = doc_id(uri, timestamp, court)
        assert result1 == result2

    def test_doc_id_different_inputs(self):
        """Test that different inputs produce different IDs."""
        base = doc_id("https://example.com/case1", "2024-01-01T00:00:00Z", "N.D.Cal")

        # Different URI
        assert (
            doc_id("https://example.com/case2", "2024-01-01T00:00:00Z", "N.D.Cal")
            != base
        )

        # Different timestamp
        assert (
            doc_id("https://example.com/case1", "2024-01-02T00:00:00Z", "N.D.Cal")
            != base
        )

        # Different court
        assert (
            doc_id("https://example.com/case1", "2024-01-01T00:00:00Z", "D.Mass")
            != base
        )

    def test_doc_id_no_court(self):
        """Test doc ID generation without court."""
        result = doc_id("https://example.com/case1", "2024-01-01T00:00:00Z")
        assert result.startswith("doc_")
        assert validate_id_format(result, "doc_")

    def test_doc_id_validation(self):
        """Test doc ID input validation."""
        # Empty URI
        with pytest.raises(ValueError, match="source_uri cannot be empty"):
            doc_id("", "2024-01-01T00:00:00Z")


class TestQuoteId:
    """Test quote ID generation."""

    def test_quote_id_basic(self):
        """Test basic quote ID generation."""
        doc_id_val = "doc_AAAAAAAAAAAAAAAAAAAAAA"
        result = quote_id(doc_id_val, 10, 20, "we did nothing wrong.")
        assert result.startswith("q_")
        assert validate_id_format(result, "q_")
        assert extract_namespace(result) == "q"

    def test_quote_id_deterministic(self):
        """Test that same inputs produce same quote ID."""
        doc_id_val = "doc_AAAAAAAAAAAAAAAAAAAAAA"
        start, end = 10, 20
        text = "we did nothing wrong."

        result1 = quote_id(doc_id_val, start, end, text)
        result2 = quote_id(doc_id_val, start, end, text)
        assert result1 == result2

    def test_quote_id_different_inputs(self):
        """Test that different inputs produce different IDs."""
        base = quote_id("doc_AAAAAAAAAAAAAAAAAAAAAA", 10, 20, "we did nothing wrong.")

        # Different doc_id
        assert (
            quote_id("doc_BBBBBBBBBBBBBBBBBBBBBB", 10, 20, "we did nothing wrong.")
            != base
        )

        # Different span
        assert (
            quote_id("doc_AAAAAAAAAAAAAAAAAAAAAA", 15, 25, "we did nothing wrong.")
            != base
        )

        # Different text
        assert (
            quote_id("doc_AAAAAAAAAAAAAAAAAAAAAA", 10, 20, "we did something wrong.")
            != base
        )

    def test_quote_id_validation(self):
        """Test quote ID input validation."""
        # Empty doc_id
        with pytest.raises(ValueError, match="doc_id cannot be empty"):
            quote_id("", 10, 20, "text")

        # Invalid span
        with pytest.raises(ValueError, match="Invalid span coordinates"):
            quote_id("doc_AAAAAAAAAAAAAAAAAAAAAA", 20, 10, "text")

        # Empty text
        with pytest.raises(ValueError, match="text_norm cannot be empty"):
            quote_id("doc_AAAAAAAAAAAAAAAAAAAAAA", 10, 20, "")


class TestCaseId:
    """Test case ID generation."""

    def test_case_id_basic(self):
        """Test basic case ID generation."""
        result = case_id("N.D.Cal", "1:24-cv-00001")
        assert result.startswith("case_")
        assert validate_id_format(result, "case_")
        assert extract_namespace(result) == "case"

    def test_case_id_deterministic(self):
        """Test that same inputs produce same case ID."""
        court = "N.D.Cal"
        docket = "1:24-cv-00001"

        result1 = case_id(court, docket)
        result2 = case_id(court, docket)
        assert result1 == result2

    def test_case_id_different_inputs(self):
        """Test that different inputs produce different IDs."""
        base = case_id("N.D.Cal", "1:24-cv-00001")

        # Different court
        assert case_id("D.Mass", "1:24-cv-00001") != base

        # Different docket
        assert case_id("N.D.Cal", "1:24-cv-00002") != base

    def test_case_id_validation(self):
        """Test case ID input validation."""
        # Empty court
        with pytest.raises(ValueError, match="court cannot be empty"):
            case_id("", "1:24-cv-00001")

        # Empty docket
        with pytest.raises(ValueError, match="docket cannot be empty"):
            case_id("N.D.Cal", "")


class TestValidateIdFormat:
    """Test ID format validation."""

    def test_validate_valid_ids(self):
        """Test validation of valid IDs."""
        # Generate some valid IDs and test them
        doc_id_val = doc_id("https://example.com/test", "2024-01-01T00:00:00Z")
        quote_id_val = quote_id("doc_AAAAAAAAAAAAAAAAAAAAAA", 10, 20, "test")
        case_id_val = case_id("N.D.Cal", "1:24-cv-00001")

        assert validate_id_format(doc_id_val, "doc_")
        assert validate_id_format(quote_id_val, "q_")
        assert validate_id_format(case_id_val, "case_")

    def test_validate_invalid_ids(self):
        """Test validation of invalid IDs."""
        # Wrong prefix
        assert not validate_id_format("doc_AAAAAAAAAAAAAAAAAAAAAA", "q_")

        # Empty string
        assert not validate_id_format("", "doc_")

        # None input
        assert not validate_id_format(None, "doc_")

        # Wrong format
        assert not validate_id_format("invalid", "doc_")


class TestExtractNamespace:
    """Test namespace extraction."""

    def test_extract_valid_namespaces(self):
        """Test extracting valid namespaces."""
        assert extract_namespace("doc_AAAAAAAAAAAAAAAAAAAAAA") == "doc"
        assert extract_namespace("q_BBBBBBBBBBBBBBBBBBBBB") == "q"
        assert extract_namespace("case_CCCCCCCCCCCCCCCCCCCCC") == "case"

    def test_extract_invalid_namespaces(self):
        """Test extracting invalid namespaces."""
        assert extract_namespace("invalid_AAAAAAAAAAAAAAAAAAAAAA") is None
        assert extract_namespace("no_underscore") is None
        assert extract_namespace("") is None


class TestLegacyAliases:
    """Test legacy function aliases for backward compatibility."""

    def test_legacy_doc_id(self):
        """Test legacy generate_doc_id function."""
        result = generate_doc_id("https://example.com/test")
        assert result.startswith("doc_")
        assert validate_id_format(result, "doc_")

    def test_legacy_quote_id(self):
        """Test legacy generate_quote_id function."""
        result = generate_quote_id("doc_AAAAAAAAAAAAAAAAAAAAAA", 10, 20, "test")
        assert result.startswith("q_")
        assert validate_id_format(result, "q_")

    def test_legacy_case_id(self):
        """Test legacy generate_case_id function."""
        result = generate_case_id("N.D.Cal", "1:24-cv-00001")
        assert result.startswith("case_")
        assert validate_id_format(result, "case_")
