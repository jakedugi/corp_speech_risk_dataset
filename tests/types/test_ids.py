"""
Tests for ID generation utilities in corpus-types.
"""

import pytest
from corp_speech_risk_dataset.types.ids.generate import (
    generate_doc_id,
    generate_quote_id,
    generate_case_id,
    validate_id_format,
    extract_namespace,
    _hash_string,
)


class TestHashString:
    """Test the internal hash string function."""

    def test_hash_string_deterministic(self):
        """Test that hash_string produces consistent results."""
        content = "test content"
        result1 = _hash_string(content)
        result2 = _hash_string(content)
        assert result1 == result2

    def test_hash_string_different_inputs(self):
        """Test that different inputs produce different hashes."""
        result1 = _hash_string("content1")
        result2 = _hash_string("content2")
        assert result1 != result2

    def test_hash_string_with_namespace(self):
        """Test hash_string with namespace."""
        result1 = _hash_string("content", "ns1")
        result2 = _hash_string("content", "ns2")
        assert result1 != result2


class TestGenerateDocId:
    """Test document ID generation."""

    def test_generate_doc_id_basic(self):
        """Test basic doc ID generation."""
        uri = "https://example.com/doc.pdf"
        doc_id = generate_doc_id(uri)
        assert doc_id.startswith("doc:")
        assert validate_id_format(doc_id, "doc:")
        assert extract_namespace(doc_id) == "doc"

    def test_generate_doc_id_with_timestamp(self):
        """Test doc ID generation with timestamp."""
        uri = "https://example.com/doc.pdf"
        timestamp = "2024-01-01T12:00:00Z"
        doc_id = generate_doc_id(uri, timestamp)
        assert doc_id.startswith("doc:")
        assert validate_id_format(doc_id, "doc:")

    def test_generate_doc_id_deterministic(self):
        """Test that same inputs produce same doc ID."""
        uri = "https://example.com/doc.pdf"
        doc_id1 = generate_doc_id(uri)
        doc_id2 = generate_doc_id(uri)
        assert doc_id1 == doc_id2

    def test_generate_doc_id_different_uris(self):
        """Test that different URIs produce different IDs."""
        uri1 = "https://example.com/doc1.pdf"
        uri2 = "https://example.com/doc2.pdf"
        doc_id1 = generate_doc_id(uri1)
        doc_id2 = generate_doc_id(uri2)
        assert doc_id1 != doc_id2

    def test_generate_doc_id_empty_uri(self):
        """Test that empty URI raises error."""
        with pytest.raises(ValueError, match="source_uri cannot be empty"):
            generate_doc_id("")


class TestGenerateQuoteId:
    """Test quote ID generation."""

    def test_generate_quote_id_basic(self):
        """Test basic quote ID generation."""
        doc_id = "doc:abc123"
        span_start, span_end = 100, 150
        text = "This is a quote"
        quote_id = generate_quote_id(doc_id, span_start, span_end, text)
        assert quote_id.startswith("quote:")
        assert validate_id_format(quote_id, "quote:")
        assert extract_namespace(quote_id) == "quote"

    def test_generate_quote_id_deterministic(self):
        """Test that same inputs produce same quote ID."""
        doc_id = "doc:abc123"
        span_start, span_end = 100, 150
        text = "This is a quote"
        quote_id1 = generate_quote_id(doc_id, span_start, span_end, text)
        quote_id2 = generate_quote_id(doc_id, span_start, span_end, text)
        assert quote_id1 == quote_id2

    def test_generate_quote_id_different_spans(self):
        """Test that different spans produce different IDs."""
        doc_id = "doc:abc123"
        text = "This is a quote"
        quote_id1 = generate_quote_id(doc_id, 100, 150, text)
        quote_id2 = generate_quote_id(doc_id, 200, 250, text)
        assert quote_id1 != quote_id2

    def test_generate_quote_id_empty_doc_id(self):
        """Test that empty doc_id raises error."""
        with pytest.raises(ValueError, match="doc_id cannot be empty"):
            generate_quote_id("", 100, 150, "text")

    def test_generate_quote_id_invalid_span(self):
        """Test that invalid spans raise errors."""
        doc_id = "doc:abc123"
        text = "This is a quote"

        # Negative start
        with pytest.raises(ValueError, match="Invalid span coordinates"):
            generate_quote_id(doc_id, -1, 150, text)

        # End before start
        with pytest.raises(ValueError, match="Invalid span coordinates"):
            generate_quote_id(doc_id, 150, 100, text)

    def test_generate_quote_id_empty_text(self):
        """Test that empty text raises error."""
        doc_id = "doc:abc123"
        with pytest.raises(ValueError, match="text cannot be empty"):
            generate_quote_id(doc_id, 100, 150, "")


class TestGenerateCaseId:
    """Test case ID generation."""

    def test_generate_case_id_basic(self):
        """Test basic case ID generation."""
        court = "scotus"
        docket = "123-456"
        case_id = generate_case_id(court, docket)
        assert case_id.startswith("case:")
        assert validate_id_format(case_id, "case:")
        assert extract_namespace(case_id) == "case"

    def test_generate_case_id_with_year(self):
        """Test case ID generation with year."""
        court = "scotus"
        docket = "123-456"
        year = 2024
        case_id = generate_case_id(court, docket, year)
        assert case_id.startswith("case:")

    def test_generate_case_id_deterministic(self):
        """Test that same inputs produce same case ID."""
        court = "scotus"
        docket = "123-456"
        case_id1 = generate_case_id(court, docket)
        case_id2 = generate_case_id(court, docket)
        assert case_id1 == case_id2

    def test_generate_case_id_different_courts(self):
        """Test that different courts produce different IDs."""
        docket = "123-456"
        case_id1 = generate_case_id("scotus", docket)
        case_id2 = generate_case_id("ca1", docket)
        assert case_id1 != case_id2

    def test_generate_case_id_empty_court(self):
        """Test that empty court raises error."""
        with pytest.raises(ValueError, match="court cannot be empty"):
            generate_case_id("", "123-456")

    def test_generate_case_id_empty_docket(self):
        """Test that empty docket raises error."""
        with pytest.raises(ValueError, match="docket_number cannot be empty"):
            generate_case_id("scotus", "")


class TestValidateIdFormat:
    """Test ID format validation."""

    def test_validate_doc_id_format(self):
        """Test doc ID format validation."""
        from corp_speech_risk_dataset.types.ids.generate import generate_doc_id

        valid_id = generate_doc_id("https://example.com/test.pdf")
        assert validate_id_format(valid_id, "doc:")

        invalid_id = "quote:abc123def456"
        assert not validate_id_format(invalid_id, "doc:")

        # Test invalid format
        assert not validate_id_format("", "doc:")
        assert not validate_id_format("invalid", "doc:")
        assert not validate_id_format("doc:", "doc:")  # Empty hash part


class TestExtractNamespace:
    """Test namespace extraction."""

    def test_extract_namespace_valid(self):
        """Test extracting valid namespaces."""
        assert extract_namespace("doc:abc123") == "doc"
        assert extract_namespace("quote:abc123") == "quote"
        assert extract_namespace("case:abc123") == "case"

    def test_extract_namespace_invalid(self):
        """Test extracting invalid namespaces."""
        assert extract_namespace("invalid:abc123") is None
        assert extract_namespace("no_colon") is None
        assert extract_namespace("") is None
