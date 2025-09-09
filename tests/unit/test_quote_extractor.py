"""Tests for the quote extractor."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from corp_speech_risk_dataset.extractors.quote_extractor import QuoteExtractor


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    The court held that "the defendant's actions violated Section 5 of the FTC Act."
    However, the plaintiff argued that "the conduct was not deceptive."
    """


@pytest.fixture
def extractor():
    """Create a test extractor."""
    return QuoteExtractor()


def test_extract_quotes(extractor, sample_text):
    """Test quote extraction."""
    # TODO: Implement once quote extraction is implemented
    quotes = extractor.extract_quotes(sample_text)
    assert isinstance(quotes, list)


def test_process_file(tmp_path, extractor):
    """Test file processing."""
    # Create test input file
    input_file = tmp_path / "test.json"
    input_data = [{"id": 1, "opinion_text": "Test opinion with quotes."}]
    input_file.write_text(json.dumps(input_data))

    # Process file
    output_file = tmp_path / "output.json"
    extractor.process_file(input_file, output_file)

    # Check output
    assert output_file.exists()
    output_data = json.loads(output_file.read_text())
    assert isinstance(output_data, list)


def test_process_directory(tmp_path, extractor):
    """Test directory processing."""
    # Create test directory with files
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    for i in range(3):
        file = input_dir / f"test_{i}.json"
        file.write_text(json.dumps([{"id": i, "opinion_text": "Test"}]))

    # Process directory
    output_dir = tmp_path / "output"
    extractor.process_directory(input_dir, output_dir)

    # Check output
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.json"))) == 3
