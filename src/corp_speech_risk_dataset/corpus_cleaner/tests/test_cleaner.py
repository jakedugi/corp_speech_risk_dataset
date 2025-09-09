"""Tests for the TextCleaner functionality."""

import pytest
from corp_speech_risk_dataset.cleaner import TextCleaner


class TestTextCleaner:
    """Test cases for TextCleaner text normalization."""

    @pytest.fixture
    def cleaner(self):
        """Create a TextCleaner instance for testing."""
        return TextCleaner()

    def test_clean_basic_text(self, cleaner):
        """Test basic text cleaning functionality."""
        input_text = "Hello   world\n\nwith  extra    spaces."
        expected = "Hello world\n\nwith extra spaces."
        result = cleaner.clean(input_text)
        assert result == expected

    def test_clean_unicode_normalization(self, cleaner):
        """Test that unicode characters are properly normalized."""
        input_text = "café résumé naïve"
        result = cleaner.clean(input_text)
        # TextCleaner should preserve unicode but normalize whitespace
        assert "café" in result
        assert "résumé" in result
        assert "naïve" in result

    def test_clean_empty_text(self, cleaner):
        """Test cleaning empty or whitespace-only text."""
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""
        assert cleaner.clean("\n\t  \n") == ""

    def test_clean_multiple_spaces(self, cleaner):
        """Test that multiple spaces are collapsed to single spaces."""
        input_text = "word1    word2  word3"
        expected = "word1 word2 word3"
        result = cleaner.clean(input_text)
        assert result == expected

    def test_clean_newlines_preserved(self, cleaner):
        """Test that single newlines are preserved."""
        input_text = "line1\nline2\n\nline3"
        result = cleaner.clean(input_text)
        assert "\n" in result
        assert "\n\n" in result

    def test_clean_tabs_converted(self, cleaner):
        """Test that tabs are converted to spaces."""
        input_text = "word1\t\tword2"
        result = cleaner.clean(input_text)
        assert "\t" not in result
        assert "word1 word2" in result

    def test_clean_quotes_preserved(self, cleaner):
        """Test that quotes are preserved in text cleaning."""
        input_text = '"Hello world" said the person.'
        result = cleaner.clean(input_text)
        assert '"Hello world"' in result
        assert "said the person" in result

    def test_clean_case_preserved(self, cleaner):
        """Test that case is preserved in text cleaning."""
        input_text = "Hello World TEST"
        result = cleaner.clean(input_text)
        assert result == input_text  # Should be unchanged

    def test_clean_numbers_preserved(self, cleaner):
        """Test that numbers are preserved in text cleaning."""
        input_text = "The year is 2023 and the price is $10.99"
        result = cleaner.clean(input_text)
        assert "2023" in result
        assert "$10.99" in result
