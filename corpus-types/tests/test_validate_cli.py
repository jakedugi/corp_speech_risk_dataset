"""
Tests for corpus-types validation CLI.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

import sys
from pathlib import Path

# Add the main project src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from corp_speech_risk_dataset.types.cli.validate import app


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


class TestValidateJsonl:
    """Test JSONL validation command."""

    def test_validate_docs_fixture(self, runner, fixtures_dir):
        """Test validating docs fixture."""
        docs_file = fixtures_dir / "docs.small.jsonl"
        result = runner.invoke(app, ["jsonl", "Doc", str(docs_file)])
        assert result.exit_code == 0
        assert "✅ Validation successful" in result.output
        assert "1 records validated" in result.output

    def test_validate_quotes_fixture(self, runner, fixtures_dir):
        """Test validating quotes fixture."""
        quotes_file = fixtures_dir / "quotes.small.jsonl"
        result = runner.invoke(app, ["jsonl", "Quote", str(quotes_file)])
        assert result.exit_code == 0
        assert "✅ Validation successful" in result.output
        assert "1 records validated" in result.output

    def test_validate_outcomes_fixture(self, runner, fixtures_dir):
        """Test validating outcomes fixture."""
        outcomes_file = fixtures_dir / "outcomes.small.jsonl"
        result = runner.invoke(app, ["jsonl", "Outcome", str(outcomes_file)])
        assert result.exit_code == 0
        assert "✅ Validation successful" in result.output
        assert "1 records validated" in result.output

    def test_validate_invalid_schema(self, runner, fixtures_dir):
        """Test validation with invalid schema name."""
        docs_file = fixtures_dir / "docs.small.jsonl"
        result = runner.invoke(app, ["jsonl", "InvalidSchema", str(docs_file)])
        assert result.exit_code == 2
        assert "❌ Unknown schema" in result.output

    def test_validate_missing_file(self, runner):
        """Test validation with missing file."""
        result = runner.invoke(app, ["jsonl", "Doc", "nonexistent.jsonl"])
        assert result.exit_code == 1
        assert "❌ File not found" in result.output

    def test_validate_with_limit(self, runner, fixtures_dir):
        """Test validation with record limit."""
        docs_file = fixtures_dir / "docs.small.jsonl"
        result = runner.invoke(app, ["jsonl", "Doc", str(docs_file), "--limit", "1"])
        assert result.exit_code == 0
        assert "✅ Validation successful" in result.output


class TestGenerateSchemas:
    """Test schema generation command."""

    def test_generate_schemas(self, runner, tmp_path):
        """Test schema generation."""
        result = runner.invoke(app, ["generate-schemas", str(tmp_path)])
        assert result.exit_code == 0
        assert "✅ Schema generation complete" in result.output

        # Check that some schema files were created
        schema_files = list(tmp_path.glob("*.schema.json"))
        assert len(schema_files) > 0

        # Check that expected schemas exist
        expected_schemas = [
            "doc.schema.json",
            "quote.schema.json",
            "outcome.schema.json",
        ]
        for schema_file in expected_schemas:
            assert (tmp_path / schema_file).exists()


class TestListModels:
    """Test list-models command."""

    def test_list_models(self, runner):
        """Test listing available models."""
        result = runner.invoke(app, ["list-models"])
        assert result.exit_code == 0
        assert "Available corpus-types models:" in result.output
        assert "Doc" in result.output
        assert "Quote" in result.output
        assert "Outcome" in result.output


class TestCliHelp:
    """Test CLI help and basic functionality."""

    def test_cli_help(self, runner):
        """Test main CLI help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "corpus-types" in result.output

    def test_validate_help(self, runner):
        """Test validate command help."""
        result = runner.invoke(app, ["jsonl", "--help"])
        assert result.exit_code == 0
        assert "Validate JSONL file" in result.output

    def test_no_args(self, runner):
        """Test running CLI with no arguments."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0  # Should show help
        assert "corpus-types" in result.output


class TestQuietMode:
    """Test quiet mode functionality."""

    def test_quiet_mode(self, runner, fixtures_dir):
        """Test quiet mode suppresses success messages."""
        docs_file = fixtures_dir / "docs.small.jsonl"
        result = runner.invoke(app, ["jsonl", "Doc", str(docs_file), "--quiet"])
        assert result.exit_code == 0
        # In quiet mode, success messages should be suppressed
        assert "✅ Validation successful" not in result.output
