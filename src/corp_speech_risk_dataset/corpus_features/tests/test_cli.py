"""Tests for corpus-features CLI."""

import json
import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ..cli.encode import app


class TestCLI:
    """Test the CLI interface."""

    @pytest.fixture
    def runner(self):
        """CLI runner fixture."""
        return CliRunner()

    @pytest.fixture
    def sample_quotes_file(self):
        """Create a sample quotes JSONL file."""
        quotes_data = [
            {
                "quote_id": "q_1",
                "doc_id": "doc_1",
                "text": "This is a test quote about legal matters.",
                "speaker": "Attorney",
            },
            {
                "quote_id": "q_2",
                "doc_id": "doc_2",
                "text": "Another quote discussing corporate policy.",
                "speaker": "CEO",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for quote in quotes_data:
                f.write(json.dumps(quote) + "\n")
            return Path(f.name)

    def test_encode_command(self, runner, sample_quotes_file):
        """Test the encode command."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            result = runner.invoke(
                app,
                [
                    "encode",
                    "--in",
                    str(sample_quotes_file),
                    "--out",
                    str(output_path),
                    "--version",
                    "v1",
                ],
            )

            assert result.exit_code == 0
            assert "Feature encoding complete" in result.stdout

            # Check output file was created and has content
            assert output_path.exists()
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 2  # Should have 2 feature records

                # Check first feature record
                feature_data = json.loads(lines[0])
                assert "quote_id" in feature_data
                assert "feature_version" in feature_data
                assert feature_data["feature_version"] == "v1"

        finally:
            output_path.unlink(missing_ok=True)
            sample_quotes_file.unlink(missing_ok=True)

    def test_validate_command_valid(self, runner):
        """Test validation of valid features."""
        # Create a valid features file
        features_data = [
            {
                "quote_id": "q_1",
                "feature_version": "v1",
                "vector": [0.1, 0.2, 0.3],
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for feature in features_data:
                f.write(json.dumps(feature) + "\n")
            features_file = Path(f.name)

        try:
            result = runner.invoke(
                app, ["validate", "--in", str(features_file), "--version", "v1"]
            )

            assert result.exit_code == 0
            assert "Feature validation passed" in result.stdout

        finally:
            features_file.unlink(missing_ok=True)

    def test_validate_command_invalid_version(self, runner):
        """Test validation fails with wrong version."""
        # Create features file with wrong version
        features_data = [
            {
                "quote_id": "q_1",
                "feature_version": "v2",  # Wrong version
                "vector": [0.1, 0.2, 0.3],
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for feature in features_data:
                f.write(json.dumps(feature) + "\n")
            features_file = Path(f.name)

        try:
            result = runner.invoke(
                app,
                [
                    "validate",
                    "--in",
                    str(features_file),
                    "--version",
                    "v1",  # Expecting v1
                ],
            )

            assert result.exit_code == 1  # Should fail
            assert "Feature version mismatch" in result.stdout

        finally:
            features_file.unlink(missing_ok=True)
