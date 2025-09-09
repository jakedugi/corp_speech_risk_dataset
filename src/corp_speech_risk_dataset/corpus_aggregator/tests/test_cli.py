"""Tests for corpus-aggregator CLI."""

import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ..cli.aggregate import app


class TestCLI:
    """Test the CLI interface."""

    @pytest.fixture
    def runner(self):
        """CLI runner fixture."""
        return CliRunner()

    @pytest.fixture
    def sample_quote_features_file(self):
        """Create a sample quote features JSONL file."""
        quote_data = [
            {
                "case_id": "case_001",
                "doc_id": "doc_001",
                "text": "Sample quote text",
                "docket_number": 1,
                "global_token_start": 0,
                "num_tokens": 50,
                "mlp_probability": 0.7,
                "mlp_pred_strict": 1,
                "mlp_pred_recallT": 1,
                "outcome_bin": 0,
            },
            {
                "case_id": "case_001",
                "doc_id": "doc_002",
                "text": "Another quote text",
                "docket_number": 2,
                "global_token_start": 100,
                "num_tokens": 40,
                "mlp_probability": 0.3,
                "mlp_pred_strict": 0,
                "mlp_pred_recallT": 0,
                "outcome_bin": 0,
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            import json

            for quote in quote_data:
                f.write(json.dumps(quote) + "\n")
            return Path(f.name)

    @pytest.fixture
    def sample_outcomes_file(self):
        """Create a sample outcomes JSONL file."""
        outcome_data = [
            {
                "case_id": "case_001",
                "outcome_bucket": 0,
                "outcome_label": "low",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            import json

            for outcome in outcome_data:
                f.write(json.dumps(outcome) + "\n")
            return Path(f.name)

    def test_cases_command_basic(
        self, runner, sample_quote_features_file, sample_outcomes_file
    ):
        """Test the cases command with basic functionality."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            result = runner.invoke(
                app,
                [
                    "cases",
                    "--feats",
                    str(sample_quote_features_file),
                    "--outcomes",
                    str(sample_outcomes_file),
                    "--out",
                    str(output_path),
                    "--thresholds",
                    "token_2500",
                ],
            )

            # The command should execute (even if it doesn't do full processing)
            assert result.exit_code == 0
            assert "Aggregating cases" in result.stdout

        finally:
            output_path.unlink(missing_ok=True)
            sample_quote_features_file.unlink(missing_ok=True)
            sample_outcomes_file.unlink(missing_ok=True)

    def test_predict_command_basic(self, runner):
        """Test the predict command with basic functionality."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as cases_file:
            cases_path = Path(cases_file.name)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            result = runner.invoke(
                app,
                [
                    "predict",
                    "--cases",
                    str(cases_path),
                    "--out",
                    str(output_path),
                    "--model",
                    "lr",
                ],
            )

            # The command should execute
            assert result.exit_code == 0
            assert "Making predictions" in result.stdout

        finally:
            cases_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_evaluate_command_basic(self, runner):
        """Test the evaluate command with basic functionality."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as preds_file:
            preds_path = Path(preds_file.name)

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False
        ) as outcomes_file:
            outcomes_path = Path(outcomes_file.name)

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--preds",
                    str(preds_path),
                    "--outcomes",
                    str(outcomes_path),
                    "--out",
                    str(output_path),
                ],
            )

            # The command should execute
            assert result.exit_code == 0
            assert "Evaluating predictions" in result.stdout

    def test_minimal_command_basic(self, runner):
        """Test the minimal command."""
        with tempfile.TemporaryDirectory() as mirror_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                mirror_path = Path(mirror_dir)
                output_path = Path(output_dir)

                result = runner.invoke(
                    app,
                    [
                        "minimal",
                        "--mirror",
                        str(mirror_path),
                        "--out",
                        str(output_path),
                    ],
                )

                # The command should execute
                assert result.exit_code == 0
                assert "minimal prediction workflow" in result.stdout

    def test_infra_command_basic(self, runner):
        """Test the infra command."""
        with tempfile.TemporaryDirectory() as mirror_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                mirror_path = Path(mirror_dir)
                output_path = Path(output_dir)

                result = runner.invoke(
                    app,
                    [
                        "infra",
                        "--mirror",
                        str(mirror_path),
                        "--out",
                        str(output_path),
                    ],
                )

                # The command should execute
                assert result.exit_code == 0
                assert "prediction with existing infrastructure" in result.stdout
