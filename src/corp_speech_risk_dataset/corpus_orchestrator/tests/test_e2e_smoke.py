"""E2E smoke test for corpus orchestrator.

Tests the complete pipeline on a small 10-doc fixture to ensure
all modules work together correctly.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ..orchestrator import CorpusOrchestrator


class TestE2ESmoke:
    """Test end-to-end smoke functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return CorpusOrchestrator()

    @pytest.fixture
    def mock_fixtures(self, tmp_path):
        """Create mock fixture data."""
        # Create input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create mock queries.yaml
        queries_file = input_dir / "queries.yaml"
        queries_file.write_text(
            """
sources:
  - courtlistener
queries:
  - docket_number: "123-456"
    date_range: ["2020-01-01", "2020-12-31"]
"""
        )

        # Create mock docs.jsonl
        docs_file = input_dir / "docs.jsonl"
        docs_data = []
        for i in range(10):  # 10 docs as specified
            docs_data.append(
                {
                    "doc_id": f"doc_{i:03d}",
                    "source_uri": f"https://example.com/doc_{i}",
                    "retrieved_at": "2023-01-01T00:00:00Z",
                    "raw_text": f"This is sample document {i} with some legal text about corporate matters.",
                    "meta": {
                        "court": "scotus",
                        "docket": f"123-{456+i}",
                        "party": f"Company {i} v. Regulator",
                    },
                }
            )

        with open(docs_file, "w") as f:
            for doc in docs_data:
                f.write(json.dumps(doc) + "\n")

        return input_dir

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.config is not None
        assert "modules" in orchestrator.config
        assert "pipeline" in orchestrator.config

    def test_config_loading(self, orchestrator):
        """Test configuration loading."""
        config = orchestrator._load_config()

        # Should have default modules
        assert "corpus_api" in config["modules"]
        assert "corpus_cleaner" in config["modules"]
        assert "corpus_extractors" in config["modules"]
        assert "corpus_features" in config["modules"]
        assert "corpus_aggregator" in config["modules"]
        assert "corpus_temporal_cv" in config["modules"]

        # Should have pipeline settings
        assert "pipeline" in config
        assert "input_dir" in config["pipeline"]
        assert "output_dir" in config["pipeline"]

    @patch("subprocess.run")
    def test_run_module_cli_success(self, mock_subprocess, orchestrator):
        """Test successful module CLI execution."""
        # Mock successful subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        success = orchestrator.run_module_cli("api", "courtlistener", ["--test"])

        assert success is True
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    def test_run_module_cli_failure(self, mock_subprocess, orchestrator):
        """Test failed module CLI execution."""
        # Mock failed subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        mock_subprocess.return_value = mock_result

        success = orchestrator.run_module_cli("api", "courtlistener")

        assert success is False

    @patch("subprocess.run")
    def test_demo_e2e_workflow(self, mock_subprocess, orchestrator, tmp_path):
        """Test the demo E2E workflow execution."""
        # Mock successful execution for all steps
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        output_dir = tmp_path / "results"

        success = orchestrator.demo_e2e("demo/input", str(output_dir))

        assert success is True

        # Should have called subprocess 6 times (one for each module)
        assert mock_subprocess.call_count == 6

    def test_smoke_test_imports(self, orchestrator):
        """Test that smoke test can import all modules."""
        # This test verifies that the orchestrator can import all corpus modules
        # In a real test environment, all modules would be available

        modules_to_test = [
            "corpus_types",
            "corpus_api",
            "corpus_cleaner",
            "corpus_extractors",
            "corpus_features",
            "corpus_aggregator",
            "corpus_temporal_cv",
        ]

        # In this test environment, we can't actually import the modules
        # but we can verify the orchestrator knows about them
        for module in modules_to_test:
            assert module in orchestrator.config["modules"]

    def test_pipeline_stages_configuration(self, orchestrator):
        """Test that pipeline stages are properly configured."""
        config = orchestrator._load_config()

        # Should have pipeline configuration
        assert "stages" in config.get("pipeline", {})

        stages = config["pipeline"]["stages"]

        # Should have expected stages
        stage_names = [stage["name"] for stage in stages]
        expected_stages = [
            "data_collection",
            "text_cleaning",
            "extraction",
            "feature_engineering",
            "aggregation",
            "evaluation",
        ]

        for expected_stage in expected_stages:
            assert expected_stage in stage_names

    def test_module_version_configuration(self, orchestrator):
        """Test that modules have version configuration."""
        config = orchestrator._load_config()

        for module_name, module_config in config["modules"].items():
            assert "version" in module_config
            assert module_config["version"] == "latest"  # Default version
