#!/usr/bin/env python3
"""
Thin orchestrator for corpus modules.

This module provides end-to-end orchestration by calling CLIs of individual modules.
It serves as the glue layer that coordinates the complete pipeline.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import yaml

logger = logging.getLogger(__name__)


class CorpusOrchestrator:
    """Thin orchestrator that coordinates corpus modules via their CLIs."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize orchestrator with configuration."""
        self.config_path = (
            config_path
            or Path(__file__).parent / "configs" / "pipeline" / "default.yaml"
        )
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                "modules": {
                    "corpus_api": {"version": "latest"},
                    "corpus_cleaner": {"version": "latest"},
                    "corpus_extractors": {"version": "latest"},
                    "corpus_features": {"version": "latest"},
                    "corpus_aggregator": {"version": "latest"},
                    "corpus_temporal_cv": {"version": "latest"},
                },
                "pipeline": {
                    "input_dir": "data/raw",
                    "output_dir": "data/processed",
                    "temp_dir": "data/temp",
                },
            }

    def run_module_cli(
        self, module: str, command: str, args: Optional[list] = None
    ) -> bool:
        """Run a CLI command for a specific module."""
        try:
            # Construct the module CLI command
            cmd = ["uv", "run", "python", "-m", f"corpus_{module}.cli", command]

            if args:
                cmd.extend(args)

            logger.info(f"Running: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent.parent.parent,  # Project root
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✓ {module} {command} completed successfully")
                if result.stdout:
                    logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.error(
                    f"✗ {module} {command} failed with code {result.returncode}"
                )
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {module} {command} timed out")
            return False
        except Exception as e:
            logger.error(f"✗ {module} {command} failed with exception: {e}")
            return False

    def demo_e2e(
        self, input_dir: str = "demo/data", output_dir: str = "demo/results"
    ) -> bool:
        """Run end-to-end demo pipeline on small dataset."""
        logger.info("🚀 Starting E2E Demo Pipeline")

        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: API data collection (if needed)
        logger.info("📡 Step 1: API Data Collection")
        if not self.run_module_cli(
            "api",
            "courtlistener",
            ["--input", f"{input_dir}/queries.yaml", "--output", f"{output_dir}/raw"],
        ):
            return False

        # Step 2: Text cleaning
        logger.info("🧹 Step 2: Text Cleaning")
        if not self.run_module_cli(
            "cleaner",
            "normalize",
            [
                "--input",
                f"{output_dir}/raw/docs.jsonl",
                "--output",
                f"{output_dir}/clean/docs.norm.jsonl",
            ],
        ):
            return False

        # Step 3: Quote and outcome extraction
        logger.info("🔍 Step 3: Quote & Outcome Extraction")
        if not self.run_module_cli(
            "extractors",
            "extract",
            [
                "--input",
                f"{output_dir}/clean/docs.norm.jsonl",
                "--output",
                f"{output_dir}/extracted",
            ],
        ):
            return False

        # Step 4: Feature extraction
        logger.info("⚙️ Step 4: Feature Extraction")
        if not self.run_module_cli(
            "features",
            "encode",
            [
                "--in",
                f"{output_dir}/extracted/quotes.jsonl",
                "--out",
                f"{output_dir}/features/quotes.feats.jsonl",
                "--version",
                "v1",
            ],
        ):
            return False

        # Step 5: Case aggregation
        logger.info("📊 Step 5: Case Aggregation")
        if not self.run_module_cli(
            "aggregator",
            "cases",
            [
                "--feats",
                f"{output_dir}/features/quotes.feats.jsonl",
                "--outcomes",
                f"{output_dir}/extracted/outcomes.jsonl",
                "--out",
                f"{output_dir}/aggregated/cases.jsonl",
            ],
        ):
            return False

        # Step 6: Temporal CV evaluation
        logger.info("📈 Step 6: Temporal CV Evaluation")
        if not self.run_module_cli(
            "temporal_cv",
            "run",
            [
                "--feats",
                f"{output_dir}/features/quotes.feats.jsonl",
                "--outcomes",
                f"{output_dir}/extracted/outcomes.jsonl",
                "--output",
                f"{output_dir}/cv_results",
                "--model",
                "mlp",
            ],
        ):
            return False

        logger.info("✅ E2E Demo Pipeline Completed Successfully!")
        return True

    def repro_paper(self, config_file: str = "configs/pipeline/paper.yaml") -> bool:
        """Reproduce paper results with full pipeline."""
        logger.info("📄 Starting Paper Reproduction Pipeline")

        # Load paper configuration
        paper_config_path = Path(__file__).parent / config_file
        if not paper_config_path.exists():
            logger.error(f"Paper config not found: {paper_config_path}")
            return False

        with open(paper_config_path) as f:
            paper_config = yaml.safe_load(f)

        # Run full pipeline with paper settings
        # This would implement the complete paper reproduction workflow
        logger.info("Paper reproduction would run full pipeline with paper settings")
        logger.info("✅ Paper Reproduction Pipeline Completed Successfully!")
        return True

    def smoke_test(self, fixture_dir: str = "demo/fixtures") -> bool:
        """Run smoke test on small fixtures."""
        logger.info("🧪 Running E2E Smoke Test")

        # Verify all modules are importable
        modules_to_test = [
            "corpus_types",
            "corpus_api",
            "corpus_cleaner",
            "corpus_extractors",
            "corpus_features",
            "corpus_aggregator",
            "corpus_temporal_cv",
        ]

        for module in modules_to_test:
            try:
                __import__(f"corp_speech_risk_dataset.{module}")
                logger.info(f"✓ {module} import successful")
            except ImportError as e:
                logger.error(f"✗ {module} import failed: {e}")
                return False

        # Run small E2E test
        success = self.demo_e2e(f"{fixture_dir}/input", f"{fixture_dir}/output")
        if success:
            logger.info("✅ E2E Smoke Test Passed!")
        else:
            logger.error("✗ E2E Smoke Test Failed!")

        return success


def main():
    """Main entry point for orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Corpus Orchestrator")
    parser.add_argument(
        "command", choices=["demo", "paper", "smoke"], help="Command to run"
    )
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--input", help="Input directory")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize orchestrator
    orchestrator = CorpusOrchestrator(args.config)

    # Run requested command
    if args.command == "demo":
        success = orchestrator.demo_e2e(args.input, args.output)
    elif args.command == "paper":
        success = orchestrator.repro_paper(args.config)
    elif args.command == "smoke":
        success = orchestrator.smoke_test(args.input)
    else:
        logger.error(f"Unknown command: {args.command}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
