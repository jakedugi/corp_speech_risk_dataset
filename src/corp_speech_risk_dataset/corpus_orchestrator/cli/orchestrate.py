#!/usr/bin/env python3
"""
CLI interface for corpus-orchestrator module.

Usage:
    python -m corpus_orchestrator.cli.orchestrate demo --input demo/data --output demo/results
    python -m corpus_orchestrator.cli.orchestrate paper --config configs/pipeline/paper.yaml
    python -m corpus_orchestrator.cli.orchestrate smoke --input demo/fixtures
"""

import json
from pathlib import Path
from typing import Optional
import typer
import logging

from ..orchestrator import Orchestrator

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def demo(
    input_dir: str = typer.Option("demo/data", "--input", help="Input data directory"),
    output_dir: str = typer.Option(
        "demo/results", "--output", help="Output results directory"
    ),
):
    """
    Run end-to-end demo pipeline on small dataset.

    This command runs the complete corpus pipeline from raw documents
    to final case predictions using demo fixtures.
    """
    logger.info(f"Running demo pipeline: {input_dir} -> {output_dir}")

    orchestrator = Orchestrator()
    success = orchestrator.demo_e2e(input_dir=input_dir, output_dir=output_dir)

    if success:
        logger.info("✅ Demo pipeline completed successfully!")
        typer.echo("Demo pipeline completed successfully!")
    else:
        logger.error("❌ Demo pipeline failed!")
        typer.echo("Demo pipeline failed!", err=True)
        raise typer.Exit(1)


@app.command()
def paper(
    config_file: Path = typer.Option(
        ..., "--config", help="Pipeline configuration YAML"
    ),
    input_dir: Optional[str] = typer.Option(
        None, "--input", help="Input data directory"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", help="Output results directory"
    ),
):
    """
    Run full paper reproduction pipeline.

    This command runs the complete pipeline with parametrization
    as used in the research paper.
    """
    logger.info(f"Running paper pipeline with config: {config_file}")

    # Load configuration
    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)

    orchestrator = Orchestrator(config_path=config_file)
    success = orchestrator.repro_paper()

    if success:
        logger.info("✅ Paper pipeline completed successfully!")
        typer.echo("Paper pipeline completed successfully!")
    else:
        logger.error("❌ Paper pipeline failed!")
        typer.echo("Paper pipeline failed!", err=True)
        raise typer.Exit(1)


@app.command()
def smoke(
    input_dir: str = typer.Option(
        "demo/fixtures", "--input", help="Input fixtures directory"
    ),
):
    """
    Run smoke test across all modules.

    This command tests that all modules can be imported and their
    basic functionality works correctly.
    """
    logger.info(f"Running smoke test with fixtures: {input_dir}")

    orchestrator = Orchestrator()
    success = orchestrator.smoke_test(input_dir=input_dir)

    if success:
        logger.info("✅ Smoke test passed!")
        typer.echo("Smoke test passed!")
    else:
        logger.error("❌ Smoke test failed!")
        typer.echo("Smoke test failed!", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    config_file: Path = typer.Option(
        ..., "--config", help="Configuration file to validate"
    ),
):
    """
    Validate pipeline configuration file.

    This command checks that the configuration file is valid
    and all required parameters are present.
    """
    logger.info(f"Validating configuration: {config_file}")

    # Load and validate configuration
    import yaml

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Basic validation
        required_keys = ["pipeline", "modules"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")

        logger.info("✅ Configuration is valid!")
        typer.echo("Configuration is valid!")

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        typer.echo(f"Configuration validation failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
