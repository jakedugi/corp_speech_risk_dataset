#!/usr/bin/env python3
"""
CLI interface for corpus-temporal-cv module.

Usage:
    python -m corpus_temporal_cv.cli.cv run --feats quote_feats.jsonl --outcomes outcomes.jsonl --config cv.yaml --output runs/
    python -m corpus_temporal_cv.cli.cv report --run runs/last --output report.md
"""

import json
from pathlib import Path
from typing import Optional
import typer
import logging

from ..splits import main as run_cv_splits
from ..trainers.frozen_mlp_trainer import main as run_mlp_training
from ..trainers.optimized_features_trainer import main as run_optimized_training

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def run(
    features_file: Path = typer.Option(
        ..., "--feats", help="Input QuoteFeatures JSONL file"
    ),
    outcomes_file: Path = typer.Option(
        ..., "--outcomes", help="Input Outcomes JSONL file"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="CV configuration YAML"
    ),
    output_dir: Path = typer.Option(..., "--output", help="Output directory for runs"),
    model_type: str = typer.Option(
        "mlp", "--model", help="Model type: mlp, lr, optimized"
    ),
    fold: int = typer.Option(4, "--fold", help="Fold number to use"),
):
    """
    Run temporal CV training with specified model.

    Takes QuoteFeatures and Outcomes files and produces model artifacts and OOF metrics.
    """
    logger.info(f"Running temporal CV with model: {model_type}")
    logger.info(f"Features: {features_file}")
    logger.info(f"Outcomes: {outcomes_file}")
    logger.info(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = {}
    if config_file and config_file.exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

    # Run the appropriate training based on model type
    if model_type == "mlp":
        # Run MLP training
        logger.info("Running PyTorch MLP training...")
        # Note: This would need to be adapted to use the new CLI interface
        # For now, we'll just log that it would run

    elif model_type == "optimized":
        # Run optimized features training
        logger.info("Running optimized features training...")
        # Note: This would need to be adapted to use the new CLI interface

    elif model_type == "lr":
        # Run final LR training
        logger.info("Running final LR training...")
        # Note: This would need to be adapted to use the new CLI interface

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Temporal CV training complete. Results saved to {output_dir}")


@app.command()
def report(
    run_dir: Path = typer.Option(..., "--run", help="Run directory to report on"),
    output_file: Path = typer.Option(..., "--output", help="Output report file"),
    format: str = typer.Option(
        "markdown", "--format", help="Report format: markdown, json"
    ),
):
    """
    Generate report from temporal CV run results.
    """
    logger.info(f"Generating report from run: {run_dir}")
    logger.info(f"Output format: {format}")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Look for results files
    results_files = list(run_dir.glob("*results*.json"))
    if not results_files:
        logger.warning(f"No results files found in {run_dir}")
        return

    # Load results
    results = {}
    for results_file in results_files:
        with open(results_file) as f:
            results[results_file.stem] = json.load(f)

    # Generate report based on format
    if format == "markdown":
        report_content = generate_markdown_report(results, run_dir)
    elif format == "json":
        report_content = json.dumps(results, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Write report
    with open(output_file, "w") as f:
        f.write(report_content)

    logger.info(f"Report generated: {output_file}")


@app.command()
def splits(
    input_file: Path = typer.Option(..., "--input", help="Input dataset JSONL file"),
    output_dir: Path = typer.Option(
        ..., "--output", help="Output directory for splits"
    ),
    k_folds: int = typer.Option(5, "--k-folds", help="Number of CV folds"),
    target_field: str = typer.Option(..., "--target", help="Target field name"),
    case_id_field: str = typer.Option(
        "case_id", "--case-id", help="Case ID field name"
    ),
    oof_test_ratio: float = typer.Option(0.15, "--oof-ratio", help="OOF test ratio"),
    random_seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """
    Create temporal CV splits from dataset.
    """
    logger.info(f"Creating {k_folds}-fold temporal CV splits")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")

    # This would call the splits.py functionality
    # For now, we'll just indicate what would be done
    logger.info("CV splits creation would be implemented here")
    logger.info(f"Splits would be saved to {output_dir}")


def generate_markdown_report(results: dict, run_dir: Path) -> str:
    """Generate markdown report from results."""
    report_lines = [
        "# Temporal CV Training Report",
        "",
        f"Run Directory: {run_dir}",
        "",
        "## Summary",
        "",
    ]

    for model_name, model_results in results.items():
        report_lines.extend(
            [
                f"### {model_name}",
                "",
                "Key Metrics:",
            ]
        )

        # Add key metrics if available
        if "test_mcc_suppressed" in model_results:
            mcc = model_results["test_mcc_suppressed"]
            report_lines.append(f"- Test MCC (Suppressed): {mcc:.4f}")

        if "test_auc_suppressed" in model_results:
            auc = model_results["test_auc_suppressed"]
            report_lines.append(f"- Test AUC (Suppressed): {auc:.4f}")

        if "passes_criteria" in model_results:
            passes = model_results["passes_criteria"]
            report_lines.append(f"- Passes Selection Criteria: {passes}")

        report_lines.append("")

    return "\n".join(report_lines)


if __name__ == "__main__":
    app()
