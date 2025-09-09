#!/usr/bin/env python3
"""
CLI interface for corpus-aggregator module.

Usage:
    python -m corpus_aggregator.cli.aggregate cases --feats quote_feats.jsonl --outcomes outcomes.jsonl --out case_vectors.jsonl --config agg.yaml
    python -m corpus_aggregator.cli.aggregate predict --cases case_vectors.jsonl --out predictions.jsonl --model lr
"""

import json
from pathlib import Path
from typing import Optional
import typer
import logging

from ..modeling.dataset import (
    ThresholdSpec,
    FeatureConfig,
    DEFAULT_THRESHOLDS,
)
from ..modeling.models import (
    build_classification_models,
    evaluate_models_cv,
)
from ..aggregation.run_minimal_case_prediction_from_mirror import (
    main as run_minimal_prediction,
)
from ..aggregation.run_case_prediction_with_existing_infra import (
    main as run_prediction_with_infra,
)

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def cases(
    features_file: Path = typer.Option(
        ..., "--feats", help="Input QuoteFeatures JSONL file"
    ),
    outcomes_file: Path = typer.Option(
        ..., "--outcomes", help="Input Outcomes JSONL file"
    ),
    output_file: Path = typer.Option(..., "--out", help="Output CaseVector JSONL file"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="Aggregation configuration YAML"
    ),
    thresholds: Optional[str] = typer.Option(
        "token_2500", "--thresholds", help="Comma-separated threshold names"
    ),
    feature_config: Optional[str] = typer.Option(
        "all", "--features", help="Feature configuration: all, counts, probs, etc."
    ),
):
    """
    Aggregate quote-level features to case-level vectors.

    Takes QuoteFeatures and Outcomes files and produces aggregated CaseVector JSONL.
    """
    logger.info(f"Aggregating cases from {features_file} and {outcomes_file}")
    logger.info(f"Output: {output_file}")

    # Load configuration
    config = {}
    if config_file and config_file.exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

    # Parse thresholds
    threshold_names = (
        [t.strip() for t in thresholds.split(",")] if thresholds else ["token_2500"]
    )

    # Create feature configuration
    if feature_config == "all":
        feature_cfg = FeatureConfig(
            include_counts=True,
            include_mean_probabilities=True,
            include_confidence=True,
            include_pred_class=True,
            include_scores=False,
            include_position_stats=True,
            include_model_threshold=True,
        )
    elif feature_config == "counts":
        feature_cfg = FeatureConfig(include_counts=True)
    elif feature_config == "probs":
        feature_cfg = FeatureConfig(include_mean_probabilities=True)
    else:
        feature_cfg = FeatureConfig()

    # Parse threshold specifications
    threshold_specs = []
    for thresh_name in threshold_names:
        if thresh_name in DEFAULT_THRESHOLDS:
            threshold_specs.append(DEFAULT_THRESHOLDS[thresh_name])
        else:
            logger.warning(f"Unknown threshold: {thresh_name}")

    if not threshold_specs:
        logger.warning("No valid thresholds specified, using default")
        threshold_specs = [DEFAULT_THRESHOLDS["token_2500"]]

    # Load and process data
    logger.info("Loading quote features and outcomes...")

    # This would implement the actual aggregation logic
    # For now, we'll indicate what would be done
    logger.info(f"Would aggregate using {len(threshold_specs)} thresholds")
    logger.info(f"Feature config: {feature_cfg}")
    logger.info(f"Aggregation would be saved to {output_file}")


@app.command()
def predict(
    cases_file: Path = typer.Option(..., "--cases", help="Input CaseVector JSONL file"),
    output_file: Path = typer.Option(
        ..., "--out", help="Output predictions JSONL file"
    ),
    model_type: str = typer.Option("lr", "--model", help="Model type: lr, rf, xgb"),
    model_config: Optional[Path] = typer.Option(
        None, "--config", help="Model configuration file"
    ),
    fold: int = typer.Option(4, "--fold", help="Fold number for loading outcomes"),
):
    """
    Make case-level predictions using trained models.
    """
    logger.info(f"Making predictions on {cases_file} using {model_type} model")
    logger.info(f"Output: {output_file}")

    # Load model configuration
    config = {}
    if model_config and model_config.exists():
        import yaml

        with open(model_config) as f:
            config = yaml.safe_load(f)

    # This would implement the actual prediction logic
    # For now, we'll indicate what would be done
    logger.info(f"Would use fold {fold} for evaluation")
    logger.info(f"Predictions would be saved to {output_file}")


@app.command()
def evaluate(
    predictions_file: Path = typer.Option(
        ..., "--preds", help="Input predictions JSONL file"
    ),
    outcomes_file: Path = typer.Option(
        ..., "--outcomes", help="Input true outcomes JSONL file"
    ),
    output_dir: Path = typer.Option(
        ..., "--out", help="Output directory for evaluation results"
    ),
    metrics: Optional[str] = typer.Option(
        "all", "--metrics", help="Comma-separated metrics: auc,mcc,precision,recall"
    ),
):
    """
    Evaluate case-level predictions against true outcomes.
    """
    logger.info(f"Evaluating predictions from {predictions_file}")
    logger.info(f"True outcomes: {outcomes_file}")
    logger.info(f"Output directory: {output_dir}")

    # Parse metrics
    metric_list = (
        [m.strip() for m in metrics.split(",")]
        if metrics != "all"
        else ["auc", "mcc", "precision", "recall", "f1"]
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # This would implement the actual evaluation logic
    # For now, we'll indicate what would be done
    logger.info(f"Would compute metrics: {', '.join(metric_list)}")
    logger.info(f"Results would be saved to {output_dir}")


@app.command()
def minimal(
    mirror_dir: Path = typer.Option(
        ..., "--mirror", help="Mirror directory with predictions"
    ),
    output_dir: Path = typer.Option(..., "--out", help="Output directory"),
    fold: int = typer.Option(4, "--fold", help="Fold number"),
    feature_config: str = typer.Option(
        "E+3", "--features", help="Feature configuration"
    ),
):
    """
    Run minimal case prediction workflow.
    """
    logger.info(f"Running minimal prediction workflow")
    logger.info(f"Mirror: {mirror_dir}")
    logger.info(f"Output: {output_dir}")

    # This would call the run_minimal_case_prediction_from_mirror function
    logger.info("Would execute minimal case prediction workflow")


@app.command()
def infra(
    mirror_dir: Path = typer.Option(
        ..., "--mirror", help="Mirror directory with predictions"
    ),
    output_dir: Path = typer.Option(..., "--out", help="Output directory"),
    fold: int = typer.Option(4, "--fold", help="Fold number"),
    thresholds: Optional[str] = typer.Option(
        "token_2500", "--thresholds", help="Comma-separated threshold names"
    ),
):
    """
    Run case prediction with existing infrastructure.
    """
    logger.info(f"Running prediction with existing infrastructure")
    logger.info(f"Mirror: {mirror_dir}")
    logger.info(f"Output: {output_dir}")

    # This would call the run_case_prediction_with_existing_infra function
    logger.info("Would execute infrastructure-based prediction workflow")


if __name__ == "__main__":
    app()
