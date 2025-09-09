#!/usr/bin/env python3
"""
Enhanced case-level prediction using existing case_aggregation infrastructure.

Adapts the existing case_aggregation.modeling CLI to work with mirrored MLP predictions
while preserving the infrastructure for thresholds, feature engineering, and evaluation.

This version bridges the gap between the minimal KISS approach and the full infrastructure,
allowing you to leverage the existing threshold system while using the new mirrored predictions.

Usage:
    python scripts/run_case_prediction_with_existing_infra.py \
        --mirror-dir results/corrected_dnt_validation_FINAL/mirror_with_predictions \
        --output-dir results/case_prediction_with_infra \
        --thresholds token_2500 token_half token_third \
        --fold 4
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

# Import existing infrastructure
from src.corp_speech_risk_dataset.case_aggregation.modeling.dataset import (
    ThresholdSpec,
    FeatureConfig,
    DEFAULT_THRESHOLDS,
)
from src.corp_speech_risk_dataset.case_aggregation.modeling.models import (
    build_classification_models,
    evaluate_models_cv,
)
from src.corp_speech_risk_dataset.case_aggregation.modeling import reporting


def load_and_convert_mirror_to_quotes_format(mirror_dir: Path) -> List[Dict[str, Any]]:
    """
    Load mirrored predictions and convert to the format expected by case_aggregation.

    This bridges the new mirrored prediction format with the existing infrastructure.
    """
    logger.info(f"Loading and converting mirrored predictions from {mirror_dir}")

    rows = []

    # Walk through mirror directory
    for jsonl_file in mirror_dir.rglob("*.jsonl"):
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)

                        # Convert to expected format for case_aggregation
                        # Map MLP predictions to CORAL-style predictions
                        converted_row = {
                            "case_id": data.get("case_id"),
                            "doc_id": data.get("doc_id", ""),
                            "text": data.get("text", ""),
                            # Position features (if available)
                            "docket_number": data.get("docket_number", 1),
                            "global_token_start": data.get("global_token_start", 0),
                            "num_tokens": data.get(
                                "num_tokens", len(data.get("text", "").split())
                            ),
                            # Convert MLP predictions to CORAL format
                            "coral_pred_class": _map_mlp_to_coral_class(data),
                            "coral_pred_bucket": _map_mlp_to_coral_bucket(data),
                            "coral_prob_low": 1.0 - data.get("mlp_probability", 0.5),
                            "coral_prob_medium": 0.5,  # Placeholder for binary->tertile mapping
                            "coral_prob_high": data.get("mlp_probability", 0.5),
                            "coral_confidence": abs(
                                data.get("mlp_probability", 0.5) - 0.5
                            )
                            * 2,
                            "coral_model_threshold": data.get(
                                "mlp_threshold_strict", 0.5
                            ),
                            # Preserve original MLP predictions for reference
                            "mlp_probability": data.get("mlp_probability"),
                            "mlp_pred_strict": data.get("mlp_pred_strict"),
                            "mlp_pred_recallT": data.get("mlp_pred_recallT"),
                            "mlp_threshold_strict": data.get("mlp_threshold_strict"),
                            "mlp_threshold_recallT": data.get("mlp_threshold_recallT"),
                        }

                        rows.append(converted_row)

        except Exception as e:
            logger.warning(f"Error processing {jsonl_file}: {e}")
            continue

    logger.info(f"Converted {len(rows)} quote predictions for case aggregation")
    return rows


def _map_mlp_to_coral_class(data: Dict[str, Any]) -> int:
    """Map MLP binary predictions to CORAL 3-class format."""
    prob = data.get("mlp_probability", 0.5)

    if prob < 0.33:
        return 0  # low
    elif prob < 0.67:
        return 1  # medium
    else:
        return 2  # high


def _map_mlp_to_coral_bucket(data: Dict[str, Any]) -> str:
    """Map MLP binary predictions to CORAL bucket names."""
    class_idx = _map_mlp_to_coral_class(data)
    return ["low", "medium", "high"][class_idx]


def build_enhanced_feature_config() -> FeatureConfig:
    """
    Build feature configuration optimized for the minimal approach.

    Focus on density and positional features that align with the KISS principles.
    """
    return FeatureConfig(
        include_counts=True,  # Core density features
        include_mean_probabilities=True,  # Mean risk scores
        include_confidence=True,  # Confidence measures
        include_pred_class=True,  # Use class predictions
        include_scores=False,  # Skip scores for simplicity
        include_position_stats=True,  # Key for positional analysis
        include_model_threshold=True,  # Threshold information
        use_pred_class_only=False,  # Use both probs and classes
    )


def get_custom_thresholds() -> List[ThresholdSpec]:
    """Define thresholds optimized for early prediction analysis."""
    return [
        ThresholdSpec(name="complete_case", kind="complete", value=1.0),
        ThresholdSpec(name="token_2500", kind="token_budget", value=2500.0),
        ThresholdSpec(name="token_1000", kind="token_budget", value=1000.0),
        ThresholdSpec(name="token_half", kind="token_fraction", value=0.5),
        ThresholdSpec(name="token_third", kind="token_fraction", value=1.0 / 3.0),
        ThresholdSpec(name="token_quarter", kind="token_fraction", value=0.25),
        ThresholdSpec(name="docket_half", kind="docket_fraction", value=0.5),
        ThresholdSpec(name="docket_third", kind="docket_fraction", value=1.0 / 3.0),
    ]


def adapt_case_aggregation_workflow(
    rows: List[Dict[str, Any]],
    thresholds: List[ThresholdSpec],
    features: FeatureConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Adapt the existing case_aggregation workflow for mirrored predictions.
    """
    logger.info("Running adapted case aggregation workflow...")

    # Convert to Polars DataFrame (as expected by existing code)
    df = pl.DataFrame(rows)

    # Import the existing dataset building function
    from src.corp_speech_risk_dataset.case_aggregation.modeling.dataset import (
        build_datasets,
    )

    # Build datasets using existing infrastructure
    datasets = build_datasets(rows, thresholds=thresholds, features=features)

    if not datasets:
        raise ValueError("No datasets were built")

    results = {}

    # Process each threshold dataset
    for threshold_name, bundle in datasets.items():
        logger.info(f"Processing threshold: {threshold_name}")

        # Extract features and outcomes
        X = bundle.features  # Polars DataFrame
        y = bundle.outcomes  # Polars DataFrame

        if len(X) == 0:
            logger.warning(f"No data for threshold {threshold_name}")
            continue

        # Convert to pandas for sklearn compatibility
        X_pd = X.to_pandas()
        y_pd = y.to_pandas()

        # Merge on case_id
        merged = X_pd.merge(y_pd, on="case_id", how="inner")

        if len(merged) < 10:
            logger.warning(
                f"Insufficient data for threshold {threshold_name}: {len(merged)} cases"
            )
            continue

        # Simple LR training (following KISS principles)
        feature_cols = [col for col in X_pd.columns if col != "case_id"]
        X_features = merged[feature_cols].values
        y_outcome = merged["outcome_bucket"].values

        # Convert tertile to binary if needed
        if len(np.unique(y_outcome)) == 3:
            # Convert to binary: high vs (low+medium)
            y_binary = (y_outcome == 2).astype(int)
        else:
            y_binary = y_outcome

        # Simple logistic regression with court suppression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import matthews_corrcoef, make_scorer

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)

        # Extract case IDs for grouping
        case_ids = merged["case_id"].values

        # Simple LR with balanced class weights
        lr_model = LogisticRegression(
            penalty="l2", C=0.1, class_weight="balanced", max_iter=2000
        )

        # Cross-validation with case grouping
        gkf = GroupKFold(n_splits=5)
        mcc_scorer = make_scorer(matthews_corrcoef)

        cv_scores = cross_val_score(
            lr_model, X_scaled, y_binary, groups=case_ids, cv=gkf, scoring=mcc_scorer
        )

        # Train final model
        lr_model.fit(X_scaled, y_binary)

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "coefficient": lr_model.coef_[0],
                "abs_coefficient": np.abs(lr_model.coef_[0]),
            }
        ).sort_values("abs_coefficient", ascending=False)

        results[threshold_name] = {
            "n_cases": len(merged),
            "n_features": len(feature_cols),
            "cv_mcc_mean": float(cv_scores.mean()),
            "cv_mcc_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "feature_importance": feature_importance.to_dict("records"),
            "class_distribution": pd.Series(y_binary).value_counts().to_dict(),
            "feature_names": feature_cols,
        }

        logger.info(
            f"{threshold_name} - CV MCC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Case prediction using existing infrastructure"
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        required=True,
        help="Directory containing mirrored predictions",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=["token_2500", "token_half", "token_third"],
        help="Threshold names to test",
    )
    parser.add_argument(
        "--fold", type=int, default=4, help="Fold number for loading outcomes"
    )

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(args.output_dir / "case_prediction_infra.log")

    logger.info("=== CASE PREDICTION WITH EXISTING INFRASTRUCTURE ===")
    logger.info(f"Mirror dir: {args.mirror_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Thresholds: {args.thresholds}")

    try:
        # 1. Load and convert mirrored predictions
        rows = load_and_convert_mirror_to_quotes_format(args.mirror_dir)

        if not rows:
            raise ValueError("No data loaded from mirror directory")

        # 2. Parse threshold specifications
        available_thresholds = {t.name: t for t in get_custom_thresholds()}
        selected_thresholds = []

        for thresh_name in args.thresholds:
            if thresh_name in available_thresholds:
                selected_thresholds.append(available_thresholds[thresh_name])
            else:
                logger.warning(f"Unknown threshold: {thresh_name}")

        if not selected_thresholds:
            raise ValueError(
                f"No valid thresholds specified. Available: {list(available_thresholds.keys())}"
            )

        # 3. Configure features for minimal approach
        feature_config = build_enhanced_feature_config()

        # 4. Run adapted workflow
        results = adapt_case_aggregation_workflow(
            rows, selected_thresholds, feature_config, args.output_dir
        )

        # 5. Save results
        with open(args.output_dir / "threshold_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # 6. Generate summary report
        with open(args.output_dir / "executive_summary.txt", "w") as f:
            f.write("CASE-LEVEL PREDICTION RESULTS\n")
            f.write("=" * 50 + "\n\n")

            for threshold_name, result in results.items():
                f.write(f"Threshold: {threshold_name}\n")
                f.write(f"  Cases: {result['n_cases']}\n")
                f.write(f"  Features: {result['n_features']}\n")
                f.write(
                    f"  CV MCC: {result['cv_mcc_mean']:.4f} ± {result['cv_mcc_std']:.4f}\n"
                )
                f.write(f"  Class balance: {result['class_distribution']}\n")
                f.write("\n")

                f.write("  Top features:\n")
                for feat in result["feature_importance"][:5]:
                    f.write(f"    {feat['feature']}: {feat['coefficient']:.4f}\n")
                f.write("\n")

        logger.info("=== CASE PREDICTION COMPLETE ===")
        logger.info(f"Results saved to: {args.output_dir}")

        # Print summary
        print("\n" + "=" * 60)
        print("CASE PREDICTION SUMMARY")
        print("=" * 60)

        for threshold_name, result in results.items():
            print(f"\n{threshold_name.upper()}:")
            print(f"  Cases: {result['n_cases']}")
            print(f"  CV MCC: {result['cv_mcc_mean']:.4f} ± {result['cv_mcc_std']:.4f}")
            print(
                f"  Top features: {', '.join([f['feature'] for f in result['feature_importance'][:3]])}"
            )

    except Exception as e:
        logger.error(f"Error in case prediction workflow: {e}")
        raise


if __name__ == "__main__":
    main()
