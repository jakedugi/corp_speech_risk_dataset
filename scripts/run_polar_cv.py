#!/usr/bin/env python3
"""Run POLAR model with full cross-validation protocol.

This script implements the complete paper-quality POLAR training pipeline
including column governance, per-fold weights, hyperparameter search,
calibration, and comprehensive evaluation.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.polar_pipeline import (
    POLARConfig,
    train_polar_cv,
    train_final_polar_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train POLAR model with paper-quality CV protocol"
    )

    parser.add_argument(
        "--kfold-dir",
        type=str,
        default="data/final_stratified_kfold_splits_authoritative",
        help="Directory containing k-fold splits",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/polar_experiment",
        help="Output directory for results and models",
    )

    # REMOVED: --compute-tertiles option (always inherit precomputed labels from authoritative data)

    # REMOVED: --continuous-target option (not needed when inheriting precomputed labels)

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)",
    )

    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip training final model (only do CV)",
    )

    # Temporal DEV policy parameters (tiny-data friendly)
    parser.add_argument(
        "--dev-tail-frac",
        type=float,
        default=0.20,
        help="Starting fraction for DEV tail (will increase if needed)",
    )
    parser.add_argument(
        "--min-dev-cases",
        type=int,
        default=3,
        help="Minimum DEV cases required (tiny-data friendly)",
    )
    parser.add_argument(
        "--min-dev-quotes",
        type=int,
        default=150,
        help="Minimum DEV quotes required (fallback to 100)",
    )
    parser.add_argument(
        "--require-all-classes",
        action="store_true",
        help="Require all 3 classes in DEV (default: accept ≥2)",
    )
    parser.add_argument(
        "--embargo-days", type=int, default=90, help="Days to embargo before DEV"
    )
    parser.add_argument(
        "--safe-qwk",
        action="store_true",
        default=True,
        help="Use safe QWK with fallbacks",
    )
    parser.add_argument(
        "--min-cal-n",
        type=int,
        default=100,
        help="Minimum samples for direct isotonic (else use binning)",
    )
    parser.add_argument(
        "--iso-bins",
        type=int,
        default=30,
        help="Number of quantile bins for small-sample isotonic",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=50,
        help="Maximum categories before applying top-K + __OTHER__",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        Path(args.output_dir) / "polr_training.log", level="DEBUG", rotation="100 MB"
    )

    logger.info("=" * 60)
    logger.info("POLR MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"K-fold directory: {args.kfold_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using authoritative precomputed labels and weights")
    logger.info(f"Random seed: {args.seed}")

    # Create configuration
    config = POLARConfig(
        kfold_dir=args.kfold_dir,
        output_dir=args.output_dir,
        # Always inherit precomputed labels and weights from authoritative data
        continuous_target_field="final_judgement_real",  # For reference only
        seed=args.seed,
        n_jobs=args.n_jobs,
        # Temporal DEV policy
        dev_tail_frac=args.dev_tail_frac,
        min_dev_cases=args.min_dev_cases,
        min_dev_quotes=args.min_dev_quotes,
        require_all_classes=args.require_all_classes,
        embargo_days=args.embargo_days,
        safe_qwk=args.safe_qwk,
        min_cal_n=args.min_cal_n,
        iso_bins=args.iso_bins,
        max_categories=args.max_categories,
    )

    # Run cross-validation
    logger.info("\nStarting cross-validation...")
    cv_results = train_polar_cv(config)

    # Train final model
    if not args.skip_final:
        logger.info("\nTraining final model...")
        final_metadata = train_final_polar_model(config, cv_results)
        logger.info("\nTraining complete!")
        logger.info(f"Final model saved to: {final_metadata['model_path']}")
    else:
        logger.info("\nSkipping final model training (--skip-final flag set)")

    logger.info("\n" + "=" * 60)
    logger.info("POLAR TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
