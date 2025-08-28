#!/usr/bin/env python3
"""2-Fold Binary POLR Pipeline Runner

Modified version of run_binary_polr_comprehensive.py for 2-fold datasets.
This script adapts the binary POLR pipeline to work with 2-fold cross-validation data.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.binary_polar_pipeline import (
    BinaryPOLARConfig,
    train_binary_polar_cv,
    train_final_binary_polar_model,
)


def run_2fold_binary_analysis(data_dir: str, output_dir: str, model_type: str = "polr"):
    """Run binary POLR analysis on 2-fold data."""

    print("=" * 60)
    print("2-FOLD BINARY POLR ANALYSIS")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model type: {model_type}")
    print()

    # Check data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        return

    # Check fold structure
    expected_folds = ["fold_0", "fold_1", "fold_2"]
    missing_folds = []
    for fold in expected_folds:
        if not (data_path / fold).exists():
            missing_folds.append(fold)

    if missing_folds:
        print(f"❌ Error: Missing fold directories: {missing_folds}")
        return

    print("✅ Data structure validated")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create binary POLAR config adapted for 2-fold
    config = BinaryPOLARConfig(
        kfold_dir=data_dir,
        output_dir=output_dir,
        model_type=model_type,
        # CV parameters adapted for 2-fold
        cv_folds=[0, 1],  # Use folds 0 and 1 for CV
        final_fold=2,  # Use fold 2 as final training fold
        # Fast hyperparameters for testing
        hyperparameters={"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"], "max_iter": [200]},
        # Other standard settings
        seed=42,
        n_jobs=-1,
    )

    print("🚀 Starting 2-fold binary POLR analysis...")

    try:
        # Phase 1: Cross-validation on folds 0 and 1
        print("📊 Phase 1: Cross-validation (folds 0, 1)...")
        cv_results = train_binary_polar_cv(config)

        if cv_results:
            print("✅ Cross-validation completed successfully")
            print(f"   Best CV score: {cv_results.get('best_score', 'N/A')}")
            print(f"   Best parameters: {cv_results.get('best_params', 'N/A')}")

        # Phase 2: Final model training on fold 2
        print("🎯 Phase 2: Final model training (fold 2)...")
        final_results = train_final_binary_polar_model(config, cv_results)

        if final_results:
            print("✅ Final model training completed successfully")

            # Extract key metrics if available
            if "oof_metrics" in final_results:
                metrics = final_results["oof_metrics"]
                print("📊 Out-of-fold test results:")
                print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                print(f"   F1-Score: {metrics.get('f1', 'N/A'):.3f}")
                print(f"   ROC-AUC: {metrics.get('roc_auc', 'N/A'):.3f}")
                print(f"   Precision: {metrics.get('precision', 'N/A'):.3f}")
                print(f"   Recall: {metrics.get('recall', 'N/A'):.3f}")

        print(f"\n✅ 2-fold binary POLR analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")

        return {
            "status": "success",
            "cv_results": cv_results,
            "final_results": final_results,
            "config": asdict(config),
        }

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logger.exception("Full error details:")
        return {"status": "failed", "error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="2-Fold Binary POLR Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        default="data/final_stratified_kfold_splits_binary_quote_balanced_2fold",
        help="2-fold binary data directory",
    )

    parser.add_argument(
        "--output-dir",
        default=f"runs/binary_polr_2fold_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Output directory for results",
    )

    parser.add_argument(
        "--model",
        choices=["polr", "mlr", "lr_l2", "lr_l1", "lr_elasticnet"],
        default="polr",
        help="Model type to use",
    )

    args = parser.parse_args()

    # Run the analysis
    results = run_2fold_binary_analysis(
        data_dir=args.data_dir, output_dir=args.output_dir, model_type=args.model
    )

    return results


if __name__ == "__main__":
    main()
