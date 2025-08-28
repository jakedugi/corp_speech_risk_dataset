#!/usr/bin/env python3
"""Comprehensive Binary POLR Pipeline Runner

This script provides a complete interface for running the binary POLR pipeline in different modes
while ensuring all weights and labels are inherited from the binary dataset.

CRITICAL: This script NEVER recomputes weights or labels - everything is inherited
from the pre-computed binary data in data/final_stratified_kfold_splits_binary_quote_balanced/

Features:
- Binary classification (2 classes: low/high risk)
- Exactly 10 features from feature importance analysis
- 4-fold CV (fold_0,1,2,3) + fold_4 final training
- Built-in validation and confirmation steps
- Support for both POLR and MLR models
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


@dataclass
class BinaryComprehensiveConfig:
    """Configuration for comprehensive binary POLR pipeline runner."""

    # Data paths (FIXED - always use binary data)
    kfold_dir: str = "data/final_stratified_kfold_splits_binary_quote_balanced"
    output_dir: str = "runs/binary_polr_comprehensive"

    # Mode selection
    mode: str = "full"  # cv-only, final-only, full, validate, assets

    # Model selection
    model_type: str = "polr"  # "polr" or "mlr"

    # Hyperparameter configuration
    hyperparameter_preset: str = "default"  # default, fast, thorough, custom
    custom_hyperparameters: Optional[Dict[str, List[Any]]] = None

    # Pipeline options
    generate_assets: bool = True
    skip_validation: bool = False
    require_confirmation: bool = True

    # Advanced options
    seed: int = 42
    n_jobs: int = -1

    def __post_init__(self):
        """Validate configuration and set hyperparameter grid."""
        if self.hyperparameter_preset == "fast":
            if self.model_type == "lr_elasticnet":
                self.custom_hyperparameters = {
                    "C": [1.0],
                    "l1_ratio": [0.5],
                    "solver": ["saga"],
                    "max_iter": [200],
                }
            elif self.model_type == "lr_l1":
                self.custom_hyperparameters = {
                    "C": [1.0],
                    "solver": ["liblinear"],
                    "max_iter": [200],
                }
            elif self.model_type == "svm_linear":
                self.custom_hyperparameters = {
                    "C": [1.0],
                    "loss": ["squared_hinge"],
                    "max_iter": [1000],
                }
            else:
                self.custom_hyperparameters = {
                    "C": [1.0],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                }
        elif self.hyperparameter_preset == "thorough":
            self.custom_hyperparameters = {
                "C": [0.001, 0.01, 0.1, 1.0, 10, 100],
                "solver": ["lbfgs", "newton-cg"],
                "max_iter": [200, 500, 1000],
            }
        elif self.hyperparameter_preset == "default":
            if self.model_type == "mlr":
                self.custom_hyperparameters = {
                    "C": [0.01, 1, 100],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                    "tol": [1e-4],
                    "class_weight": ["balanced", None],
                }
            elif self.model_type == "lr_l2":
                self.custom_hyperparameters = {
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                    "tol": [1e-4],
                }
            elif self.model_type == "lr_l1":
                self.custom_hyperparameters = {
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["liblinear"],
                    "max_iter": [200],
                    "tol": [1e-4],
                }
            elif self.model_type == "lr_elasticnet":
                self.custom_hyperparameters = {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "solver": ["saga"],
                    "max_iter": [500],
                    "tol": [1e-4],
                }
            elif self.model_type == "svm_linear":
                self.custom_hyperparameters = {
                    "C": [0.03, 0.1, 0.3, 1.0, 3.0],
                    "loss": ["squared_hinge"],
                    "max_iter": [5000],
                    "tol": [1e-4],
                }
            else:
                self.custom_hyperparameters = {
                    "C": [0.01, 1, 100],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                }


class BinaryValidationError(Exception):
    """Raised when binary validation checks fail."""

    pass


class BinaryPipelineRunner:
    """Comprehensive binary POLR pipeline runner with validation and safety checks."""

    def __init__(self, config: BinaryComprehensiveConfig):
        self.config = config
        self.setup_logging()
        self.validate_config()

    def setup_logging(self):
        """Setup comprehensive logging."""
        logger.remove()

        # Console logging
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        # File logging
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.add(
            output_dir
            / f"binary_polr_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            level="DEBUG",
            rotation="100 MB",
            retention="7 days",
        )

    def validate_config(self):
        """Validate configuration and paths."""
        logger.info("🔍 Validating binary configuration...")

        # Check binary data directory
        kfold_path = Path(self.config.kfold_dir)
        if not kfold_path.exists():
            raise BinaryValidationError(
                f"Binary data directory not found: {kfold_path}"
            )

        # Check required files
        required_files = ["per_fold_metadata.json", "fold_statistics.json"]

        for file_name in required_files:
            file_path = kfold_path / file_name
            if not file_path.exists():
                raise BinaryValidationError(
                    f"Required metadata file not found: {file_path}"
                )

        # Check fold directories (0,1,2,3,4 + oof_test)
        required_folds = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "oof_test"]
        for fold_name in required_folds:
            fold_path = kfold_path / fold_name
            if not fold_path.exists():
                raise BinaryValidationError(
                    f"Required fold directory not found: {fold_path}"
                )

        logger.info("✅ Binary configuration validation passed")

    def validate_binary_data_integrity(self) -> Dict[str, Any]:
        """Validate binary data integrity and weight inheritance."""
        logger.info("🔍 Validating binary data integrity and weight inheritance...")

        kfold_path = Path(self.config.kfold_dir)
        validation_results = {
            "metadata_valid": False,
            "weights_inherited": False,
            "labels_inherited": False,
            "fold_consistency": False,
            "binary_classification": False,
            "issues": [],
        }

        try:
            # Load and validate metadata
            metadata_path = kfold_path / "per_fold_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check metadata structure
            required_keys = ["binning", "weights"]
            for key in required_keys:
                if key not in metadata:
                    validation_results["issues"].append(f"Missing metadata key: {key}")

            # Check binary classification type
            if metadata.get("binning", {}).get("classification_type") == "binary":
                validation_results["binary_classification"] = True
            else:
                validation_results["issues"].append(
                    "Metadata does not indicate binary classification"
                )

            # Check binning metadata for 5 folds
            if "binning" in metadata:
                binning = metadata["binning"]
                if "fold_edges" in binning:
                    fold_edges = binning["fold_edges"]
                    expected_folds = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

                    for fold in expected_folds:
                        if fold not in fold_edges:
                            validation_results["issues"].append(
                                f"Missing fold edges for {fold}"
                            )
                        else:
                            edges = fold_edges[fold]
                            # Binary should have single edge
                            if not isinstance(edges, list) or len(edges) != 1:
                                validation_results["issues"].append(
                                    f"Invalid binary edges for {fold}: {edges}"
                                )

                    if not any(
                        "Missing fold edges" in issue
                        for issue in validation_results["issues"]
                    ):
                        validation_results["labels_inherited"] = True

            # Check weights metadata for 5 folds
            if "weights" in metadata:
                weights = metadata["weights"]
                expected_folds = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

                for fold in expected_folds:
                    if fold not in weights:
                        validation_results["issues"].append(
                            f"Missing weights for {fold}"
                        )
                    else:
                        fold_weights = weights[fold]
                        if "class_weights" not in fold_weights:
                            validation_results["issues"].append(
                                f"Missing class_weights for {fold}"
                            )
                        else:
                            # Check binary class weights (should have keys "0" and "1")
                            class_weights = fold_weights["class_weights"]
                            if not all(k in class_weights for k in ["0", "1"]):
                                validation_results["issues"].append(
                                    f"Missing binary class weights for {fold}"
                                )

                if not any(
                    "Missing weights" in issue or "Missing class_weights" in issue
                    for issue in validation_results["issues"]
                ):
                    validation_results["weights_inherited"] = True

            validation_results["metadata_valid"] = not validation_results["issues"]

            # Check sample data for precomputed fields
            logger.info("🔍 Checking sample binary data for precomputed fields...")
            sample_checks = []

            for fold_name in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]:
                fold_dir = kfold_path / fold_name
                train_file = fold_dir / "train.jsonl"

                if train_file.exists():
                    # Check first few rows
                    sample_data = []
                    with open(train_file, "r") as f:
                        for i, line in enumerate(f):
                            if i >= 5:  # Check first 5 rows
                                break
                            sample_data.append(json.loads(line))

                    if sample_data:
                        sample_row = sample_data[0]
                        required_fields = ["outcome_bin", "sample_weight"]
                        missing_fields = []

                        for field in required_fields:
                            if field not in sample_row:
                                missing_fields.append(field)

                        # Check binary target values
                        if "outcome_bin" in sample_row:
                            outcome_value = sample_row["outcome_bin"]
                            if outcome_value not in [0, 1]:
                                validation_results["issues"].append(
                                    f"{fold_name}: outcome_bin has non-binary value {outcome_value}"
                                )

                        if missing_fields:
                            sample_checks.append(
                                f"{fold_name}: missing {missing_fields}"
                            )
                        else:
                            sample_checks.append(
                                f"{fold_name}: ✅ all required binary fields present"
                            )

            if sample_checks:
                logger.info("Binary data sample checks:")
                for check in sample_checks:
                    if "✅" in check:
                        logger.info(f"  {check}")
                    else:
                        logger.warning(f"  {check}")
                        validation_results["issues"].append(check)

            # Check OOF test set
            oof_test_file = kfold_path / "oof_test" / "test.jsonl"
            if oof_test_file.exists():
                with open(oof_test_file, "r") as f:
                    first_line = f.readline()
                    if first_line:
                        oof_sample = json.loads(first_line)
                        if "outcome_bin" in oof_sample:
                            if oof_sample["outcome_bin"] in [0, 1]:
                                logger.info(
                                    "✅ OOF test set has binary outcome_bin labels"
                                )
                            else:
                                validation_results["issues"].append(
                                    f"OOF test set has non-binary outcome_bin value: {oof_sample['outcome_bin']}"
                                )
                        else:
                            validation_results["issues"].append(
                                "OOF test set missing outcome_bin labels"
                            )

            validation_results["fold_consistency"] = (
                len(validation_results["issues"]) == 0
            )

            # Summary
            if not validation_results["issues"]:
                logger.info(
                    "✅ Binary data integrity validation passed - all weights and labels properly inherited"
                )
            else:
                logger.error("❌ Binary data integrity validation failed:")
                for issue in validation_results["issues"]:
                    logger.error(f"  - {issue}")

        except Exception as e:
            logger.error(
                f"❌ Binary data integrity validation failed with exception: {e}"
            )
            validation_results["issues"].append(f"Validation exception: {str(e)}")

        return validation_results

    def get_user_confirmation(self, message: str) -> bool:
        """Get user confirmation for critical operations."""
        if not self.config.require_confirmation:
            return True

        logger.info(f"🤔 {message}")
        response = input("Continue? [y/N]: ").lower().strip()
        return response in ["y", "yes"]

    def create_binary_polar_config(self) -> BinaryPOLARConfig:
        """Create binary POLAR configuration from comprehensive config."""
        return BinaryPOLARConfig(
            kfold_dir=self.config.kfold_dir,
            output_dir=self.config.output_dir,
            model_type=self.config.model_type,
            hyperparameter_grid=self.config.custom_hyperparameters,
            seed=self.config.seed,
            n_jobs=self.config.n_jobs,
        )

    def run_cv_mode(self) -> Dict[str, Any]:
        """Run cross-validation only mode."""
        logger.info("🔄 Starting binary CV-only mode...")

        if not self.get_user_confirmation(
            "This will run binary hyperparameter search using folds 0, 1, 2, 3"
        ):
            logger.info("❌ Binary CV mode cancelled by user")
            return {}

        binary_config = self.create_binary_polar_config()

        logger.info("🚀 Starting binary cross-validation...")
        cv_results = train_binary_polar_cv(binary_config)

        logger.info("✅ Binary CV-only mode completed")
        return cv_results

    def run_final_mode(self) -> Dict[str, Any]:
        """Run final binary model training only mode."""
        logger.info("🎯 Starting binary final-only mode...")

        # Check for existing CV results; if missing, proceed with defaults
        cv_results_path = Path(self.config.output_dir) / "cv_results.json"
        if not cv_results_path.exists():
            logger.warning(
                f"CV results not found at {cv_results_path}. Proceeding with default hyperparameters for final training."
            )
            cv_results = {"folds": {}}
        else:
            with open(cv_results_path, "r") as f:
                cv_results = json.load(f)

        if not self.get_user_confirmation(
            "This will train final binary model on fold_4 and evaluate on OOF test"
        ):
            logger.info("❌ Binary final mode cancelled by user")
            return {}

        binary_config = self.create_binary_polar_config()

        logger.info("🚀 Starting final binary model training...")
        final_results = train_final_binary_polar_model(binary_config, cv_results)

        logger.info("✅ Binary final-only mode completed")
        return final_results

    def run_full_mode(self) -> Dict[str, Any]:
        """Run complete binary pipeline mode."""
        logger.info("🎭 Starting binary full pipeline mode...")

        if not self.get_user_confirmation(
            "This will run complete binary pipeline: 4-fold CV + final training on fold_4 + OOF evaluation"
        ):
            logger.info("❌ Binary full mode cancelled by user")
            return {}

        binary_config = self.create_binary_polar_config()

        # Run CV
        logger.info("🚀 Phase 1: Binary cross-validation...")
        cv_results = train_binary_polar_cv(binary_config)

        # Run final training
        logger.info("🚀 Phase 2: Final binary model training...")
        final_results = train_final_binary_polar_model(binary_config, cv_results)

        logger.info("✅ Binary full pipeline completed")
        return {"cv_results": cv_results, "final_results": final_results}

    def run_validate_mode(self) -> Dict[str, Any]:
        """Run binary validation-only mode."""
        logger.info("🔍 Starting binary validation-only mode...")

        validation_results = self.validate_binary_data_integrity()

        # Print detailed validation report
        logger.info("\n" + "=" * 80)
        logger.info("BINARY VALIDATION REPORT")
        logger.info("=" * 80)

        logger.info(
            f"Metadata valid: {'✅' if validation_results['metadata_valid'] else '❌'}"
        )
        logger.info(
            f"Weights inherited: {'✅' if validation_results['weights_inherited'] else '❌'}"
        )
        logger.info(
            f"Labels inherited: {'✅' if validation_results['labels_inherited'] else '❌'}"
        )
        logger.info(
            f"Fold consistency: {'✅' if validation_results['fold_consistency'] else '❌'}"
        )
        logger.info(
            f"Binary classification: {'✅' if validation_results['binary_classification'] else '❌'}"
        )

        if validation_results["issues"]:
            logger.info("\nIssues found:")
            for issue in validation_results["issues"]:
                logger.warning(f"  - {issue}")
        else:
            logger.info("\n🎉 All binary validation checks passed!")

        logger.info("=" * 80)

        return validation_results

    def run(self) -> Dict[str, Any]:
        """Run the binary pipeline in the specified mode."""
        logger.info("🚀 Starting Binary POLR Comprehensive Pipeline Runner")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.config.mode}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Binary data directory: {self.config.kfold_dir}")
        logger.info(f"Model type: {self.config.model_type}")
        logger.info(f"Hyperparameter preset: {self.config.hyperparameter_preset}")
        logger.info("=" * 80)

        # Always validate binary data integrity unless skipped
        if not self.config.skip_validation:
            validation_results = self.validate_binary_data_integrity()
            if validation_results["issues"] and not self.get_user_confirmation(
                f"Found {len(validation_results['issues'])} validation issues. Continue anyway?"
            ):
                logger.error("❌ Pipeline cancelled due to validation issues")
                return {"status": "cancelled", "validation": validation_results}

        # Run mode-specific logic
        try:
            if self.config.mode == "cv-only":
                results = self.run_cv_mode()
            elif self.config.mode == "final-only":
                results = self.run_final_mode()
            elif self.config.mode == "full":
                results = self.run_full_mode()
            elif self.config.mode == "validate":
                results = self.run_validate_mode()
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

            logger.info("\n🎉 Binary pipeline completed successfully!")
            return {"status": "success", "results": results}

        except Exception as e:
            logger.error(f"❌ Binary pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Binary POLR Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full binary pipeline with default settings (POLR)
  python scripts/run_binary_polr_comprehensive.py --mode full

  # Test multinomial LR with binary data
  python scripts/run_binary_polr_comprehensive.py --mode full --model mlr --output-dir runs/binary_mlr

  # CV only with thorough hyperparameter search
  python scripts/run_binary_polr_comprehensive.py --mode cv-only --hyperparameters thorough

  # Validate binary data integrity
  python scripts/run_binary_polr_comprehensive.py --mode validate

  # Fast development run with MLR
  python scripts/run_binary_polr_comprehensive.py --mode full --model mlr --hyperparameters fast --no-confirmation
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["cv-only", "final-only", "full", "validate"],
        default="full",
        help="Pipeline mode to run (default: full)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["polr", "mlr", "lr_l2", "lr_l1", "lr_elasticnet", "svm_linear"],
        default="polr",
        help=(
            "Model type: polr (Proportional Odds LR), mlr (Multinomial LR), "
            "lr_l2 (Logistic L2), lr_l1 (Logistic L1), lr_elasticnet (ElasticNet Logistic), "
            "svm_linear (Linear SVM with Platt calibration)"
        ),
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        default="data/final_stratified_kfold_splits_binary_quote_balanced",
        help="Binary data directory (default: data/final_stratified_kfold_splits_binary_quote_balanced)",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: runs/binary_polr_comprehensive_TIMESTAMP)",
    )

    # Hyperparameter configuration
    parser.add_argument(
        "--hyperparameters",
        choices=["default", "fast", "thorough", "custom"],
        default="default",
        help="Hyperparameter preset (default: default)",
    )

    parser.add_argument(
        "--custom-hyperparameters",
        type=str,
        help="Custom hyperparameters as JSON string (for custom preset)",
    )

    # Safety and validation
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip binary data integrity validation",
    )

    parser.add_argument(
        "--no-confirmation", action="store_true", help="Skip user confirmation prompts"
    )

    # General options
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel jobs (default: -1)"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"runs/binary_polr_comprehensive_{timestamp}"

    # Handle custom hyperparameters
    custom_hyperparameters = None
    if args.hyperparameters == "custom":
        if args.custom_hyperparameters:
            try:
                custom_hyperparameters = json.loads(args.custom_hyperparameters)
            except json.JSONDecodeError as e:
                print(f"Error parsing custom hyperparameters: {e}")
                sys.exit(1)
        else:
            print("Error: --custom-hyperparameters required when using custom preset")
            sys.exit(1)

    # Create configuration
    config = BinaryComprehensiveConfig(
        kfold_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        model_type=args.model,
        hyperparameter_preset=args.hyperparameters,
        custom_hyperparameters=custom_hyperparameters,
        skip_validation=args.skip_validation,
        require_confirmation=not args.no_confirmation,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    # Run pipeline
    runner = BinaryPipelineRunner(config)
    results = runner.run()

    # Print final results
    print("\n" + "=" * 80)
    print("BINARY PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Status: {results['status']}")
    if results["status"] == "success":
        print(f"Output directory: {config.output_dir}")
        print("✅ Binary pipeline completed successfully!")
        print("\nKey features:")
        print("  - Binary classification (low/high risk)")
        print("  - 10 features from importance analysis")
        print("  - 4-fold CV + fold_4 final training")
        print("  - Inherited weights and labels")
        print("  - polr_ prefixed predictions")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
