#!/usr/bin/env python3
"""Comprehensive POLR Pipeline Runner

This script provides a complete interface for running the POLR pipeline in different modes
while ensuring all weights and labels are inherited from the authoritative dataset.

CRITICAL: This script NEVER recomputes weights, tertiles, or labels - everything is inherited
from the pre-computed authoritative data in data/final_stratified_kfold_splits_authoritative/

Modes:
1. cv-only: Run cross-validation for hyperparameter search only
2. final-only: Train final model (requires existing CV results)
3. full: Complete pipeline (CV + final training + evaluation)
4. validate: Validate data integrity and weight inheritance
5. assets: Generate paper assets from existing results

Features:
- Built-in validation and confirmation steps
- Flexible hyperparameter configuration
- Comprehensive logging and progress reporting
- Optional figure and table generation
- Sanity checks and data integrity validation
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.polar_pipeline import (
    POLARConfig,
    train_polar_cv,
    train_final_polar_model,
)


@dataclass
class ComprehensiveConfig:
    """Configuration for comprehensive POLR pipeline runner."""

    # Data paths (FIXED - always use authoritative data)
    kfold_dir: str = "data/final_stratified_kfold_splits_authoritative"
    output_dir: str = "runs/polr_comprehensive"

    # Mode selection
    mode: str = "full"  # cv-only, final-only, full, validate, assets

    # Model selection
    model_type: str = "polr"  # "polr" or "mlr"

    # Hyperparameter configuration
    hyperparameter_preset: str = "default"  # default, fast, thorough, custom
    custom_hyperparameters: Optional[Dict[str, List[Any]]] = None

    # Pipeline options
    generate_assets: bool = True
    generate_figures: bool = True
    generate_tables: bool = True
    skip_validation: bool = False
    require_confirmation: bool = True

    # Advanced options
    seed: int = 42
    n_jobs: int = -1

    # Safety options
    force_authoritative: bool = True  # Always use authoritative data
    validate_inheritance: bool = True  # Validate weight inheritance

    def __post_init__(self):
        """Validate configuration and set hyperparameter grid."""
        if self.hyperparameter_preset == "fast":
            self.custom_hyperparameters = {
                "C": [1.0],
                "solver": ["lbfgs"],
                "max_iter": [200],
                "tol": [1e-4],
            }
        elif self.hyperparameter_preset == "thorough":
            self.custom_hyperparameters = {
                "C": [0.001, 0.01, 0.1, 1.0, 10, 100],
                "solver": ["lbfgs", "newton-cg"],
                "max_iter": [200, 500, 1000],
                "tol": [1e-4, 1e-5],
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
            else:
                self.custom_hyperparameters = {
                    "C": [0.01, 1, 100],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                    "tol": [1e-4],
                }
        # custom preset uses provided custom_hyperparameters


class ValidationError(Exception):
    """Raised when validation checks fail."""

    pass


class PipelineRunner:
    """Comprehensive POLR pipeline runner with validation and safety checks."""

    def __init__(self, config: ComprehensiveConfig):
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
            / f"polr_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            level="DEBUG",
            rotation="100 MB",
            retention="7 days",
        )

    def validate_config(self):
        """Validate configuration and paths."""
        logger.info("🔍 Validating configuration...")

        # Check authoritative data directory
        kfold_path = Path(self.config.kfold_dir)
        if not kfold_path.exists():
            raise ValidationError(
                f"Authoritative data directory not found: {kfold_path}"
            )

        # Check required files
        required_files = ["per_fold_metadata.json", "fold_statistics.json"]

        for file_name in required_files:
            file_path = kfold_path / file_name
            if not file_path.exists():
                raise ValidationError(f"Required metadata file not found: {file_path}")

        # Check fold directories
        required_folds = ["fold_0", "fold_1", "fold_2", "fold_3", "oof_test"]
        for fold_name in required_folds:
            fold_path = kfold_path / fold_name
            if not fold_path.exists():
                raise ValidationError(f"Required fold directory not found: {fold_path}")

        logger.info("✅ Configuration validation passed")

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and weight inheritance."""
        logger.info("🔍 Validating data integrity and weight inheritance...")

        kfold_path = Path(self.config.kfold_dir)
        validation_results = {
            "metadata_valid": False,
            "weights_inherited": False,
            "labels_inherited": False,
            "fold_consistency": False,
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

            # Check binning metadata
            if "binning" in metadata:
                binning = metadata["binning"]
                if "fold_edges" in binning:
                    fold_edges = binning["fold_edges"]
                    expected_folds = ["fold_0", "fold_1", "fold_2", "fold_3"]

                    for fold in expected_folds:
                        if fold not in fold_edges:
                            validation_results["issues"].append(
                                f"Missing fold edges for {fold}"
                            )
                        else:
                            edges = fold_edges[fold]
                            if not isinstance(edges, list) or len(edges) != 2:
                                validation_results["issues"].append(
                                    f"Invalid edges for {fold}: {edges}"
                                )

                    if not validation_results["issues"]:
                        validation_results["labels_inherited"] = True

            # Check weights metadata
            if "weights" in metadata:
                weights = metadata["weights"]
                expected_folds = ["fold_0", "fold_1", "fold_2", "fold_3"]

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

                if not any(
                    "Missing weights" in issue or "Missing class_weights" in issue
                    for issue in validation_results["issues"]
                ):
                    validation_results["weights_inherited"] = True

            validation_results["metadata_valid"] = not validation_results["issues"]

            # Check sample data for precomputed fields
            logger.info("🔍 Checking sample data for precomputed fields...")
            sample_checks = []

            for fold_name in ["fold_0", "fold_1", "fold_2", "fold_3"]:
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

                        if missing_fields:
                            sample_checks.append(
                                f"{fold_name}: missing {missing_fields}"
                            )
                        else:
                            sample_checks.append(
                                f"{fold_name}: ✅ all required fields present"
                            )

            if sample_checks:
                logger.info("Sample data checks:")
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
                            logger.info(
                                "✅ OOF test set has precomputed outcome_bin labels"
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
                    "✅ Data integrity validation passed - all weights and labels properly inherited"
                )
            else:
                logger.error("❌ Data integrity validation failed:")
                for issue in validation_results["issues"]:
                    logger.error(f"  - {issue}")

        except Exception as e:
            logger.error(f"❌ Data integrity validation failed with exception: {e}")
            validation_results["issues"].append(f"Validation exception: {str(e)}")

        return validation_results

    def get_user_confirmation(self, message: str) -> bool:
        """Get user confirmation for critical operations."""
        if not self.config.require_confirmation:
            return True

        logger.info(f"🤔 {message}")
        response = input("Continue? [y/N]: ").lower().strip()
        return response in ["y", "yes"]

    def create_polar_config(self) -> POLARConfig:
        """Create POLAR configuration from comprehensive config."""
        return POLARConfig(
            kfold_dir=self.config.kfold_dir,
            output_dir=self.config.output_dir,
            model_type=self.config.model_type,
            hyperparameter_grid=self.config.custom_hyperparameters,
            seed=self.config.seed,
            n_jobs=self.config.n_jobs,
        )

    def run_cv_mode(self) -> Dict[str, Any]:
        """Run cross-validation only mode."""
        logger.info("🔄 Starting CV-only mode...")

        if not self.get_user_confirmation(
            "This will run hyperparameter search using folds 0, 1, 2"
        ):
            logger.info("❌ CV mode cancelled by user")
            return {}

        polar_config = self.create_polar_config()

        logger.info("🚀 Starting cross-validation...")
        cv_results = train_polar_cv(polar_config)

        logger.info("✅ CV-only mode completed")
        return cv_results

    def run_final_mode(self) -> Dict[str, Any]:
        """Run final model training only mode."""
        logger.info("🎯 Starting final-only mode...")

        # Check for existing CV results
        cv_results_path = Path(self.config.output_dir) / "cv_results.json"
        if not cv_results_path.exists():
            raise ValidationError(
                f"CV results not found at {cv_results_path}. Run CV first or use 'full' mode."
            )

        with open(cv_results_path, "r") as f:
            cv_results = json.load(f)

        if not self.get_user_confirmation(
            "This will train final model on fold 3 and evaluate on OOF test"
        ):
            logger.info("❌ Final mode cancelled by user")
            return {}

        polar_config = self.create_polar_config()

        logger.info("🚀 Starting final model training...")
        final_results = train_final_polar_model(polar_config, cv_results)

        logger.info("✅ Final-only mode completed")
        return final_results

    def run_full_mode(self) -> Dict[str, Any]:
        """Run complete pipeline mode."""
        logger.info("🎭 Starting full pipeline mode...")

        if not self.get_user_confirmation(
            "This will run complete pipeline: CV + final training + evaluation"
        ):
            logger.info("❌ Full mode cancelled by user")
            return {}

        polar_config = self.create_polar_config()

        # Run CV
        logger.info("🚀 Phase 1: Cross-validation...")
        cv_results = train_polar_cv(polar_config)

        # Run final training
        logger.info("🚀 Phase 2: Final model training...")
        final_results = train_final_polar_model(polar_config, cv_results)

        logger.info("✅ Full pipeline completed")
        return {"cv_results": cv_results, "final_results": final_results}

    def run_validate_mode(self) -> Dict[str, Any]:
        """Run validation-only mode."""
        logger.info("🔍 Starting validation-only mode...")

        validation_results = self.validate_data_integrity()

        # Print detailed validation report
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION REPORT")
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

        if validation_results["issues"]:
            logger.info("\nIssues found:")
            for issue in validation_results["issues"]:
                logger.warning(f"  - {issue}")
        else:
            logger.info("\n🎉 All validation checks passed!")

        logger.info("=" * 80)

        return validation_results

    def run_assets_mode(self) -> Dict[str, Any]:
        """Generate paper assets from existing results."""
        logger.info("📊 Starting assets generation mode...")

        output_dir = Path(self.config.output_dir)

        # Check for required files
        required_files = {
            "cv_results": output_dir / "cv_results.json",
            "oof_predictions": output_dir / "final_oof_predictions.jsonl",
        }

        missing_files = []
        for file_type, file_path in required_files.items():
            if not file_path.exists():
                missing_files.append(f"{file_type}: {file_path}")

        if missing_files:
            logger.error("❌ Required files missing for asset generation:")
            for missing in missing_files:
                logger.error(f"  - {missing}")
            return {"error": "missing_files", "missing": missing_files}

        if not self.get_user_confirmation(
            "This will generate paper assets (tables, figures) from existing results"
        ):
            logger.info("❌ Assets mode cancelled by user")
            return {}

        results = {}
        assets_dir = output_dir / "paper_assets"
        assets_dir.mkdir(exist_ok=True)

        try:
            if self.config.generate_tables:
                logger.info("📊 Generating LaTeX tables...")
                results["tables"] = self.generate_tables(
                    required_files["oof_predictions"],
                    required_files["cv_results"],
                    assets_dir,
                )

            if self.config.generate_figures:
                logger.info("📈 Generating figures...")
                results["figures"] = self.generate_figures(
                    required_files["oof_predictions"],
                    required_files["cv_results"],
                    assets_dir,
                )

            logger.info("✅ Assets generation completed")

        except Exception as e:
            logger.error(f"❌ Asset generation failed: {e}")
            results["error"] = str(e)

        return results

    def generate_tables(
        self, oof_path: Path, cv_path: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """Generate LaTeX tables."""
        import subprocess

        cmd = [
            sys.executable,
            "scripts/make_paper_tables.py",
            "--oof",
            str(oof_path),
            "--cv",
            str(cv_path),
            "--out",
            str(output_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0:
            logger.info("✅ Tables generated successfully")
            return {"status": "success", "output": result.stdout}
        else:
            logger.error(f"❌ Table generation failed: {result.stderr}")
            return {"status": "failed", "error": result.stderr}

    def generate_figures(
        self, oof_path: Path, cv_path: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """Generate paper figures."""
        import subprocess

        cmd = [
            sys.executable,
            "scripts/make_paper_figures.py",
            "--oof",
            str(oof_path),
            "--cv",
            str(cv_path),
            "--out",
            str(output_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0:
            logger.info("✅ Figures generated successfully")
            return {"status": "success", "output": result.stdout}
        else:
            logger.error(f"❌ Figure generation failed: {result.stderr}")
            return {"status": "failed", "error": result.stderr}

    def run(self) -> Dict[str, Any]:
        """Run the pipeline in the specified mode."""
        logger.info("🚀 Starting POLR Comprehensive Pipeline Runner")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.config.mode}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Data directory: {self.config.kfold_dir}")
        logger.info(f"Hyperparameter preset: {self.config.hyperparameter_preset}")
        logger.info("=" * 80)

        # Always validate data integrity unless skipped
        if not self.config.skip_validation:
            validation_results = self.validate_data_integrity()
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

                # Generate assets if requested
                if self.config.generate_assets and results:
                    logger.info("🎨 Generating additional paper assets...")
                    asset_results = self.run_assets_mode()
                    results["assets"] = asset_results

            elif self.config.mode == "validate":
                results = self.run_validate_mode()
            elif self.config.mode == "assets":
                results = self.run_assets_mode()
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

            logger.info("\n🎉 Pipeline completed successfully!")
            return {"status": "success", "results": results}

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive POLR Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default settings (POLR)
  python scripts/run_polr_comprehensive.py --mode full

  # Test multinomial LR with enhanced features
  python scripts/run_polr_comprehensive.py --mode full --model mlr --output-dir runs/mlr_enhanced

  # CV only with thorough hyperparameter search
  python scripts/run_polr_comprehensive.py --mode cv-only --hyperparameters thorough

  # Validate data integrity
  python scripts/run_polr_comprehensive.py --mode validate

  # Generate assets from existing results
  python scripts/run_polr_comprehensive.py --mode assets --output-dir runs/existing_run

  # Fast development run with MLR
  python scripts/run_polr_comprehensive.py --mode full --model mlr --hyperparameters fast --no-confirmation
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["cv-only", "final-only", "full", "validate", "assets"],
        default="full",
        help="Pipeline mode to run (default: full)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["polr", "mlr"],
        default="polr",
        help="Model type: polr (Proportional Odds LR) or mlr (Multinomial LR) (default: polr)",
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        default="data/final_stratified_kfold_splits_authoritative",
        help="Authoritative data directory (default: data/final_stratified_kfold_splits_authoritative)",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: runs/polr_comprehensive_TIMESTAMP)",
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

    # Asset generation
    parser.add_argument(
        "--generate-assets",
        action="store_true",
        default=True,
        help="Generate paper assets after training (default: True)",
    )

    parser.add_argument(
        "--no-assets", action="store_true", help="Skip asset generation"
    )

    parser.add_argument(
        "--no-figures", action="store_true", help="Skip figure generation"
    )

    parser.add_argument(
        "--no-tables", action="store_true", help="Skip table generation"
    )

    # Safety and validation
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip data integrity validation"
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
        args.output_dir = f"runs/polr_comprehensive_{timestamp}"

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
    config = ComprehensiveConfig(
        kfold_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        model_type=args.model,
        hyperparameter_preset=args.hyperparameters,
        custom_hyperparameters=custom_hyperparameters,
        generate_assets=args.generate_assets and not args.no_assets,
        generate_figures=not args.no_figures,
        generate_tables=not args.no_tables,
        skip_validation=args.skip_validation,
        require_confirmation=not args.no_confirmation,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    # Run pipeline
    runner = PipelineRunner(config)
    results = runner.run()

    # Print final results
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Status: {results['status']}")
    if results["status"] == "success":
        print(f"Output directory: {config.output_dir}")
        print("✅ Pipeline completed successfully!")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
