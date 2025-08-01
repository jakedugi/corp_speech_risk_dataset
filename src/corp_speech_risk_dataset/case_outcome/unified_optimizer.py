#!/usr/bin/env python3
"""
unified_optimizer.py

Unified optimization system that combines all optimization approaches:
- Bayesian optimization with scikit-optimize
- Grid search optimization with parallel processing
- Monitoring and logging capabilities
- Fast mode optimization
- All high/low signal pattern features

This script provides a single entry point for all optimization needs while preserving
all existing features, logging, and monitoring capabilities.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Setup project imports
try:
    from corp_speech_risk_dataset.case_outcome.bayesian_optimizer import (
        BayesianOptimizer,
        BayesianOptimizationResult,
    )
    from corp_speech_risk_dataset.case_outcome.grid_search_optimizer import (
        GridSearchOptimizer,
        OptimizationResult,
    )
    from corp_speech_risk_dataset.case_outcome.monitor_optimization import (
        monitor_optimization_log,
    )

    BAYESIAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Some optimization modules not available: {e}")
    try:
        # Try fallback local imports
        from .bayesian_optimizer import BayesianOptimizer, BayesianOptimizationResult
        from .grid_search_optimizer import GridSearchOptimizer, OptimizationResult
        from .monitor_optimization import monitor_optimization_log

        BAYESIAN_AVAILABLE = True
    except ImportError:
        # Create dummy classes for type hints
        class BayesianOptimizationResult:
            pass

        class OptimizationResult:
            pass

        def monitor_optimization_log(*args, **kwargs):
            pass

        BAYESIAN_AVAILABLE = False


class UnifiedOptimizer:
    """
    Unified optimization system that orchestrates all optimization approaches.
    Preserves all existing functionality while providing a clean interface.
    """

    def __init__(
        self,
        gold_standard_path: str,
        extracted_data_root: str,
        optimization_type: str = "bayesian",
        max_evaluations: int = 100,
        max_workers: int = 4,
        fast_mode: bool = False,
        random_state: int = 42,
        output_dir: str = "optimization_results",
    ):
        """
        Initialize the unified optimizer.

        Args:
            gold_standard_path: Path to gold standard CSV
            extracted_data_root: Root directory containing extracted case data
            optimization_type: Type of optimization ("bayesian", "grid", "fast")
            max_evaluations: Maximum number of evaluations
            max_workers: Number of parallel workers (for grid search)
            fast_mode: Enable fast mode for reduced logging/I/O
            random_state: Random seed for reproducibility
            output_dir: Directory to save results
        """
        self.gold_standard_path = Path(gold_standard_path)
        self.extracted_data_root = Path(extracted_data_root)
        self.optimization_type = optimization_type
        self.max_evaluations = max_evaluations
        self.max_workers = max_workers
        self.fast_mode = fast_mode
        self.random_state = random_state
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Validate inputs
        self._validate_inputs()

        self.logger.info(f"🚀 Unified Optimizer initialized")
        self.logger.info(f"   Type: {optimization_type}")
        self.logger.info(f"   Max evaluations: {max_evaluations}")
        self.logger.info(f"   Fast mode: {fast_mode}")
        self.logger.info(f"   Output directory: {output_dir}")

    def _setup_logging(self):
        """Setup comprehensive logging system."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Generate timestamp for unique log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"unified_optimization_{timestamp}.log"

        # Configure logging
        log_level = logging.INFO if not self.fast_mode else logging.WARNING

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)
        self.log_file = log_file

        self.logger.info(f"📝 Logging to: {log_file}")

    def _validate_inputs(self):
        """Validate input paths and parameters."""
        if not self.gold_standard_path.exists():
            raise FileNotFoundError(
                f"Gold standard not found: {self.gold_standard_path}"
            )

        if not self.extracted_data_root.exists():
            raise FileNotFoundError(
                f"Extracted data root not found: {self.extracted_data_root}"
            )

        if self.optimization_type == "bayesian" and not BAYESIAN_AVAILABLE:
            raise ImportError(
                "Bayesian optimization requires scikit-optimize. Install with: pip install scikit-optimize"
            )

        self.logger.info(f"✅ Gold standard: {self.gold_standard_path}")
        self.logger.info(f"✅ Extracted data: {self.extracted_data_root}")

    def run_bayesian_optimization(self) -> BayesianOptimizationResult:
        """Run Bayesian optimization with all features."""
        self.logger.info("🎯 Starting Bayesian hyperparameter optimization")

        optimizer = BayesianOptimizer(
            gold_standard_path=str(self.gold_standard_path),
            extracted_data_root=str(self.extracted_data_root),
            max_evaluations=self.max_evaluations,
            random_state=self.random_state,
            fast_mode=self.fast_mode,
        )

        # Run optimization
        result = optimizer.optimize()

        # Generate comprehensive reports
        optimizer.print_optimization_report(result)

        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"bayesian_results_{timestamp}.json"
        optimizer.save_results(result, str(output_file))

        # Save best parameters summary
        self._save_best_parameters_summary(
            optimizer.get_current_best_parameters(), "bayesian"
        )

        self.logger.info(
            f"✅ Bayesian optimization complete! Best MSE: {result.best_score:.2e}"
        )
        return result

    def run_grid_search_optimization(self) -> list[OptimizationResult]:
        """Run grid search optimization with all features."""
        self.logger.info("🎯 Starting grid search hyperparameter optimization")

        optimizer = GridSearchOptimizer(
            gold_standard_path=str(self.gold_standard_path),
            extracted_data_root=str(self.extracted_data_root),
        )

        # Run optimization
        results = optimizer.optimize(
            max_workers=self.max_workers, max_combinations=self.max_evaluations
        )

        # Print results
        optimizer.print_best_results(results, top_k=10)

        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"grid_search_results_{timestamp}.json"
        optimizer.save_results(results, str(output_file))

        # Save best parameters summary
        best_result = results[0] if results else None
        if best_result:
            best_params = {
                "hyperparams": best_result.hyperparams,
                "mse_loss": best_result.mse_loss,
                "f1_score": best_result.f1_score,
                "precision": best_result.precision,
                "recall": best_result.recall,
                "exact_matches": best_result.exact_matches,
                "total_cases": best_result.total_cases,
            }
            self._save_best_parameters_summary(best_params, "grid_search")

        self.logger.info(
            f"✅ Grid search complete! Best MSE: {results[0].mse_loss:.2e}"
        )
        return results

    def run_fast_optimization(self) -> BayesianOptimizationResult:
        """Run fast Bayesian optimization with reduced evaluations."""
        self.logger.info("⚡ Starting fast Bayesian optimization")

        # Override for fast mode
        fast_evaluations = min(self.max_evaluations, 30)

        optimizer = BayesianOptimizer(
            gold_standard_path=str(self.gold_standard_path),
            extracted_data_root=str(self.extracted_data_root),
            max_evaluations=fast_evaluations,
            random_state=self.random_state,
            fast_mode=True,  # Force fast mode
        )

        # Run optimization
        result = optimizer.optimize()

        # Generate reports
        optimizer.print_best_parameters_summary()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"fast_bayesian_results_{timestamp}.json"
        optimizer.save_results(result, str(output_file))

        # Save best parameters summary
        self._save_best_parameters_summary(
            optimizer.get_current_best_parameters(), "fast_bayesian"
        )

        self.logger.info(
            f"⚡ Fast optimization complete! Best MSE: {result.best_score:.2e}"
        )
        return result

    def _save_best_parameters_summary(
        self, best_params: Dict[str, Any], optimization_type: str
    ):
        """Save a summary of the best parameters found."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = (
            self.output_dir / f"best_parameters_{optimization_type}_{timestamp}.json"
        )

        with open(summary_file, "w") as f:
            json.dump(best_params, f, indent=2, default=str)

        self.logger.info(f"💾 Best parameters saved to: {summary_file}")

    def run_optimization(self) -> Any:
        """
        Run the specified optimization type.

        Returns:
            Optimization results (type depends on optimization method)
        """
        start_time = datetime.now()
        self.logger.info(
            f"🚀 Starting {self.optimization_type} optimization at {start_time}"
        )

        try:
            if self.optimization_type == "bayesian":
                result = self.run_bayesian_optimization()
            elif self.optimization_type == "grid":
                result = self.run_grid_search_optimization()
            elif self.optimization_type == "fast":
                result = self.run_fast_optimization()
            else:
                raise ValueError(f"Unknown optimization type: {self.optimization_type}")

            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info(f"🎉 Optimization completed successfully!")
            self.logger.info(f"   Duration: {duration}")
            self.logger.info(f"   Results saved to: {self.output_dir}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            raise

    def start_monitoring(self, check_interval: int = 30):
        """Start monitoring the optimization log file."""
        self.logger.info(f"👁️  Starting log monitoring (interval: {check_interval}s)")
        monitor_optimization_log(str(self.log_file), check_interval)


def create_unified_argument_parser() -> argparse.ArgumentParser:
    """Create a comprehensive argument parser for all optimization options."""
    parser = argparse.ArgumentParser(
        description="Unified optimization system for case outcome imputer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Bayesian optimization (recommended)
  python unified_optimizer.py --type bayesian --max-evaluations 100

  # Fast Bayesian optimization for quick testing
  python unified_optimizer.py --type fast --max-evaluations 30

  # Grid search optimization
  python unified_optimizer.py --type grid --max-evaluations 50 --max-workers 4

  # With custom data paths
  python unified_optimizer.py --type bayesian \\
    --gold-standard /path/to/gold_standard.csv \\
    --extracted-data /path/to/extracted \\
    --max-evaluations 200
        """,
    )

    # Core optimization settings
    parser.add_argument(
        "--type",
        choices=["bayesian", "grid", "fast"],
        default="bayesian",
        help="Type of optimization to run (default: bayesian)",
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=100,
        help="Maximum number of evaluations (default: 100)",
    )

    # Data paths
    parser.add_argument(
        "--gold-standard",
        default="data/gold_standard/case_outcome_amounts_hand_annotated.csv",
        help="Path to gold standard CSV file",
    )

    parser.add_argument(
        "--extracted-data",
        default="data/extracted",
        help="Root directory containing extracted case data",
    )

    # Performance settings
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for grid search (default: 4)",
    )

    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast mode for reduced logging and I/O overhead",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        default="optimization_results",
        help="Directory to save optimization results (default: optimization_results)",
    )

    # Monitoring
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Start log monitoring after optimization begins",
    )

    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=30,
        help="Log monitoring check interval in seconds (default: 30)",
    )

    return parser


def main():
    """Main function that orchestrates the unified optimization system."""
    parser = create_unified_argument_parser()
    args = parser.parse_args()

    print("🔧 Unified Case Outcome Optimization System")
    print("=" * 60)
    print(f"Optimization Type: {args.type}")
    print(f"Max Evaluations: {args.max_evaluations}")
    print(f"Fast Mode: {args.fast_mode}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    try:
        # Initialize unified optimizer
        optimizer = UnifiedOptimizer(
            gold_standard_path=args.gold_standard,
            extracted_data_root=args.extracted_data,
            optimization_type=args.type,
            max_evaluations=args.max_evaluations,
            max_workers=args.max_workers,
            fast_mode=args.fast_mode,
            random_state=args.random_state,
            output_dir=args.output_dir,
        )

        # Start monitoring if requested
        if args.monitor:
            print("🔍 Log monitoring will start after optimization begins...")
            # Note: In practice, you'd want to run monitoring in a separate thread/process

        # Run optimization
        result = optimizer.run_optimization()

        print(f"\n🎉 Optimization Complete!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📝 Logs saved to: logs/")

        # Print command for running monitoring separately
        print(f"\n📊 To monitor logs in real-time, run:")
        print(
            f"python -m corp_speech_risk_dataset.case_outcome.monitor_optimization --log-file logs/unified_optimization_*.log"
        )

    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
