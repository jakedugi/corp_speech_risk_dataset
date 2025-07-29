#!/usr/bin/env python3
"""
run_optimization.py

Wrapper script to run the grid search optimization with proper setup and error handling.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

try:
    from grid_search_optimizer import GridSearchOptimizer
except ImportError:
    GridSearchOptimizer = None


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(
                f'logs/optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )


def validate_paths(gold_standard_path: str, extracted_data_root: str):
    """Validate that required paths exist."""
    gold_standard = Path(gold_standard_path)
    extracted_root = Path(extracted_data_root)

    if not gold_standard.exists():
        raise FileNotFoundError(f"Gold standard file not found: {gold_standard_path}")

    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted data root not found: {extracted_data_root}")

    print(f"✅ Gold standard: {gold_standard}")
    print(f"✅ Extracted data root: {extracted_root}")


def main():
    """Main function with proper error handling."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for case outcome imputer"
    )
    parser.add_argument(
        "--gold-standard",
        default="../../../data/gold_standard/case_outcome_amounts_hand_annotated.csv",
        help="Path to gold standard CSV file",
    )
    parser.add_argument(
        "--extracted-data-root",
        default="../../../data/extracted",
        help="Root directory containing extracted case data",
    )
    parser.add_argument(
        "--output", default="optimization_results.json", help="Output file for results"
    )
    parser.add_argument(
        "--max-workers", type=int, default=2, help="Number of parallel workers"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full hyperparameter grid (slower but more comprehensive)",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Maximum number of combinations to evaluate (None = all)",
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Use Bayesian optimization instead of grid search",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast mode for optimization (reduces logging and I/O)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Validate paths
        validate_paths(args.gold_standard, args.extracted_data_root)

        # Initialize optimizer
        optimizer = GridSearchOptimizer(
            gold_standard_path=args.gold_standard,
            extracted_data_root=args.extracted_data_root,
        )

        # Use full grid if requested
        if args.full_grid:
            optimizer.search_grid = optimizer.hyperparameter_grid
            logger.info("Using full hyperparameter grid")
        else:
            logger.info("Using reduced hyperparameter grid for faster testing")

        # Run optimization
        if args.bayesian:
            logger.info("🎯 Starting Bayesian hyperparameter optimization...")
            try:
                from bayesian_optimizer import BayesianOptimizer

                bayesian_optimizer = BayesianOptimizer(
                    gold_standard_path=args.gold_standard,
                    extracted_data_root=args.extracted_data_root,
                    max_evaluations=args.max_combinations or 50,
                    random_state=42,
                    fast_mode=args.fast_mode,
                )

                result = bayesian_optimizer.optimize()
                bayesian_optimizer.print_optimization_report(result)
                bayesian_optimizer.save_results(result, args.output)

                logger.info(
                    f"✅ Bayesian optimization complete! Best MSE Loss: {result.best_score:.2e}"
                )

            except ImportError:
                logger.error(
                    "❌ Bayesian optimization requires scikit-optimize. Install with: pip install scikit-optimize"
                )
                sys.exit(1)

        else:
            logger.info("🎯 Starting grid search hyperparameter optimization...")
            results = optimizer.optimize(
                max_workers=args.max_workers, max_combinations=args.max_combinations
            )

            # Print best results (only for grid search)
            optimizer.print_best_results(results, top_k=10)

            # Save results
            optimizer.save_results(results, args.output)

            logger.info(
                f"✅ Grid search complete! Best MSE Loss: {results[0].mse_loss:.2e}"
            )

            # Print summary statistics
            print(f"\n📊 Summary Statistics:")
            print(f"   Total combinations tested: {len(results)}")
            print(f"   Best MSE Loss: {results[0].mse_loss:.2e}")
            print(f"   Best Precision: {results[0].precision:.3f}")
            print(f"   Best Recall: {results[0].recall:.3f}")
            print(f"   Best F1 Score: {results[0].f1_score:.3f}")

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
