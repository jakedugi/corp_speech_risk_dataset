#!/usr/bin/env python3
"""
run_fast_optimization.py

Quick script to run optimized Bayesian optimization in fast mode.
Anchored around the best parameters found previously.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Run fast optimization with best parameters as anchor points."""

    # Set default paths relative to project structure
    default_gold_standard = "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    default_extracted_data = "data/extracted"

    parser = argparse.ArgumentParser(description="Fast Bayesian optimization")
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=30,
        help="Number of evaluations (default: 30)",
    )
    parser.add_argument(
        "--gold-standard", default=default_gold_standard, help="Gold standard CSV path"
    )
    parser.add_argument(
        "--extracted-data", default=default_extracted_data, help="Extracted data root"
    )
    parser.add_argument(
        "--output", default="fast_optimization_results.json", help="Output file"
    )

    args = parser.parse_args()

    print("🚀 Starting FAST Bayesian optimization...")
    print(f"   Max evaluations: {args.max_evaluations}")
    print(f"   Gold standard: {args.gold_standard}")
    print(f"   Extracted data: {args.extracted_data}")
    print(f"   Output: {args.output}")

    # Check if paths exist
    if not Path(args.gold_standard).exists():
        print(f"❌ Gold standard not found: {args.gold_standard}")
        sys.exit(1)

    if not Path(args.extracted_data).exists():
        print(f"❌ Extracted data not found: {args.extracted_data}")
        sys.exit(1)

    try:
        from bayesian_optimizer import BayesianOptimizer

        # Create optimizer with fast mode enabled
        optimizer = BayesianOptimizer(
            gold_standard_path=args.gold_standard,
            extracted_data_root=args.extracted_data,
            max_evaluations=args.max_evaluations,
            random_state=42,
            fast_mode=True,  # Enable fast mode for speed
        )

        # Run optimization
        result = optimizer.optimize()

        # Print results
        optimizer.print_optimization_report(result)

        # Save results
        optimizer.save_results(result, args.output)

        print(
            f"\n✅ Fast optimization complete! Best MSE Loss: {result.best_score:.2e}"
        )

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure scikit-optimize is installed: pip install scikit-optimize")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
