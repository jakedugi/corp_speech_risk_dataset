#!/usr/bin/env python3
"""
Standalone script to run Bayesian optimization for case outcome imputer.
This script is independent of grid search and handles dismissals properly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Run Bayesian optimization with all hyperparameters."""
    print("🚀 Starting Comprehensive Bayesian Hyperparameter Optimization")
    print("=" * 80)
    print("📊 Optimizing 40+ hyperparameters including:")
    print("   • Core extraction parameters (4)")
    print("   • Position thresholds (2)")
    print("   • Case flag thresholds (4)")
    print("   • All VotingWeights parameters (30+)")
    print("   • High/Low signal pattern weights (5)")
    print("=" * 80)

    try:
        from corp_speech_risk_dataset.case_outcome.bayesian_optimizer import (
            BayesianOptimizer,
        )

        print("✅ Bayesian optimizer imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure scikit-optimize is installed: uv pip install scikit-optimize")
        return 1

    # Configuration
    config = {
        "gold_standard": "data/gold_standard/case_outcome_amounts_hand_annotated.csv",
        "extracted_data": "data/extracted/courtlistener",
        "max_evaluations": 100,  # Start with 100 for comprehensive search
        "random_state": 42,
        "fast_mode": True,  # Enable for faster execution
        "output": "bayesian_optimization_results_comprehensive.json",
    }

    print(f"📂 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    # Verify data paths exist
    gold_standard_path = Path(config["gold_standard"])
    extracted_data_path = Path(config["extracted_data"])

    if not gold_standard_path.exists():
        print(f"❌ Gold standard file not found: {gold_standard_path}")
        return 1

    if not extracted_data_path.exists():
        print(f"❌ Extracted data directory not found: {extracted_data_path}")
        return 1

    print(f"✅ Data paths verified")

    # Create optimizer
    print(f"🔧 Creating Bayesian optimizer...")
    optimizer = BayesianOptimizer(
        gold_standard_path=config["gold_standard"],
        extracted_data_root=config["extracted_data"],
        max_evaluations=config["max_evaluations"],
        random_state=config["random_state"],
        fast_mode=config["fast_mode"],
    )

    print(f"✅ Optimizer created with {len(optimizer.search_space)} hyperparameters")
    print(f"🎯 Starting optimization with {config['max_evaluations']} evaluations...")
    print()

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print("\n" + "=" * 80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 80)

    optimizer.print_optimization_report(result)

    # Save results
    if config["output"]:
        optimizer.save_results(result, config["output"])
        print(f"💾 Results saved to {config['output']}")

    print(f"\n🏆 Best MSE Loss: {result.best_score:.2e}")
    print(f"🎯 Total Evaluations: {result.total_evaluations}")
    print("✅ Bayesian optimization completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
