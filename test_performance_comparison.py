#!/usr/bin/env python3
"""
Performance comparison test between different optimization modes.
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Set up optimal multiprocessing for macOS
if __name__ == "__main__":
    try:
        mp.set_start_method("forkserver")
    except RuntimeError:
        pass

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.corp_speech_risk_dataset.case_outcome.bayesian_optimizer import (
    BayesianOptimizer,
)


def test_performance():
    """Test performance of different modes."""

    gold_standard = "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    extracted_data = "data/extracted/courtlistener"

    print("🚀 ULTRA-FAST OPTIMIZATION PERFORMANCE TEST")
    print("=" * 60)

    # Test ultra-fast mode with different evaluation counts
    evaluation_counts = [10, 20, 50]

    for num_evals in evaluation_counts:
        print(f"\n⚡ Testing {num_evals} evaluations...")

        start_time = time.time()

        optimizer = BayesianOptimizer(
            gold_standard_path=gold_standard,
            extracted_data_root=extracted_data,
            max_evaluations=num_evals,
            random_state=42,
            fast_mode=True,
            ultra_fast_mode=True,
        )

        result = optimizer.optimize()

        end_time = time.time()
        duration = end_time - start_time

        print(
            f"  ✅ Completed {result.total_evaluations} evaluations in {duration:.3f}s"
        )
        print(
            f"  📊 Speed: {duration / max(result.total_evaluations, 1):.4f}s per evaluation"
        )
        print(f"  🎯 Best MSE: {result.best_score:.2e}")

        # Calculate theoretical throughput
        evals_per_second = result.total_evaluations / duration
        print(f"  🔥 Throughput: {evals_per_second:.1f} evaluations/second")

        # Estimated time for 100 evaluations
        est_time_100 = 100 / evals_per_second
        print(f"  📈 Est. time for 100 evals: {est_time_100:.2f}s")


def run_comparison_with_baseline():
    """Compare ultra-fast vs fast mode performance."""

    print("\n" + "=" * 60)
    print("🏁 PERFORMANCE COMPARISON: Fast vs Ultra-Fast")
    print("=" * 60)

    gold_standard = "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    extracted_data = "data/extracted/courtlistener"

    modes = [
        ("⚡ Fast Mode", True, False),
        ("🚀 Ultra-Fast Mode", True, True),
    ]

    results = {}

    for mode_name, fast_mode, ultra_fast_mode in modes:
        print(f"\n{mode_name}:")

        start_time = time.time()

        optimizer = BayesianOptimizer(
            gold_standard_path=gold_standard,
            extracted_data_root=extracted_data,
            max_evaluations=10,  # Minimum required
            random_state=42,
            fast_mode=fast_mode,
            ultra_fast_mode=ultra_fast_mode,
        )

        result = optimizer.optimize()

        end_time = time.time()
        duration = end_time - start_time

        results[mode_name] = duration

        print(f"  Duration: {duration:.3f}s")
        print(f"  Speed per eval: {duration / max(result.total_evaluations, 1):.4f}s")
        print(f"  Best MSE: {result.best_score:.2e}")

    # Calculate speedup
    if len(results) == 2:
        fast_time = results["⚡ Fast Mode"]
        ultra_fast_time = results["🚀 Ultra-Fast Mode"]
        speedup = fast_time / ultra_fast_time

        print(f"\n🏆 RESULTS:")
        print(f"  Speedup: {speedup:.1f}x faster")
        print(
            f"  Time reduction: {((fast_time - ultra_fast_time) / fast_time * 100):.1f}% faster"
        )


if __name__ == "__main__":
    test_performance()
    run_comparison_with_baseline()
