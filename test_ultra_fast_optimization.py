#!/usr/bin/env python3
"""
Test script for ultra-fast optimization modes.
Demonstrates 10x+ speedup using pre-computed feature matrix and caching.
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Set up optimal multiprocessing for macOS
if __name__ == "__main__":
    # 🚀 OPTIMIZATION: Use optimal multiprocessing method for macOS
    try:
        mp.set_start_method("forkserver")
    except RuntimeError:
        pass  # Already set

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.corp_speech_risk_dataset.case_outcome.bayesian_optimizer import (
    BayesianOptimizer,
)


def run_optimization_comparison():
    """Compare different optimization modes for performance."""

    gold_standard = "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    extracted_data = "data/extracted/courtlistener"

    if not Path(gold_standard).exists():
        print(f"❌ Gold standard file not found: {gold_standard}")
        return

    if not Path(extracted_data).exists():
        print(f"❌ Extracted data directory not found: {extracted_data}")
        return

    modes = [
        ("🐌 Normal Mode", False, False),
        ("⚡ Fast Mode", True, False),
        ("🚀 Ultra-Fast Mode", True, True),
    ]

    results = {}

    for mode_name, fast_mode, ultra_fast_mode in modes:
        print(f"\n{'='*60}")
        print(f"{mode_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            optimizer = BayesianOptimizer(
                gold_standard_path=gold_standard,
                extracted_data_root=extracted_data,
                max_evaluations=5,  # Small number for testing
                random_state=42,
                fast_mode=fast_mode,
                ultra_fast_mode=ultra_fast_mode,
            )

            result = optimizer.optimize()

            end_time = time.time()
            duration = end_time - start_time

            results[mode_name] = {
                "duration": duration,
                "best_score": result.best_score,
                "evaluations": result.total_evaluations,
                "speed_per_eval": duration / max(result.total_evaluations, 1),
            }

            print(f"\n📊 Results:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Best MSE: {result.best_score:.2e}")
            print(f"  Evaluations: {result.total_evaluations}")
            print(
                f"  Speed per eval: {duration / max(result.total_evaluations, 1):.3f}s"
            )

        except Exception as e:
            print(f"❌ Error in {mode_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison
    print(f"\n{'='*60}")
    print("🏁 PERFORMANCE COMPARISON")
    print(f"{'='*60}")

    if len(results) > 1:
        baseline_time = None
        for mode_name, data in results.items():
            duration = data["duration"]
            if baseline_time is None:
                baseline_time = duration
                speedup = 1.0
            else:
                speedup = baseline_time / duration

            print(f"{mode_name}:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Speedup: {speedup:.1f}x")
            print(f"  Speed per eval: {data['speed_per_eval']:.3f}s")
            print(f"  Best MSE: {data['best_score']:.2e}")
            print()


def run_ultra_fast_test():
    """Run a quick test of ultra-fast mode only."""
    print("🚀 Running Ultra-Fast Mode Test")
    print("=" * 50)

    gold_standard = "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    extracted_data = "data/extracted/courtlistener"

    start_time = time.time()

    optimizer = BayesianOptimizer(
        gold_standard_path=gold_standard,
        extracted_data_root=extracted_data,
        max_evaluations=10,
        random_state=42,
        fast_mode=True,
        ultra_fast_mode=True,
    )

    result = optimizer.optimize()

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n🎯 Ultra-Fast Results:")
    print(f"  Total Duration: {duration:.2f}s")
    print(f"  Evaluations: {result.total_evaluations}")
    print(f"  Speed per eval: {duration / max(result.total_evaluations, 1):.3f}s")
    print(f"  Best MSE: {result.best_score:.2e}")

    # Save results
    optimizer.save_results(result, "ultra_fast_test_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ultra-fast optimization")
    parser.add_argument(
        "--mode",
        choices=["compare", "ultra-fast"],
        default="ultra-fast",
        help="Test mode: compare all modes or just ultra-fast",
    )

    args = parser.parse_args()

    if args.mode == "compare":
        run_optimization_comparison()
    else:
        run_ultra_fast_test()
