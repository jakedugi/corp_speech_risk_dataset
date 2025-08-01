#!/usr/bin/env python3
"""
Test script for parallel optimization modes.
Demonstrates parallel Bayesian optimization using Optuna and vectorized operations.
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


def test_parallel_modes():
    """Test different parallel optimization modes."""

    from src.corp_speech_risk_dataset.case_outcome.bayesian_optimizer import (
        BayesianOptimizer,
    )

    gold_standard = "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    extracted_data = "data/extracted/courtlistener"

    if not Path(gold_standard).exists():
        print(f"❌ Gold standard file not found: {gold_standard}")
        return

    if not Path(extracted_data).exists():
        print(f"❌ Extracted data directory not found: {extracted_data}")
        return

    print("🚀 PARALLEL OPTIMIZATION TEST")
    print("=" * 60)

    # Test different parallel configurations
    test_configs = [
        {
            "name": "🔥 Ultra-Fast + Parallel (4 jobs)",
            "ultra_fast_mode": True,
            "parallel_jobs": 4,
            "use_optuna": True,
            "evaluations": 20,
        },
        {
            "name": "⚡ Fast + Parallel (2 jobs)",
            "fast_mode": True,
            "ultra_fast_mode": False,
            "parallel_jobs": 2,
            "use_optuna": True,
            "evaluations": 10,
        },
        {
            "name": "🚀 Ultra-Fast Sequential",
            "ultra_fast_mode": True,
            "parallel_jobs": 1,
            "use_optuna": False,
            "evaluations": 10,
        },
    ]

    results = {}

    for config in test_configs:
        print(f"\n{config['name']}:")
        print("-" * 40)

        start_time = time.time()

        try:
            optimizer = BayesianOptimizer(
                gold_standard_path=gold_standard,
                extracted_data_root=extracted_data,
                max_evaluations=config["evaluations"],
                random_state=42,
                fast_mode=config.get("fast_mode", False),
                ultra_fast_mode=config.get("ultra_fast_mode", False),
                parallel_jobs=config.get("parallel_jobs", 1),
                use_optuna=config.get("use_optuna", False),
            )

            result = optimizer.optimize()

            end_time = time.time()
            duration = end_time - start_time

            results[config["name"]] = {
                "duration": duration,
                "best_score": result.best_score,
                "evaluations": result.total_evaluations,
                "speed_per_eval": duration / max(result.total_evaluations, 1),
                "parallel_jobs": config.get("parallel_jobs", 1),
            }

            print(
                f"  ✅ Completed {result.total_evaluations} evaluations in {duration:.2f}s"
            )
            print(
                f"  📊 Speed: {duration / max(result.total_evaluations, 1):.4f}s per evaluation"
            )
            print(f"  🎯 Best MSE: {result.best_score:.2e}")
            print(f"  🔧 Parallel jobs: {config.get('parallel_jobs', 1)}")

            # Calculate theoretical throughput
            evals_per_second = result.total_evaluations / duration
            print(f"  🔥 Throughput: {evals_per_second:.1f} evaluations/second")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison
    print(f"\n{'='*60}")
    print("🏁 PARALLEL OPTIMIZATION COMPARISON")
    print(f"{'='*60}")

    if results:
        # Sort by speed (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["speed_per_eval"])

        for i, (name, data) in enumerate(sorted_results):
            ranking = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."

            print(f"{ranking} {name}:")
            print(f"   Duration: {data['duration']:.2f}s")
            print(f"   Speed per eval: {data['speed_per_eval']:.4f}s")
            print(
                f"   Throughput: {data['evaluations'] / data['duration']:.1f} evals/sec"
            )
            print(
                f"   Parallel efficiency: {data['evaluations'] / data['duration'] / data['parallel_jobs']:.1f} evals/sec/job"
            )
            print(f"   Best MSE: {data['best_score']:.2e}")
            print()


def test_vectorized_processing():
    """Test vectorized text processing capabilities."""
    print("\n🔧 VECTORIZED PROCESSING TEST")
    print("=" * 50)

    try:
        from src.corp_speech_risk_dataset.case_outcome.fast_text_processing import (
            FastTextProcessor,
        )

        # Test data
        test_texts = [
            "The settlement amount of $5,000,000 was awarded to the plaintiff.",
            "Damages in the amount of $2.5 million were ordered by the court.",
            "The jury awarded $750,000 in compensation for the injuries.",
            "A penalty of $100,000 was assessed against the defendant.",
            "Settlement fund of $10 million was established for class members.",
        ] * 20  # 100 texts total

        processor = FastTextProcessor(fast_mode=True)

        # Test amount extraction
        start_time = time.time()
        amounts = processor.extract_amounts_vectorized(test_texts)
        end_time = time.time()

        total_amounts = sum(len(text_amounts) for text_amounts in amounts)

        print(f"  ✅ Processed {len(test_texts)} texts in {end_time - start_time:.4f}s")
        print(f"  📊 Found {total_amounts} amounts total")
        print(
            f"  🔥 Speed: {len(test_texts) / (end_time - start_time):.1f} texts/second"
        )

        # Test feature scoring
        contexts = [
            amount["context"] for text_amounts in amounts for amount in text_amounts
        ]
        if contexts:
            weights = {"financial_terms_weight": 1.5, "legal_proceedings_weight": 1.2}

            start_time = time.time()
            scores = processor.compute_feature_scores_vectorized(contexts, weights)
            end_time = time.time()

            print(
                f"  ✅ Computed scores for {len(contexts)} contexts in {end_time - start_time:.4f}s"
            )
            print(f"  📊 Average score: {scores.mean():.2f}")
            print(
                f"  🔥 Speed: {len(contexts) / (end_time - start_time):.1f} contexts/second"
            )

    except ImportError as e:
        print(f"  ❌ Vectorized processing not available: {e}")


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark of all optimization modes."""
    print("\n🏆 COMPREHENSIVE OPTIMIZATION BENCHMARK")
    print("=" * 70)

    # This would test with different problem sizes and configurations
    print("  This would run extensive benchmarks with:")
    print("  • Different evaluation counts (10, 50, 100)")
    print("  • Different parallel job counts (1, 2, 4, 8)")
    print("  • Ultra-fast vs fast vs normal modes")
    print("  • Optuna vs scikit-optimize")
    print("  • With and without vectorized processing")
    print("  • Memory usage profiling")
    print("  • CPU utilization analysis")
    print("\n  📝 Results would be saved to benchmark_results.json")


if __name__ == "__main__":
    test_parallel_modes()
    test_vectorized_processing()
    run_comprehensive_benchmark()
