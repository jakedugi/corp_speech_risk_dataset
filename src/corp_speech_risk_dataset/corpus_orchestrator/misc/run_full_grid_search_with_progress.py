#!/usr/bin/env python3
"""
Full Grid Search with Progress Tracking and ETA
"""

import subprocess
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path


def run_with_progress():
    """Run full grid search with progress tracking and ETA estimation."""

    print("🔍 FULL GRID SEARCH - COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    # Grid search parameters
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_validated_features_with_embeddings_OPTIMIZED.py",
        "--data-dir",
        "data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage",
        "--output-dir",
        "results/full_grid_search_final",
        "--fold",
        "4",
        "--grid-search",  # Enable full grid search
    ]

    print(f"📋 Command: {' '.join(cmd)}")
    print(f"🎯 Models: 13 variants")
    print(f"🔬 Grid sizes: 4-12 hyperparameter combinations per model")
    print(f"⚡ Parallel: n_jobs=6 for GridSearchCV")
    print(
        f"📊 Features: {', '.join(['feat_interact_hedge_x_guarantee', 'feat_new4_neutral_to_disclaimer_transition_rate', 'lex_disclaimers_present', 'legal_bert_emb'])}"
    )

    # Estimated timing based on current run
    model_times = {
        "3VF": {
            "L2": 4,
            "L1": 8,
            "ElasticNet": 10,
            "SVM": 5,
            "MLR": 12,
        },  # minutes per model
        "E": {"L2": 8, "L1": 15, "ElasticNet": 18},  # embeddings take longer
        "E+3": {"L2": 10, "L1": 18, "ElasticNet": 22},  # embeddings + scalars
    }

    grid_multiplier = 3.5  # Average grid size expansion for full search
    total_models = 13

    # Calculate ETA
    base_time = sum(
        [
            model_times["3VF"]["L2"]
            + model_times["3VF"]["L1"]
            + model_times["3VF"]["ElasticNet"]
            + model_times["3VF"]["SVM"]
            + model_times["3VF"]["MLR"] * 2,  # 2 MLR variants
            model_times["E"]["L2"]
            + model_times["E"]["L1"]
            + model_times["E"]["ElasticNet"],
            model_times["E+3"]["L2"]
            + model_times["E+3"]["L1"]
            + model_times["E+3"]["ElasticNet"],
            5,  # POLR
        ]
    )

    estimated_minutes = int(base_time * grid_multiplier)
    estimated_end = datetime.now() + timedelta(minutes=estimated_minutes)

    print(
        f"⏱️  Estimated time: {estimated_minutes} minutes ({estimated_minutes//60}h {estimated_minutes%60}m)"
    )
    print(f"🏁 Estimated completion: {estimated_end.strftime('%H:%M:%S')}")
    print("=" * 80)

    # Start timing
    start_time = time.time()

    # Run the command
    try:
        print("🚀 Starting full grid search...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        model_count = 0
        for line in iter(process.stdout.readline, ""):
            print(line, end="")

            # Track progress
            if "EVALUATING:" in line:
                model_count += 1
                elapsed = time.time() - start_time
                progress = model_count / total_models

                if model_count > 1:  # After first model, we can estimate
                    avg_time_per_model = elapsed / (model_count - 1)
                    remaining_models = total_models - model_count
                    eta_seconds = remaining_models * avg_time_per_model
                    eta_time = datetime.now() + timedelta(seconds=eta_seconds)

                    print(
                        f"\n📈 PROGRESS: {model_count}/{total_models} models ({progress*100:.1f}%)"
                    )
                    print(f"⏰ Elapsed: {elapsed//60:.0f}m {elapsed%60:.0f}s")
                    print(
                        f"🎯 ETA: {eta_time.strftime('%H:%M:%S')} ({eta_seconds//60:.0f}m remaining)"
                    )
                    print("-" * 40)

        process.wait()

        total_time = time.time() - start_time
        print(f"\n🎉 COMPLETED! Total time: {total_time//60:.0f}m {total_time%60:.0f}s")

        if process.returncode == 0:
            print("✅ Grid search completed successfully!")
            print(f"📂 Results saved to: results/full_grid_search_final/")
        else:
            print(f"❌ Grid search failed with exit code: {process.returncode}")

    except KeyboardInterrupt:
        print("\n⏹️  Grid search interrupted by user")
        process.terminate()
        return False

    return process.returncode == 0


if __name__ == "__main__":
    success = run_with_progress()
    sys.exit(0 if success else 1)
