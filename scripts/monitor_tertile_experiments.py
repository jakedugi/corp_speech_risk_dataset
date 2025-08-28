#!/usr/bin/env python3
"""
Monitor the progress of tertile experiments.
"""

import time
import orjson
from pathlib import Path
from datetime import datetime
import sys


def check_results():
    """Check for completed experiments and display results."""
    # Find latest results directory
    results_dirs = list(Path("results").glob("tertile_comprehensive_*"))
    if not results_dirs:
        print("No tertile experiment results found yet.")
        return None

    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Monitoring: {latest_dir}")

    # Check feature validation
    feature_val = latest_dir / "feature_validation" / "validation_results_iter_1.json"
    if feature_val.exists():
        with open(feature_val, "rb") as f:
            data = orjson.loads(f.read())
            summary = data["summary"]
            print(f"\n✅ Feature Validation Complete:")
            print(f"   - Total features: {summary['total_features']}")
            print(f"   - Passed features: {summary['passed_features']}")
            print(
                f"   - Pass rate: {summary['passed_features']/summary['total_features']*100:.1f}%"
            )

    # Check model results
    models_dir = latest_dir / "models"
    if models_dir.exists():
        print(f"\n📊 Model Training Progress:")
        for model_dir in sorted(models_dir.glob("*_full")):
            model_name = model_dir.name.replace("_full", "")
            results_file = model_dir / "training_results.json"

            if results_file.exists():
                with open(results_file, "rb") as f:
                    results = orjson.loads(f.read())
                    oof = results["oof_metrics"]
                    print(f"\n   {model_name.upper()}:")
                    print(f"   - QWK: {oof['qwk']:.4f}")
                    print(f"   - Macro F1: {oof['f1_macro']:.4f}")
                    print(f"   - Accuracy: {oof['accuracy']:.4f}")
            else:
                print(f"\n   {model_name}: Training in progress...")

    # Check for final summary
    summary_file = latest_dir / "TERTILE_COMPARISON_SUMMARY.md"
    if summary_file.exists():
        print(f"\n✅ ALL EXPERIMENTS COMPLETE!")
        print(f"   Summary saved to: {summary_file}")
        return True

    return False


def main():
    """Monitor experiments until complete."""
    print("Monitoring tertile experiments...")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 60)

    start_time = time.time()

    try:
        while True:
            complete = check_results()

            if complete:
                elapsed = time.time() - start_time
                print(f"\nTotal time: {elapsed/60:.1f} minutes")
                break

            # Show elapsed time
            elapsed = time.time() - start_time
            print(f"\nElapsed: {elapsed/60:.1f} minutes", end="", flush=True)

            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            print("\r" + " " * 50 + "\r", end="", flush=True)  # Clear line

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
