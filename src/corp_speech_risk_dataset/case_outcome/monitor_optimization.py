#!/usr/bin/env python3
"""
monitor_optimization.py

Simple script to monitor the current best parameters during Bayesian optimization.
"""

import time
import json
from pathlib import Path
from datetime import datetime


def monitor_optimization_log(log_file_path: str, check_interval: int = 30):
    """
    Monitor the optimization log file for new best parameters.

    Args:
        log_file_path: Path to the optimization log file
        check_interval: How often to check for updates (seconds)
    """
    log_path = Path(log_file_path)

    if not log_path.exists():
        print(f"❌ Log file not found: {log_file_path}")
        return

    print(f"🔍 Monitoring optimization log: {log_file_path}")
    print(f"⏱️  Checking every {check_interval} seconds...")
    print("=" * 80)

    last_position = 0
    best_params_seen = set()

    while True:
        try:
            with open(log_path, "r") as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()

                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Look for "NEW BEST!" messages
                    if "🏆 NEW BEST!" in line:
                        print(f"\n{datetime.now().strftime('%H:%M:%S')} - {line}")

                        # Extract evaluation number
                        if "Evaluation" in line:
                            eval_num = line.split("Evaluation ")[1].split("/")[0]
                            print(f"   📈 Evaluation #{eval_num}")

                    # Look for best hyperparameters
                    elif "Best hyperparams:" in line:
                        print(f"   {line}")
                    elif "Position thresholds:" in line:
                        print(f"   {line}")
                    elif "Voting weights:" in line:
                        print(f"   {line}")
                    elif "Exact matches:" in line:
                        print(f"   {line}")
                        print("   " + "-" * 60)

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring log: {e}")
            time.sleep(check_interval)


def main():
    """Main function to start monitoring."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor Bayesian optimization progress"
    )
    parser.add_argument(
        "--log-file",
        default="logs/bayesian_optimization_*.log",
        help="Path to optimization log file (supports glob patterns)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)",
    )

    args = parser.parse_args()

    # Find the most recent log file
    from glob import glob

    log_files = glob(args.log_file)

    if not log_files:
        print(f"❌ No log files found matching: {args.log_file}")
        return

    # Use the most recent log file
    latest_log = max(log_files, key=lambda f: Path(f).stat().st_mtime)
    print(f"📁 Using log file: {latest_log}")

    monitor_optimization_log(latest_log, args.interval)


if __name__ == "__main__":
    main()
