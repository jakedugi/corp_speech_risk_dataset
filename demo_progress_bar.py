#!/usr/bin/env python3
"""
Demo of the progress bar that will appear once the optimization works
"""
import time
from datetime import timedelta


def demo_progress_bar(
    current: int, total: int, eta_str: str = "", mse: float = 0.0, f1: float = 0.0
):
    """Demo progress bar display."""
    progress = current / total
    bar_length = 40
    filled_length = int(bar_length * progress)

    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    percentage = progress * 100

    # Color coding for terminal
    if percentage < 25:
        color = "\033[91m"  # Red
    elif percentage < 50:
        color = "\033[93m"  # Yellow
    elif percentage < 75:
        color = "\033[94m"  # Blue
    else:
        color = "\033[92m"  # Green
    reset = "\033[0m"

    # Format metrics
    metrics = f"MSE: {mse:.2e}, F1: {f1:.3f}" if mse > 0 else "Calculating..."

    # Create progress line
    progress_line = f"\r{color}[{bar}] {percentage:6.1f}%{reset} | Eval {current}/{total} | {metrics}"
    if eta_str:
        progress_line += f" | ETA: {eta_str}"

    # Print without newline to overwrite previous line
    print(progress_line, end="", flush=True)

    # Add newline at completion
    if current == total:
        print()


def demo_optimization():
    """Demo what the optimization progress will look like"""
    print("\n🎯 Optimization Progress Demo:")
    print("=" * 80)

    total_evals = 20
    for i in range(1, total_evals + 1):
        # Simulate realistic values
        mse = 5.0e17 / (1 + i * 0.1)  # Decreasing MSE
        f1 = 0.1 + (i / total_evals) * 0.3  # Improving F1

        # Calculate ETA
        if i > 1:
            remaining = total_evals - i
            eta_seconds = remaining * 2  # 2 seconds per eval
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "calculating..."

        demo_progress_bar(i, total_evals, eta_str, mse, f1)

        # Add new best indicators
        if i in [3, 8, 15]:
            print()
            print(
                f"🏆 NEW BEST! Eval {i}/{total_evals} - MSE: {mse:.2e}, F1: {f1:.3f}, Matches: {i}/{total_evals}"
            )

        time.sleep(0.5)  # Simulate processing time

    print()
    print("=" * 80)
    print("✅ Optimization completed! Duration: 0:00:40.123456")
    print("=" * 80)


if __name__ == "__main__":
    demo_optimization()
