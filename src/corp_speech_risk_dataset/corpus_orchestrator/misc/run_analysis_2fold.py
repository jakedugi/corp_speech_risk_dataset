#!/usr/bin/env python3
"""
Run the full binary dataset analysis pipeline for 2-fold data.

This script dynamically patches the original analysis script to work with 2-fold data
and generates all figures and documentation.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path


def create_patched_script():
    """Create a version of the original script that works with 2-fold data."""

    # Read the original script
    original_script = Path(__file__).parent / "generate_dataset_paper_figures_binary.py"
    with open(original_script, "r") as f:
        content = f.read()

    # Apply patches for 2-fold compatibility
    patches = [
        # Replace hardcoded fold_4 references with dynamic fold detection
        (
            'final_fold_edges = per_fold_metadata["binning"]["fold_edges"]["fold_4"]',
            """# Dynamically find the final fold (highest numbered)
        fold_keys = list(per_fold_metadata['binning']['fold_edges'].keys())
        final_fold_key = max(fold_keys, key=lambda x: int(x.split('_')[1]))
        print(f"Using final training fold: {final_fold_key}")
        final_fold_edges = per_fold_metadata["binning"]["fold_edges"][final_fold_key]""",
        ),
        # Update related references
        (
            "✅ Inherited fold 4 binary boundary:",
            f"✅ Inherited {{final_fold_key}} binary boundary:",
        ),
        ("fold 4 (final training fold)", "{final_fold_key} (final training fold)"),
        # Update the main function paths for 2-fold
        (
            'input_file = "data/enhanced_combined_FINAL/final_clean_dataset_with_interpretable_features.jsonl"',
            'input_file = "data/enhanced_combined_FINAL/final_clean_dataset_with_interpretable_features.jsonl"',
        ),
        (
            'output_dir = Path("docs/dataset_analysis_binary")',
            'output_dir = Path("docs/dataset_analysis_binary_2fold")',
        ),
        (
            'kfold_dir = Path("data/final_stratified_kfold_splits_binary_quote_balanced")',
            'kfold_dir = Path("data/final_stratified_kfold_splits_binary_quote_balanced_2fold")',
        ),
        # Update title
        (
            'print("BINARY DATASET ANALYSIS AND FIGURE GENERATION")',
            'print("BINARY DATASET ANALYSIS AND FIGURE GENERATION (2-FOLD)")',
        ),
        (
            'print("BINARY ANALYSIS COMPLETE!")',
            'print("BINARY ANALYSIS COMPLETE (2-FOLD)!")',
        ),
        (
            'print("\\nReady for binary classification paper submission!")',
            'print("\\nReady for binary classification paper submission (2-fold version)!")',
        ),
    ]

    # Apply patches
    for old, new in patches:
        if "{final_fold_key}" in new:
            # This patch needs the final_fold_key variable, so we need to handle it differently
            continue
        content = content.replace(old, new)

    # For the final_fold_key references, we need a more sophisticated replacement
    # Let's find and replace the specific print statements
    content = content.replace(
        'print(f"✅ Inherited fold 4 binary boundary: ${binary_edge:,.0f}")',
        'print(f"✅ Inherited {final_fold_key} binary boundary: ${binary_edge:,.0f}")',
    )

    return content


def main():
    """Run the 2-fold analysis by creating and executing a patched script."""
    print("=" * 60)
    print("RUNNING 2-FOLD BINARY DATASET ANALYSIS")
    print("=" * 60)

    # Create temporary patched script
    patched_content = create_patched_script()

    # Write to temporary file
    temp_script = Path(__file__).parent / "temp_binary_analysis_2fold.py"

    try:
        with open(temp_script, "w") as f:
            f.write(patched_content)

        print(f"✅ Created patched script: {temp_script}")

        # Execute the patched script
        import subprocess

        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=False,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode == 0:
            print("✅ Analysis completed successfully!")
        else:
            print(f"❌ Analysis failed with return code: {result.returncode}")

    finally:
        # Clean up temporary file
        if temp_script.exists():
            temp_script.unlink()
            print(f"🧹 Cleaned up temporary script")


if __name__ == "__main__":
    main()
