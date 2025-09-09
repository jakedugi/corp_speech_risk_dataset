#!/usr/bin/env python3
"""
Generate comprehensive academic figures and statistics for binary dataset paper (2-fold version).

Modified version of generate_dataset_paper_figures_binary.py specifically for 2-fold datasets.
Creates LaTeX-ready figures and tables describing the final clean binary dataset:
- Core dataset statistics
- Court and jurisdiction coverage
- Speaker distribution
- Binary label distribution (case-wise and record-wise)
- Support statistics
- Token/word count analysis
- Case size distribution

Outputs both individual figures and a combined LaTeX document with PDF export.
"""

import sys
import os
from pathlib import Path

# Add the original script directory to path so we can import functions
sys.path.insert(0, os.path.dirname(__file__))

# Import all functions from the original script
from generate_dataset_paper_figures_binary import *


def analyze_dataset_2fold(input_file: str, kfold_dir: Path) -> dict:
    """Modified analyze_dataset function that works with any number of folds."""
    from generate_dataset_paper_figures_binary import analyze_dataset
    import json

    # First, patch the analyze_dataset function to work with 2-fold data
    # Load the metadata to find the final fold
    with open(kfold_dir / "per_fold_metadata.json") as f:
        per_fold_metadata = json.load(f)

    # Find the highest numbered fold (final training fold)
    fold_keys = list(per_fold_metadata["binning"]["fold_edges"].keys())
    final_fold_key = max(fold_keys, key=lambda x: int(x.split("_")[1]))

    print(f"Available folds: {fold_keys}")
    print(f"Using final training fold: {final_fold_key}")

    # Temporarily monkey-patch the fold reference in the original function
    original_code = None
    script_path = Path(__file__).parent / "generate_dataset_paper_figures_binary.py"

    # Read the original file
    with open(script_path, "r") as f:
        original_code = f.read()

    # Replace fold_4 reference with the actual final fold
    modified_code = original_code.replace(
        'fold_edges"]["fold_4"]', f'fold_edges"]["{final_fold_key}"]'
    )
    modified_code = modified_code.replace(
        "fold 4 (final training fold)", f"{final_fold_key} (final training fold)"
    )
    modified_code = modified_code.replace(
        "fold 4 binary boundary", f"{final_fold_key} binary boundary"
    )

    # Write temporary modified file
    temp_script_path = (
        script_path.parent / "temp_generate_dataset_paper_figures_binary.py"
    )
    with open(temp_script_path, "w") as f:
        f.write(modified_code)

    try:
        # Import from the temporary file
        import importlib.util

        spec = importlib.util.spec_from_file_location("temp_module", temp_script_path)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)

        # Call the modified function
        result = temp_module.analyze_dataset(input_file, kfold_dir)

        return result

    finally:
        # Clean up temporary file
        if temp_script_path.exists():
            temp_script_path.unlink()


def main():
    """Main execution function for binary dataset analysis (2-fold version)."""
    print("=" * 60)
    print("BINARY DATASET ANALYSIS AND FIGURE GENERATION (2-FOLD)")
    print("=" * 60)

    # Configuration - Updated for 2-fold binary classification
    input_file = "data/enhanced_combined_FINAL/final_clean_dataset_with_interpretable_features.jsonl"
    output_dir = Path("docs/dataset_analysis_binary_2fold")
    kfold_dir = Path("data/final_stratified_kfold_splits_binary_quote_balanced_2fold")

    print(f"Input file: {input_file}")
    print(f"K-fold directory: {kfold_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if input files exist
    if not Path(input_file).exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return

    if not kfold_dir.exists():
        print(f"❌ Error: K-fold directory not found: {kfold_dir}")
        print("💡 Tip: Run the binary k-fold splitting first with 2 folds")
        return

    # Analyze dataset
    print("🔍 Analyzing dataset...")
    analysis = analyze_dataset_2fold(input_file, kfold_dir)

    # Compute statistics
    print("📊 Computing summary statistics...")
    stats = create_summary_stats(analysis)

    # Load k-fold analysis
    print("📁 Loading k-fold analysis...")
    kfold_stats = load_kfold_analysis(kfold_dir)

    # Create figures
    print("🎨 Creating figures...")
    create_figures(analysis, stats, output_dir)

    # Create k-fold figures (if data available)
    print("📈 Creating k-fold figures...")
    try:
        create_kfold_figures(kfold_stats, output_dir)
        print("✅ K-fold figures created successfully")
    except Exception as e:
        print(f"⚠️  Could not create k-fold figures: {e}")

    # Create LaTeX document with k-fold analysis
    print("📝 Creating LaTeX document...")
    try:
        create_latex_document(analysis, stats, output_dir, kfold_stats)
        print("✅ LaTeX document created successfully")
    except Exception as e:
        print(f"⚠️  Could not create LaTeX document: {e}")

    print("\n" + "=" * 60)
    print("BINARY ANALYSIS COMPLETE (2-FOLD)!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Figures: {output_dir}/figures/")
    print(f"LaTeX source: {output_dir}/dataset_analysis.tex")
    print("\nReady for binary classification paper submission (2-fold version)!")


if __name__ == "__main__":
    main()
