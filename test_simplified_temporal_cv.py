#!/usr/bin/env python3
"""
Test the simplified temporal CV with outcome-only 3-bin strategy.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Test the simplified temporal CV approach."""

    print("🧪 Testing Simplified Temporal CV (Outcome-Only 3-Bin Strategy)")
    print("=" * 80)

    # Check data file
    data_file = "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl"
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return 1

    print(f"✅ Data file found: {data_file}")
    print(
        f"📊 Records: {sum(1 for _ in open(data_file)) if Path(data_file).exists() else 0:,}"
    )

    # Test temporal CV creation
    output_dir = "data/temporal_cv_simplified_test"

    print(f"\n🚀 Creating temporal CV splits...")
    print(f"Strategy: Outcome-only 3-bin (low/medium/high)")
    print(f"Support: Weighting-only (not stratification)")
    print(f"Output: {output_dir}")

    cmd = [
        sys.executable,
        "scripts/stratified_kfold_case_split.py",
        "--input",
        data_file,
        "--output-dir",
        output_dir,
        "--k-folds",
        "5",
        "--target-field",
        "final_judgement_real",
        "--stratify-type",
        "regression",
        "--use-temporal-cv",
        "--oof-test-ratio",
        "0.15",
        "--random-seed",
        "42",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Temporal CV creation successful!")

            # Analyze results
            test_dir = Path(output_dir)
            if test_dir.exists():
                print(f"\n📁 Generated structure:")

                # Check folds
                fold_count = len(
                    [
                        d
                        for d in test_dir.iterdir()
                        if d.is_dir() and d.name.startswith("fold_")
                    ]
                )
                print(f"  📂 CV Folds: {fold_count}")

                # Check OOF
                oof_dir = test_dir / "oof_test"
                if oof_dir.exists():
                    print(f"  📂 OOF Test: ✅")

                # Check DNT manifest
                manifest_path = test_dir / "dnt_manifest.json"
                if manifest_path.exists():
                    import json

                    with open(manifest_path) as f:
                        manifest = json.load(f)
                        dnt_columns = manifest.get("do_not_train", [])
                        print(f"  🚫 DNT Columns: {len(dnt_columns)}")

                        # Check that support_tertile is in DNT
                        if "support_tertile" in dnt_columns:
                            print(f"    ✅ support_tertile marked as DNT")
                        else:
                            print(f"    ❌ support_tertile NOT in DNT!")

                # Check methodology
                stats_path = test_dir / "fold_statistics.json"
                if stats_path.exists():
                    import json

                    with open(stats_path) as f:
                        stats = json.load(f)
                        print(f"\n📊 Methodology Verification:")
                        print(f"  Strategy: {stats.get('stratification_approach')}")
                        print(f"  Support: {stats.get('support_handling')}")
                        print(
                            f"  Binning: {stats.get('binning_strategy', {}).get('method')}"
                        )
                        print(
                            f"  Bins: {stats.get('binning_strategy', {}).get('bins')}"
                        )
                        print(
                            f"  Temporal Purity: {stats.get('binning_strategy', {}).get('temporal_purity')}"
                        )
                        print(
                            f"  Composite Labels: {stats.get('binning_strategy', {}).get('composite_labels')}"
                        )

                # Check a sample fold structure
                fold_0 = test_dir / "fold_0"
                if fold_0.exists():
                    print(f"\n📂 Fold 0 Sample:")
                    for split_file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
                        split_path = fold_0 / split_file
                        if split_path.exists():
                            record_count = sum(1 for _ in open(split_path))
                            print(f"  {split_file}: {record_count:,} records")

            return 0
        else:
            print("❌ Temporal CV creation failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return 1

    except subprocess.TimeoutExpired:
        print("❌ Temporal CV creation timed out!")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
