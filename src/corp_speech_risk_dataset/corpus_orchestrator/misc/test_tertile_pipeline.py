#!/usr/bin/env python3
"""
Quick test script to verify tertile pipeline functionality.
Tests with a small sample to ensure everything works before full run.
"""

import subprocess
import sys
from pathlib import Path
import json
import orjson
import time


def run_command(cmd, description):
    """Run a command and check for success."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ FAILED after {elapsed_time:.1f}s")
        print(f"STDERR:\n{result.stderr}")
        return False
    else:
        print(f"✅ SUCCESS in {elapsed_time:.1f}s")
        # Print last few lines of output
        lines = result.stdout.strip().split("\n")
        if len(lines) > 5:
            print("Last 5 lines of output:")
            for line in lines[-5:]:
                print(f"  {line}")
        return True


def main():
    """Run quick tests of tertile pipeline."""
    print("TERTILE PIPELINE TEST")
    print("====================")

    data_dir = "data/final_stratified_kfold_splits_authoritative_complete"
    test_output_base = "results/tertile_test_" + str(int(time.time()))

    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"Test output: {test_output_base}")

    all_passed = True

    # Test 1: Feature validation with tiny sample
    print("\n\nTEST 1: Feature Validation (tiny sample)")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/unified_tertile_feature_pipeline.py",
        "--data-dir",
        data_dir,
        "--output-dir",
        f"{test_output_base}/feature_validation",
        "--fold",
        "3",
        "--sample-size",
        "1000",  # Very small sample
        "--iterations",
        "1",
    ]

    if not run_command(cmd, "Tertile feature validation"):
        all_passed = False
    else:
        # Check if results were created
        results_file = Path(
            f"{test_output_base}/feature_validation/validation_results_iter_1.json"
        )
        if results_file.exists():
            with open(results_file, "rb") as f:
                data = orjson.loads(f.read())
                summary = data["summary"]
                print(f"\nValidation Summary:")
                print(f"  - Features tested: {summary['total_features']}")
                print(f"  - Features passed: {summary['passed_features']}")
        else:
            print("⚠️  Results file not found")
            all_passed = False

    # Test 2: Quick model training (ElasticNet only, one fold)
    print("\n\nTEST 2: Model Training (ElasticNet, limited)")

    # First, create a test script that trains only one fold
    test_train_script = f"{test_output_base}/test_train.py"
    Path(test_output_base).mkdir(parents=True, exist_ok=True)

    with open(test_train_script, "w") as f:
        f.write(
            """
import sys
sys.path.insert(0, ".")
from scripts.run_tertile_comprehensive_models import TertileModelTrainer
from pathlib import Path

# Quick test with just one fold
trainer = TertileModelTrainer(
    data_dir=Path(sys.argv[1]),
    output_dir=Path(sys.argv[2]),
    model_type="elasticnet"
)

# Override to train just one fold with minimal grid
trainer.HYPERPARAMETER_GRIDS['elasticnet'] = {
    'C': [1.0],
    'l1_ratio': [0.5],
    'solver': ['saga'],
    'penalty': ['elasticnet'],
    'multi_class': ['multinomial'],
    'max_iter': [100]
}

# Train just fold 3
result = trainer.train_fold(3)
print(f"Test QWK: {result['test']['qwk']:.4f}")
print("✅ Model training test passed!")
"""
        )

    cmd = [
        "uv",
        "run",
        "python",
        test_train_script,
        data_dir,
        f"{test_output_base}/model_test",
    ]

    if not run_command(cmd, "ElasticNet model training (1 fold)"):
        all_passed = False

    # Test 3: Verify data loading and feature detection
    print("\n\nTEST 3: Data and Feature Verification")

    verify_script = f"{test_output_base}/verify_data.py"
    with open(verify_script, "w") as f:
        f.write(
            f"""
import pandas as pd
import orjson
from pathlib import Path

data_dir = Path("{data_dir}")

# Load metadata
with open(data_dir / "per_fold_metadata.json", 'rb') as f:
    metadata = orjson.loads(f.read())

print("Metadata loaded successfully")
print(f"Binning method: {{metadata['binning']['method']}}")
print(f"Methodology: {{metadata['methodology']}}")
print(f"Number of folds: {{len(metadata['weights'])}}")

# Check one fold
fold_dir = data_dir / "fold_3"
# Load just first 100 lines with orjson
records = []
with open(fold_dir / "train.jsonl", 'rb') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        if line.strip():
            records.append(orjson.loads(line))
train_df = pd.DataFrame(records)

# Count features
feat_cols = [c for c in train_df.columns if c.startswith(('interpretable_', 'feat_new'))]
print(f"\\nFeature columns found: {{len(feat_cols)}}")
print(f"Sample features: {{feat_cols[:5]}}")

# Check labels
unique_labels = train_df['outcome_bin'].unique()
print(f"\\nUnique labels: {{sorted(unique_labels)}}")
print(f"Label distribution: {{train_df['outcome_bin'].value_counts().sort_index().to_dict()}}")

if set(unique_labels) == {{0, 1, 2}}:
    print("✅ Tertile labels verified!")
else:
    print("❌ Invalid labels for tertile classification")
"""
        )

    cmd = ["uv", "run", "python", verify_script]
    if not run_command(cmd, "Data and feature verification"):
        all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nThe tertile pipeline is working correctly.")
        print("You can now run the full experiments with:")
        print("  ./scripts/run_all_tertile_experiments.sh")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above before running full experiments.")

    print(f"\nTest output directory: {test_output_base}")


if __name__ == "__main__":
    main()
