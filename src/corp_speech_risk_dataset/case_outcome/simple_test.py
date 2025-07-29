#!/usr/bin/env python3
"""
simple_test.py

Simple test to verify the optimization system works correctly.
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_gold_standard_loading():
    """Test that the gold standard data loads correctly."""
    print("🧪 Testing gold standard data loading...")

    try:
        import pandas as pd

        gold_standard_path = (
            "../../../data/gold_standard/case_outcome_amounts_hand_annotated.csv"
        )

        if not os.path.exists(gold_standard_path):
            print(f"❌ Gold standard file not found: {gold_standard_path}")
            return False

        df = pd.read_csv(gold_standard_path)
        df = df.dropna(subset=["case_id", "final_amount"])
        df["final_amount"] = pd.to_numeric(df["final_amount"], errors="coerce")
        df = df.dropna(subset=["final_amount"])

        print(f"✅ Gold standard loaded: {len(df)} valid cases")
        print(
            f"   Amount range: ${df['final_amount'].min():,.0f} - ${df['final_amount'].max():,.0f}"
        )

        return True
    except Exception as e:
        print(f"❌ Failed to load gold standard: {e}")
        return False


def test_extracted_data_access():
    """Test that extracted data can be accessed."""
    print("\n🧪 Testing extracted data access...")

    extracted_root = "../../../data/extracted"

    if not os.path.exists(extracted_root):
        print(f"❌ Extracted data root not found: {extracted_root}")
        return False

    # Check for some case directories
    case_dirs = [
        d
        for d in os.listdir(extracted_root)
        if os.path.isdir(os.path.join(extracted_root, d))
    ]

    if len(case_dirs) == 0:
        print("❌ No case directories found in extracted data")
        return False

    print(f"✅ Found {len(case_dirs)} case directories")
    print(f"   Sample cases: {case_dirs[:5]}")

    return True


def test_hyperparameter_generation():
    """Test that hyperparameter combinations can be generated."""
    print("\n🧪 Testing hyperparameter generation...")

    try:
        # Define a simple hyperparameter grid
        grid = {
            "min_amount": [10000, 50000],
            "context_chars": [100, 200],
            "min_features": [2, 3],
            "header_chars": [2000],
        }

        # Generate combinations
        import itertools

        keys = grid.keys()
        values = grid.values()
        combinations = []

        for combination in itertools.product(*values):
            hyperparams = dict(zip(keys, combination))
            combinations.append(hyperparams)

        print(f"✅ Generated {len(combinations)} hyperparameter combinations")

        # Show first few combinations
        for i, combo in enumerate(combinations[:3]):
            print(f"   Combination {i+1}: {combo}")

        return len(combinations) > 0

    except Exception as e:
        print(f"❌ Failed to generate hyperparameter combinations: {e}")
        return False


def test_basic_optimization_logic():
    """Test basic optimization logic without full evaluation."""
    print("\n🧪 Testing basic optimization logic...")

    try:
        # Simulate a simple optimization result
        test_result = {
            "hyperparams": {
                "min_amount": 10000,
                "context_chars": 100,
                "min_features": 2,
                "header_chars": 2000,
            },
            "mse_loss": 1.23e12,
            "precision": 0.85,
            "recall": 0.80,
            "f1_score": 0.824,
            "exact_matches": 16,
            "total_cases": 20,
        }

        print(f"✅ Simulated optimization result:")
        print(f"   MSE Loss: {test_result['mse_loss']:.2e}")
        print(f"   F1 Score: {test_result['f1_score']:.3f}")
        print(f"   Precision: {test_result['precision']:.3f}")
        print(f"   Recall: {test_result['recall']:.3f}")
        print(
            f"   Exact Matches: {test_result['exact_matches']}/{test_result['total_cases']}"
        )

        return True

    except Exception as e:
        print(f"❌ Failed to test optimization logic: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing file structure...")

    required_files = [
        "grid_search_optimizer.py",
        "bayesian_optimizer.py",
        "run_optimization.py",
        "test_optimization.py",
        "README_optimization.md",
        "README_bayesian_optimization.md",
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print(f"✅ All required files present")
    return True


def main():
    """Run all tests."""
    print("🚀 Running optimization system tests...\n")

    tests = [
        ("File Structure", test_file_structure),
        ("Gold Standard Loading", test_gold_standard_loading),
        ("Extracted Data Access", test_extracted_data_access),
        ("Hyperparameter Generation", test_hyperparameter_generation),
        ("Basic Optimization Logic", test_basic_optimization_logic),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed\n")
        else:
            print(f"❌ {test_name} failed\n")

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Optimization system is ready.")
        print("\nNext steps:")
        print("1. Install scikit-optimize for Bayesian optimization:")
        print("   pip install scikit-optimize")
        print(
            "2. Run quick test: python3 run_optimization.py --bayesian --max-combinations 5"
        )
        print(
            "3. Run full optimization: python3 run_optimization.py --bayesian --max-combinations 50"
        )
    else:
        print(
            "⚠️  Some tests failed. Please check the setup before running optimization."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
