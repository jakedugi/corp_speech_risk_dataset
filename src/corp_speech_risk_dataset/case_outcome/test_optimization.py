#!/usr/bin/env python3
"""
test_optimization.py

Test script to validate the optimization setup and ensure everything works correctly.
"""

import sys
import pandas as pd
from pathlib import Path

# Add the parent directory to the path so we can import the optimization modules
sys.path.append(str(Path(__file__).parent))

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent.parent

from grid_search_optimizer import GridSearchOptimizer


def test_gold_standard_loading():
    """Test that the gold standard data loads correctly."""
    print("🧪 Testing gold standard data loading...")

    # Use paths relative to project root
    gold_standard_path = (
        project_root / "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    )
    extracted_data_root = project_root / "data/extracted"

    try:
        optimizer = GridSearchOptimizer(
            gold_standard_path=str(gold_standard_path),
            extracted_data_root=str(extracted_data_root),
        )

        print(f"✅ Gold standard loaded: {len(optimizer.gold_standard_data)} cases")
        print(
            f"   Amount range: ${optimizer.gold_standard_data['final_amount'].min():,.0f} - ${optimizer.gold_standard_data['final_amount'].max():,.0f}"
        )

        return True
    except Exception as e:
        print(f"❌ Failed to load gold standard: {e}")
        return False


def test_case_data_access():
    """Test that case data can be accessed for each gold standard case."""
    print("\n🧪 Testing case data access...")

    # Use paths relative to project root
    gold_standard_path = (
        project_root / "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    )
    extracted_data_root = project_root / "data/extracted"

    try:
        optimizer = GridSearchOptimizer(
            gold_standard_path=str(gold_standard_path),
            extracted_data_root=str(extracted_data_root),
        )

        accessible_cases = 0
        total_cases = len(optimizer.gold_standard_data)

        for idx, row in optimizer.gold_standard_data.iterrows():
            case_id = row["case_id"]
            case_path = optimizer._get_case_data_path(case_id)

            if case_path and case_path.exists():
                accessible_cases += 1
                print(f"   ✅ {case_id}")
            else:
                print(f"   ❌ {case_id} - data not found")

        print(
            f"\n📊 Case data accessibility: {accessible_cases}/{total_cases} cases accessible"
        )

        if accessible_cases == 0:
            print("⚠️  No case data accessible! Check extracted data paths.")
            return False

        return accessible_cases > 0

    except Exception as e:
        print(f"❌ Failed to test case data access: {e}")
        return False


def test_single_prediction():
    """Test that a single prediction can be made."""
    print("\n🧪 Testing single prediction...")

    # Use paths relative to project root
    gold_standard_path = (
        project_root / "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
    )
    extracted_data_root = project_root / "data/extracted"

    try:
        optimizer = GridSearchOptimizer(
            gold_standard_path=str(gold_standard_path),
            extracted_data_root=str(extracted_data_root),
        )

        # Test with default hyperparameters
        test_hyperparams = {
            "min_amount": 10000,
            "context_chars": 100,
            "min_features": 2,
            "case_position_threshold": 0.5,
            "docket_position_threshold": 0.5,
            "fee_shifting_ratio_threshold": 1.0,
            "patent_ratio_threshold": 20.0,
            "dismissal_ratio_threshold": 0.5,
            "bankruptcy_ratio_threshold": 0.5,
            "proximity_pattern_weight": 1.0,
            "judgment_verbs_weight": 1.0,
            "case_position_weight": 1.0,
            "docket_position_weight": 1.0,
            "all_caps_titles_weight": 1.0,
            "document_titles_weight": 1.0,
            "header_chars": 2000,
        }

        # Test on first accessible case
        for idx, row in optimizer.gold_standard_data.iterrows():
            case_id = row["case_id"]
            case_path = optimizer._get_case_data_path(case_id)

            if case_path and case_path.exists():
                actual_amount = row["final_amount"]
                predicted = optimizer._predict_case_outcome(case_id, test_hyperparams)

                print(f"   Case: {case_id}")
                print(f"   Actual: ${actual_amount:,.0f}")
                print(
                    f"   Predicted: ${predicted:,.0f}"
                    if predicted
                    else "   Predicted: None"
                )

                return True

        print("❌ No accessible cases found for testing")
        return False

    except Exception as e:
        print(f"❌ Failed to test single prediction: {e}")
        return False


def test_hyperparameter_combinations():
    """Test that hyperparameter combinations can be generated."""
    print("\n🧪 Testing hyperparameter combination generation...")

    try:
        optimizer = GridSearchOptimizer(
            gold_standard_path=str(
                project_root
                / "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
            ),
            extracted_data_root=str(project_root / "data/extracted"),
        )

        combinations = optimizer._generate_hyperparameter_combinations()

        print(f"✅ Generated {len(combinations)} hyperparameter combinations")

        # Show first few combinations
        for i, combo in enumerate(combinations[:3]):
            print(f"   Combination {i+1}: {combo}")

        return len(combinations) > 0

    except Exception as e:
        print(f"❌ Failed to generate hyperparameter combinations: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Running optimization setup tests...\n")

    tests = [
        ("Gold Standard Loading", test_gold_standard_loading),
        ("Case Data Access", test_case_data_access),
        ("Single Prediction", test_single_prediction),
        ("Hyperparameter Combinations", test_hyperparameter_combinations),
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
        print("🎉 All tests passed! Optimization setup is ready.")
        print("\nNext steps:")
        print("1. Run quick test: python run_optimization.py")
        print("2. Run full optimization: python run_optimization.py --full-grid")
    else:
        print(
            "⚠️  Some tests failed. Please check the setup before running optimization."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
