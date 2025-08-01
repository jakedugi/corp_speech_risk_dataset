#!/usr/bin/env python3
"""
Simple debug test for the optimization system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")

    try:
        from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
            VotingWeights,
            count_high_signal_financial_terms,
            count_low_signal_financial_terms,
        )

        print("✅ Stage1 module imported successfully")

        # Test VotingWeights with minimal parameters
        weights = VotingWeights()
        print(f"✅ Default VotingWeights created: {len(weights.to_dict())} parameters")

        # Test custom weights
        custom_weights = VotingWeights(
            high_signal_financial_weight=2.0,
            low_signal_financial_weight=0.5,
        )
        print("✅ Custom VotingWeights created successfully")

    except Exception as e:
        print(f"❌ Stage1 import error: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        from corp_speech_risk_dataset.case_outcome.case_outcome_imputer import (
            scan_stage1,
        )

        print("✅ Case outcome imputer imported successfully")
    except Exception as e:
        print(f"❌ Case outcome imputer error: {e}")
        return False

    return True


def test_simple_grid_search():
    """Test a minimal grid search configuration."""
    print("\n🧪 Testing simple grid search...")

    try:
        from corp_speech_risk_dataset.case_outcome.grid_search_optimizer import (
            GridSearchOptimizer,
        )

        # Check if data exists
        gold_standard = Path(
            "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
        )
        extracted_data = Path("data/extracted")

        if not gold_standard.exists():
            print(f"❌ Gold standard not found: {gold_standard}")
            return False

        if not extracted_data.exists():
            print(f"❌ Extracted data not found: {extracted_data}")
            return False

        print("✅ Data files exist")

        # Create minimal optimizer
        optimizer = GridSearchOptimizer(
            gold_standard_path=str(gold_standard),
            extracted_data_root=str(extracted_data),
        )

        print("✅ GridSearchOptimizer created successfully")
        print(f"   Gold standard cases loaded: {len(optimizer.gold_standard_data)}")

        # Test a single hyperparameter combination
        test_hyperparams = {
            "min_amount": 1000,
            "context_chars": 200,
            "min_features": 2,
            "case_position_threshold": 0.5,
            "docket_position_threshold": 0.5,
            "proximity_pattern_weight": 1.0,
            "judgment_verbs_weight": 1.0,
            "case_position_weight": 1.0,
            "docket_position_weight": 1.0,
            "all_caps_titles_weight": 1.0,
            "document_titles_weight": 1.0,
            "header_chars": 2000,
            # Add safe defaults for dismissal parameters
            "dismissal_ratio_threshold": 0.5,
            "use_weighted_dismissal_scoring": True,
            "dismissal_document_type_weight": 2.0,
        }

        print("✅ Test hyperparameters created")

        # Test prediction on first case
        if len(optimizer.gold_standard_data) > 0:
            first_case = optimizer.gold_standard_data.iloc[0]
            case_id = str(first_case["case_id"])
            actual_amount = float(first_case["final_amount"])

            print(f"🎯 Testing prediction on case: {case_id}")
            print(f"   Actual amount: ${actual_amount:,.0f}")

            try:
                predicted = optimizer._predict_case_outcome(case_id, test_hyperparams)
                print(
                    f"   Predicted amount: ${predicted:,.0f}"
                    if predicted
                    else "   Predicted: None"
                )
                print("✅ Single prediction successful")
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                import traceback

                traceback.print_exc()
                return False

        return True

    except Exception as e:
        print(f"❌ Grid search test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all debug tests."""
    print("🐛 Debug Test Suite for Optimization System")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("❌ Import tests failed - stopping")
        return 1

    # Test grid search
    if not test_simple_grid_search():
        print("❌ Grid search tests failed")
        return 1

    print("\n✅ All debug tests passed!")
    print("🚀 Optimization system is ready for use")
    return 0


if __name__ == "__main__":
    sys.exit(main())
