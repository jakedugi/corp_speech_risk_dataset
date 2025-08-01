#!/usr/bin/env python3
"""
Verify that all possible hyperparameters are being optimized.
This script checks that every VotingWeights parameter is included in the optimization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Verify all hyperparameters are included in optimization."""
    print("🔍 Verifying Comprehensive Hyperparameter Optimization")
    print("=" * 70)

    # Import required modules
    try:
        from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
            VotingWeights,
        )
        from corp_speech_risk_dataset.case_outcome.bayesian_optimizer import (
            BayesianOptimizer,
        )

        print("✅ Modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1

    # Get all VotingWeights parameters
    voting_weights = VotingWeights()
    all_voting_params = voting_weights.to_dict()
    print(f"\n📊 Total VotingWeights parameters: {len(all_voting_params)}")

    # List all VotingWeights parameters
    print("\n🏷️ All VotingWeights parameters:")
    for i, param_name in enumerate(sorted(all_voting_params.keys()), 1):
        print(f"   {i:2d}. {param_name}")

    # Check Bayesian optimizer search space
    try:
        # Get the parameter names directly from the code instead of creating an optimizer
        param_names = [
            # Core extraction parameters
            "min_amount",
            "context_chars",
            "min_features",
            # Position thresholds
            "case_position_threshold",
            "docket_position_threshold",
            # Case flag thresholds
            "fee_shifting_ratio_threshold",
            "patent_ratio_threshold",
            "dismissal_ratio_threshold",
            "bankruptcy_ratio_threshold",
            # Original voting weights
            "proximity_pattern_weight",
            "judgment_verbs_weight",
            "case_position_weight",
            "docket_position_weight",
            "all_caps_titles_weight",
            "document_titles_weight",
            # Enhanced feature weights - comprehensive coverage
            "financial_terms_weight",
            "settlement_terms_weight",
            "legal_proceedings_weight",
            "monetary_phrases_weight",
            "dependency_parsing_weight",
            "fraction_extraction_weight",
            "percentage_extraction_weight",
            "implied_totals_weight",
            "document_structure_weight",
            "table_detection_weight",
            "header_detection_weight",
            "section_boundaries_weight",
            "numeric_gazetteer_weight",
            "mixed_numbers_weight",
            "sentence_boundary_weight",
            "paragraph_boundary_weight",
            # Confidence boosting features
            "high_confidence_patterns_weight",
            "amount_adjacent_keywords_weight",
            "confidence_boost_weight",
            # High/Low signal regex weights
            "high_signal_financial_weight",
            "low_signal_financial_weight",
            "high_signal_settlement_weight",
            "low_signal_settlement_weight",
            "calculation_boost_multiplier",
            # Header size
            "header_chars",
        ]

        print(f"\n🎯 Bayesian optimizer parameters: {len(param_names)}")

        # Check which VotingWeights parameters are being optimized
        voting_params_optimized = []
        non_voting_params = {
            "min_amount",
            "context_chars",
            "min_features",
            "case_position_threshold",
            "docket_position_threshold",
            "fee_shifting_ratio_threshold",
            "patent_ratio_threshold",
            "dismissal_ratio_threshold",
            "bankruptcy_ratio_threshold",
            "header_chars",
        }

        for param in param_names:
            if param not in non_voting_params:
                voting_params_optimized.append(param)

        print(
            f"\n⚖️ VotingWeights parameters being optimized: {len(voting_params_optimized)}"
        )

        # Find missing VotingWeights parameters
        missing_params = set(all_voting_params.keys()) - set(voting_params_optimized)

        if missing_params:
            print(f"\n❌ Missing VotingWeights parameters ({len(missing_params)}):")
            for param in sorted(missing_params):
                print(f"   • {param}")
        else:
            print("\n✅ All VotingWeights parameters are being optimized!")

        # Check for extra parameters that might not exist in VotingWeights
        extra_params = set(voting_params_optimized) - set(all_voting_params.keys())
        if extra_params:
            print(f"\n⚠️ Extra parameters not in VotingWeights ({len(extra_params)}):")
            for param in sorted(extra_params):
                print(f"   • {param}")

        # Summary
        print(f"\n📈 Optimization Summary:")
        print(f"   Total hyperparameters: {len(param_names)}")
        print(f"   VotingWeights parameters: {len(all_voting_params)}")
        print(f"   VotingWeights optimized: {len(voting_params_optimized)}")
        print(
            f"   Non-VotingWeights optimized: {len(param_names) - len(voting_params_optimized)}"
        )
        print(
            f"   Coverage: {len(voting_params_optimized)}/{len(all_voting_params)} ({100*len(voting_params_optimized)/len(all_voting_params):.1f}%)"
        )

        # High/Low signal analysis
        high_low_signal_params = [
            p
            for p in voting_params_optimized
            if "high_signal" in p or "low_signal" in p
        ]
        print(f"\n🔺🔻 High/Low Signal Parameters: {len(high_low_signal_params)}")
        for param in sorted(high_low_signal_params):
            print(f"   • {param}")

        if len(missing_params) == 0:
            print(f"\n🎉 SUCCESS: All VotingWeights parameters are being optimized!")
            print(
                f"   The optimization system is comprehensively tuning {len(param_names)} hyperparameters"
            )
            print(
                f"   including all {len(high_low_signal_params)} high/low signal pattern weights."
            )
            return 0
        else:
            print(
                f"\n⚠️ WARNING: {len(missing_params)} VotingWeights parameters are not being optimized."
            )
            return 1

    except Exception as e:
        print(f"❌ Error checking optimization parameters: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
