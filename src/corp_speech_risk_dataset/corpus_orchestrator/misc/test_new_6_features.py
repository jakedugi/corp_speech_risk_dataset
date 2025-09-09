#!/usr/bin/env python3
"""
Test script for the 6 new interpretable features:
1. Uncautioned-Claim Proportion (UCP)
2. Documented-Fact Sentence Share (DFSS)
3. Remediation-Action Proximity (RAP)
4. Deontic Modality Balance
5. Ambiguity Lexicon Density
6. Qualified Argument Count

This script tests feature extraction and validates expected behavior.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


def test_new_features():
    """Test all 6 new features with targeted examples."""

    extractor = InterpretableFeatureExtractor()

    print("🔬 TESTING 6 NEW INTERPRETABLE FEATURES")
    print("=" * 60)

    # Test cases designed to trigger specific new features
    test_cases = [
        {
            "name": "High Risk - Uncautioned Claims",
            "text": "We guarantee best results always. This will definitely work. Our product is the most effective solution.",
            "expected_features": [
                "lr_uncautioned_claim_proportion",
                "ling_deontic_oblig_count",
            ],
            "description": "Should have HIGH UCP (many guarantees/superlatives without hedges)",
        },
        {
            "name": "Low Risk - Documented Facts",
            "text": "According to the Q3 report, revenue was $2.5 million, representing 15% growth. The study showed 23% improvement in customer satisfaction.",
            "expected_features": [
                "lr_documented_fact_sentence_share",
                "lr_evidential_strength",
            ],
            "description": "Should have HIGH DFSS (sentences with numbers+units+past/evidential)",
        },
        {
            "name": "Medium Risk - Remediation Context",
            "text": "The company discovered misleading statements but has remediated the issue. Corrective actions were implemented to address the deceptive practices.",
            "expected_features": ["lr_remediation_action_proximity_count"],
            "description": "Should have HIGH RAP (remediation near risk terms)",
        },
        {
            "name": "Low Risk - Deontic Permissions",
            "text": "Customers may cancel anytime. Users can opt-out if desired. Participation is optional and discretionary.",
            "expected_features": [
                "ling_deontic_balance_norm",
                "ling_deontic_perm_count",
            ],
            "description": "Should have HIGH deontic balance (many permissions vs obligations)",
        },
        {
            "name": "Medium Risk - Ambiguous Terms",
            "text": "The product is substantially effective and generally reliable. Results are typically reasonable but may vary.",
            "expected_features": ["lex_ambiguity_count", "lex_ambiguity_norm"],
            "description": "Should have HIGH ambiguity density",
        },
        {
            "name": "Low Risk - Qualified Arguments",
            "text": "We claim success because independent studies show results. The company asserts growth since quarterly data supports this.",
            "expected_features": ["lr_qualified_argument_count"],
            "description": "Should have HIGH qualified arguments (claims with backing)",
        },
    ]

    # Track all new feature names we expect to see
    expected_new_features = {
        # UCP features
        "lr_uncautioned_claim_proportion",
        "lr_uncautioned_claim_count",
        # DFSS features
        "lr_documented_fact_sentence_share",
        "lr_documented_fact_sentence_count",
        # RAP features
        "lr_remediation_action_proximity_count",
        "lr_remediation_action_proximity_norm",
        # Deontic features
        "ling_deontic_oblig_count",
        "ling_deontic_perm_count",
        "ling_deontic_balance_norm",
        # Ambiguity features (from lexicon extraction)
        "lex_ambiguity_count",
        "lex_ambiguity_norm",
        "lex_ambiguity_present",
        "lex_ambiguity_quote_count",
        "lex_ambiguity_quote_norm",
        # Qualified argument features
        "lr_qualified_argument_count",
        "lr_qualified_argument_norm",
    }

    all_found_features = set()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 TEST {i}: {test_case['name']}")
        print(
            f"   Text: {test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}"
        )
        print(f"   Expected: {test_case['description']}")

        # Extract features
        features = extractor.extract_features(test_case["text"])
        all_found_features.update(features.keys())

        # Check for expected features
        found_expected = 0
        for expected_feature in test_case["expected_features"]:
            if expected_feature in features:
                value = features[expected_feature]
                print(f"   ✅ {expected_feature}: {value:.4f}")
                found_expected += 1
            else:
                print(f"   ❌ {expected_feature}: NOT FOUND")

        # Show other relevant new features
        print(f"   📊 Other new features in this text:")
        for feature_name in sorted(features.keys()):
            if (
                feature_name in expected_new_features
                and feature_name not in test_case["expected_features"]
                and features[feature_name] > 0
            ):
                print(f"      {feature_name}: {features[feature_name]:.4f}")

        success_rate = found_expected / len(test_case["expected_features"])
        print(
            f"   🎯 Success rate: {found_expected}/{len(test_case['expected_features'])} ({success_rate:.1%})"
        )

    # Summary of all new features
    print(f"\n" + "=" * 60)
    print("📊 NEW FEATURE SUMMARY")
    print("=" * 60)

    found_new_features = expected_new_features & all_found_features
    missing_features = expected_new_features - all_found_features

    print(
        f"✅ Found {len(found_new_features)}/{len(expected_new_features)} expected new features:"
    )
    for feature in sorted(found_new_features):
        print(f"   - {feature}")

    if missing_features:
        print(f"\n❌ Missing {len(missing_features)} expected features:")
        for feature in sorted(missing_features):
            print(f"   - {feature}")

    # Test with comprehensive example
    print(f"\n🧪 COMPREHENSIVE TEST")
    print("=" * 60)

    comprehensive_text = """
    The company may provide substantial returns, according to Q3 data showing $5.2 million revenue.
    However, we guarantee nothing and must comply with all regulations. Results are generally reasonable
    but we claim success only because independent audits verify our performance. The board has remediated
    previous misleading statements through corrective action.
    """

    features = extractor.extract_features(comprehensive_text.strip())

    print("All 6 new feature categories in comprehensive example:")

    categories = {
        "UCP (Uncautioned Claims)": [
            "lr_uncautioned_claim_proportion",
            "lr_uncautioned_claim_count",
        ],
        "DFSS (Documented Facts)": [
            "lr_documented_fact_sentence_share",
            "lr_documented_fact_sentence_count",
        ],
        "RAP (Remediation)": [
            "lr_remediation_action_proximity_count",
            "lr_remediation_action_proximity_norm",
        ],
        "Deontic Modality": [
            "ling_deontic_oblig_count",
            "ling_deontic_perm_count",
            "ling_deontic_balance_norm",
        ],
        "Ambiguity": ["lex_ambiguity_count", "lex_ambiguity_norm"],
        "Qualified Arguments": [
            "lr_qualified_argument_count",
            "lr_qualified_argument_norm",
        ],
    }

    for category, feature_list in categories.items():
        print(f"\n{category}:")
        for feature in feature_list:
            if feature in features:
                print(f"   {feature}: {features[feature]:.4f}")
            else:
                print(f"   {feature}: NOT FOUND")

    total_features = len(features)
    new_features_count = len([f for f in features.keys() if f in expected_new_features])

    print(f"\n🎉 EXTRACTION COMPLETE")
    print(f"   Total features extracted: {total_features}")
    print(
        f"   New features extracted: {new_features_count}/{len(expected_new_features)}"
    )
    print(f"   Success rate: {new_features_count/len(expected_new_features):.1%}")

    return features


if __name__ == "__main__":
    test_new_features()
