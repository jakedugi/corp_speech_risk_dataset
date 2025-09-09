#!/usr/bin/env python3
"""
Test script to verify the unified feature extraction pipeline works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


def test_feature_extraction():
    """Test that feature extraction works with derived features."""
    print("Testing unified feature extraction...")

    # Create extractor
    extractor = InterpretableFeatureExtractor()

    # Test text
    test_text = """
    We guarantee that our product will absolutely save you money!
    This is the best deal you'll ever find. However, results may vary
    and we cannot be held responsible for any losses. Free shipping included!
    """

    test_context = "Corporate marketing statement from Company X"

    # Extract features
    features = extractor.extract_features(test_text, test_context)

    print(f"\nExtracted {len(features)} features")

    # Check for base features
    base_features = [
        f for f in features if not any(x in f for x in ["ratio", "interact"])
    ]
    print(f"Base features: {len(base_features)}")

    # Check for derived features
    derived_features = [
        f for f in features if any(x in f for x in ["ratio", "interact"])
    ]
    print(f"Derived features: {len(derived_features)}")

    # Print some example features
    print("\nExample base features:")
    for feature in ["lex_guarantee_norm", "lex_hedges_present", "ling_high_certainty"]:
        if feature in features:
            print(f"  {feature}: {features[feature]:.4f}")

    print("\nExample derived features:")
    for feature in ["ratio_guarantee_vs_hedge", "interact_guarantee_x_cert"]:
        if feature in features:
            print(f"  {feature}: {features[feature]:.4f}")

    # Verify expected features exist
    expected_derived = [
        "ratio_guarantee_vs_hedge",
        "ratio_deception_vs_hedge",
        "ratio_guarantee_vs_superlative",
        "interact_guarantee_x_cert",
        "interact_superlative_x_cert",
        "interact_hedge_x_guarantee",
    ]

    missing = [f for f in expected_derived if f not in features]
    if missing:
        print(f"\n⚠️  Missing expected derived features: {missing}")
        return False
    else:
        print("\n✅ All expected derived features present!")
        return True


def test_feature_consistency():
    """Test that features are consistent across multiple extractions."""
    print("\n\nTesting feature consistency...")

    extractor = InterpretableFeatureExtractor()

    # Extract features multiple times
    text = "We guarantee the best results with no risk!"

    features1 = extractor.extract_features(text)
    features2 = extractor.extract_features(text)

    # Check consistency
    if features1 == features2:
        print("✅ Features are consistent across extractions")
        return True
    else:
        print("❌ Features differ across extractions!")
        for key in features1:
            if features1[key] != features2.get(key):
                print(f"  {key}: {features1[key]} vs {features2.get(key)}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("UNIFIED FEATURE PIPELINE TEST")
    print("=" * 60)

    tests_passed = 0
    tests_total = 2

    if test_feature_extraction():
        tests_passed += 1

    if test_feature_consistency():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)

    if tests_passed == tests_total:
        print("\n✅ All tests passed! Pipeline is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
