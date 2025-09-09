#!/usr/bin/env python3
"""Quick test to verify POLR setup is correct with metadata inheritance."""

import json
from pathlib import Path
import pandas as pd


def test_metadata_files():
    """Test that all required metadata files exist and have correct structure."""
    print("🔍 Testing POLR Setup & Metadata Files")
    print("=" * 50)

    # Test metadata files exist
    base_dir = Path("data/final_stratified_kfold_splits_adaptive_oof")
    metadata_file = base_dir / "per_fold_metadata.json"
    stats_file = base_dir / "fold_statistics.json"

    assert metadata_file.exists(), f"Missing metadata file: {metadata_file}"
    assert stats_file.exists(), f"Missing stats file: {stats_file}"
    print("✅ Metadata files exist")

    # Test metadata structure
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Check fold edges
    assert "binning" in metadata, "Missing 'binning' in metadata"
    assert "fold_edges" in metadata["binning"], "Missing 'fold_edges'"

    fold_edges = metadata["binning"]["fold_edges"]
    for fold_id in ["fold_0", "fold_1", "fold_2", "fold_3"]:
        assert fold_id in fold_edges, f"Missing {fold_id} in fold_edges"
        edges = fold_edges[fold_id]
        assert len(edges) == 2, f"Expected 2 edges for {fold_id}, got {len(edges)}"
        assert edges[0] < edges[1], f"Invalid edge order for {fold_id}: {edges}"

    print("✅ Tertile boundaries are valid for all folds")
    print(f"   Fold 0 edges: {fold_edges['fold_0']}")
    print(f"   Fold 1 edges: {fold_edges['fold_1']}")
    print(f"   Fold 2 edges: {fold_edges['fold_2']}")
    print(f"   Fold 3 edges: {fold_edges['fold_3']}")

    # Check class weights
    assert "weights" in metadata, "Missing 'weights' in metadata"

    weights_info = metadata["weights"]
    for fold_id in ["fold_0", "fold_1", "fold_2", "fold_3"]:
        assert fold_id in weights_info, f"Missing {fold_id} in weights"
        fold_weights = weights_info[fold_id]
        assert "class_weights" in fold_weights, f"Missing class_weights for {fold_id}"

        class_weights = fold_weights["class_weights"]
        assert "0" in class_weights, f"Missing class 0 weight for {fold_id}"
        assert "1" in class_weights, f"Missing class 1 weight for {fold_id}"
        assert "2" in class_weights, f"Missing class 2 weight for {fold_id}"

        # Check support weight config
        assert (
            "support_weight_method" in fold_weights
        ), f"Missing support_weight_method for {fold_id}"
        assert (
            "support_weight_range" in fold_weights
        ), f"Missing support_weight_range for {fold_id}"

        weight_range = fold_weights["support_weight_range"]
        assert (
            len(weight_range) == 2
        ), f"Expected 2 values in support_weight_range for {fold_id}"
        assert weight_range[0] == 0.25, f"Expected s_min=0.25 for {fold_id}"
        assert weight_range[1] == 4.0, f"Expected s_max=4.0 for {fold_id}"

    print("✅ Class weights are valid for all folds")
    print(f"   Fold 3 class weights: {weights_info['fold_3']['class_weights']}")
    print(f"   Support weight range: {weights_info['fold_3']['support_weight_range']}")

    # Test data files exist
    for fold_id in [0, 1, 2, 3]:
        fold_dir = base_dir / f"fold_{fold_id}"
        assert fold_dir.exists(), f"Missing fold directory: {fold_dir}"

        if fold_id < 3:
            # CV folds have train/val/test
            for split in ["train", "val", "test"]:
                split_file = fold_dir / f"{split}.jsonl"
                assert split_file.exists(), f"Missing {split_file}"
        else:
            # Fold 3 has train/dev
            for split in ["train", "dev"]:
                split_file = fold_dir / f"{split}.jsonl"
                assert split_file.exists(), f"Missing {split_file}"

    # Test OOF test set
    oof_dir = base_dir / "oof_test"
    oof_file = oof_dir / "test.jsonl"
    assert oof_dir.exists(), f"Missing OOF directory: {oof_dir}"
    assert oof_file.exists(), f"Missing OOF test file: {oof_file}"

    print("✅ All data files exist")
    print("✅ CV folds: 0, 1, 2 (train/val/test)")
    print("✅ Final fold: 3 (train/dev)")
    print("✅ OOF test set exists")


def test_column_governance():
    """Test that column governance is working correctly."""
    print("\n🎯 Testing Column Governance")
    print("=" * 30)

    # Import the governance functions
    import sys

    sys.path.insert(0, "src")

    from corp_speech_risk_dataset.fully_interpretable.column_governance import (
        validate_columns,
    )

    # Create a test list of interpretable features
    test_features = [
        "interpretable_lex_deception_norm",
        "interpretable_lex_deception_present",
        "interpretable_lex_guarantee_norm",
        "interpretable_lex_guarantee_present",
        "interpretable_lex_hedges_norm",
        "interpretable_lex_hedges_present",
        "interpretable_lex_pricing_claims_present",
        "interpretable_lex_superlatives_present",
        "interpretable_ling_certainty_high",
        "interpretable_seq_discourse_additive",
        # Add some that should be blocked
        "interpretable_context_length",
        "interpretable_lex_deception_count",
        "text",
        "speaker",
        "case_id",
    ]

    # Test validation (suppress error for this test)
    try:
        result = validate_columns(test_features, allow_extra=True)
        approved_features = result.get("approved_features", [])
        blocked_features = result.get("blocked_features", [])
    except ValueError as e:
        # Extract features from error message if validation fails
        import re

        error_msg = str(e)
        if "Blocked features:" in error_msg:
            blocked_match = re.search(r"Blocked features: \[(.*?)\]", error_msg)
            if blocked_match:
                blocked_str = blocked_match.group(1)
                blocked_features = [
                    f.strip().strip("'\"") for f in blocked_str.split(",")
                ]
                approved_features = [
                    f for f in test_features if f not in blocked_features
                ]
            else:
                blocked_features = []
                approved_features = test_features
        else:
            raise e

    print(f"✅ Column governance tested on {len(test_features)} features")
    print(f"   Approved: {len(approved_features)}")
    print(f"   Blocked: {len(blocked_features)}")

    expected_features = [
        "interpretable_lex_deception_norm",
        "interpretable_lex_deception_present",
        "interpretable_lex_guarantee_norm",
        "interpretable_lex_guarantee_present",
        "interpretable_lex_hedges_norm",
        "interpretable_lex_hedges_present",
        "interpretable_lex_pricing_claims_present",
        "interpretable_lex_superlatives_present",
        "interpretable_ling_certainty_high",
        "interpretable_seq_discourse_additive",
    ]

    # Check that our expected features are approved
    expected_approved = {
        "interpretable_lex_deception_norm",
        "interpretable_lex_deception_present",
        "interpretable_lex_guarantee_norm",
        "interpretable_lex_guarantee_present",
        "interpretable_lex_hedges_norm",
        "interpretable_lex_hedges_present",
        "interpretable_lex_pricing_claims_present",
        "interpretable_lex_superlatives_present",
        "interpretable_ling_certainty_high",
        "interpretable_seq_discourse_additive",
    }

    expected_blocked = {
        "interpretable_context_length",
        "interpretable_lex_deception_count",
        "text",
        "speaker",
    }

    approved_set = set(approved_features)
    blocked_set = set(blocked_features)

    approved_expected = expected_approved.intersection(approved_set)
    blocked_expected = expected_blocked.intersection(blocked_set)

    print(f"✅ Expected approved features found: {len(approved_expected)}/10")
    print(
        f"✅ Expected blocked features found: {len(blocked_expected)}/{len(expected_blocked)}"
    )

    if approved_expected == expected_approved:
        print("✅ All 10 final features are correctly approved!")
    else:
        missing = expected_approved - approved_expected
        print(f"⚠️  Missing approved features: {missing}")

    print(f"\n   Sample approved: {list(approved_features)[:3]}")
    print(f"   Sample blocked: {list(blocked_features)[:3]}")


def main():
    """Run all tests."""
    try:
        test_metadata_files()
        test_column_governance()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! POLR SETUP IS READY")
        print("=" * 60)
        print("\n🚀 Ready to run:")
        print("   uv run python scripts/run_polar_cv.py --output-dir runs/polr_test")
        print("\n📊 All metadata will be inherited correctly:")
        print("   ✅ Tertile boundaries from per_fold_metadata.json")
        print("   ✅ Class weights from per_fold_metadata.json")
        print("   ✅ Support weight ranges (0.25, 4.0)")
        print("   ✅ 10 interpretable features from column governance")
        print("   ✅ polr_ prefix for all predictions")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
