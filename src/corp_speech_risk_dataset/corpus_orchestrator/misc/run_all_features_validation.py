#!/usr/bin/env python3
"""
Run ALL Available Features Through Corrected Validation Pipeline

This script runs every available feature through the fixed validation pipeline,
excluding only labels, provenance, and weights.

Total Features to Test: ~674
"""

import sys
from pathlib import Path
import orjson
import pandas as pd
from loguru import logger

# Add path for unified pipeline
sys.path.insert(0, str(Path(__file__).parent))
from unified_binary_feature_pipeline import BinaryFeaturePipeline


def identify_testable_features(sample_file: Path, data_dir: Path) -> list:
    """Identify all testable features from a sample data file, respecting DNT manifest."""

    with open(sample_file, "rb") as f:
        sample_data = orjson.loads(f.readline())

    cols = sorted(sample_data.keys())

    # Load DNT manifest
    dnt_file = data_dir / "dnt_manifest.json"
    dnt_features = set()
    if dnt_file.exists():
        with open(dnt_file, "rb") as f:
            dnt_data = orjson.loads(f.read())
            dnt_features = set(dnt_data.get("do_not_train", []))
        logger.info(f"Loaded DNT manifest: {len(dnt_features)} excluded features")

    # Define exclusion patterns for labels, provenance, weights
    exclude_patterns = [
        # Labels/targets
        "outcome",
        "final_judgement",
        "liability",
        "certainty",
        "target",
        "label",
        "class",
        # Provenance/metadata
        "_metadata",
        "_src",
        "case_id",
        "court",
        "speaker",
        "venue",
        "timestamp",
        "date",
        "path",
        "file",
        "doc_id",
        # Weights
        "weight",
        "support",
        # System/internal
        "coral_",
        "bin_",
        "byte_",
        "_leakage",
        "created_at",
        "updated_at",
    ]

    # Get testable features
    testable = []
    excluded = []

    for col in cols:
        col_lower = col.lower()
        is_excluded = False

        # FIRST: Check DNT manifest (takes precedence)
        if col in dnt_features:
            excluded.append(col)
            is_excluded = True

        # Check exclusion patterns
        if not is_excluded:
            for pattern in exclude_patterns:
                if pattern in col_lower:
                    excluded.append(col)
                    is_excluded = True
                    break

        # Skip if starts with underscore (internal)
        if not is_excluded and col.startswith("_"):
            excluded.append(col)
            is_excluded = True

        if not is_excluded:
            testable.append(col)

    logger.info(f"Total columns: {len(cols)}")
    logger.info(f"Testable features: {len(testable)}")
    logger.info(f"Excluded features: {len(excluded)}")

    return testable


def main():
    """Run comprehensive validation on ALL available features."""

    # Paths - use the enhanced dataset with GraphSAGE
    data_dir = Path(
        "data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage"
    )
    output_dir = Path("results/all_features_with_graphsage_comprehensive_validation")
    fold = 4
    sample_size = 25000  # Larger sample for comprehensive test

    # Identify all testable features
    sample_file = data_dir / f"fold_{fold}" / "train.jsonl"
    testable_features = identify_testable_features(sample_file, data_dir)

    # Create custom pipeline to test ALL features (not just interpretable)
    pipeline = BinaryFeaturePipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        fold=fold,
        sample_size=sample_size,
        auto_update_governance=False,
    )

    logger.info(
        f"Starting comprehensive validation on {len(testable_features)} features..."
    )

    # Load data
    train_df, dev_df, test_df = pipeline.load_binary_data()

    # Filter to only testable features that exist in the data
    available_features = [f for f in testable_features if f in train_df.columns]
    logger.info(f"Features available in data: {len(available_features)}")

    # Log feature categories
    interpretable_count = len(
        [f for f in available_features if f.startswith("interpretable_")]
    )
    feat_count = len([f for f in available_features if f.startswith("feat_")])
    other_count = len(available_features) - interpretable_count - feat_count

    logger.info(f"Feature breakdown:")
    logger.info(f"  - Interpretable: {interpretable_count}")
    logger.info(f"  - Raw feat_: {feat_count}")
    logger.info(f"  - Other: {other_count}")

    # Run comprehensive tests on ALL features
    results = pipeline.run_comprehensive_tests(
        train_df, available_features, iteration=1
    )

    # Generate summary
    passed_features = [
        f for f, r in results["feature_results"].items() if r["overall_pass"]
    ]
    failed_features = [
        f for f, r in results["feature_results"].items() if not r["overall_pass"]
    ]

    logger.success(f"Comprehensive validation complete!")
    logger.success(f"Features PASSED: {len(passed_features)}/{len(available_features)}")
    logger.success(f"Pass rate: {len(passed_features)/len(available_features):.1%}")

    # Save detailed results
    with open(output_dir / "all_features_validation_results.json", "w") as f:
        # Simple JSON dump for complex results
        import json

        json.dump(results, f, indent=2, default=str)

    # Create summary report
    with open(output_dir / "ALL_FEATURES_SUMMARY.md", "w") as f:
        f.write("# Comprehensive All-Features Validation Results\n\n")
        f.write(f"**Total Features Tested**: {len(available_features)}\n")
        f.write(f"**Features Passed**: {len(passed_features)}\n")
        f.write(f"**Features Failed**: {len(failed_features)}\n")
        f.write(
            f"**Pass Rate**: {len(passed_features)/len(available_features):.1%}\n\n"
        )

        f.write("## Top Performing Features\n\n")

        # Sort by AUC
        feature_aucs = []
        for fname, result in results["feature_results"].items():
            if result["overall_pass"] and "discriminative" in result["tests"]:
                auc = result["tests"]["discriminative"]["metrics"].get("auc_mean", 0.5)
                feature_aucs.append((fname, auc))

        feature_aucs.sort(key=lambda x: x[1], reverse=True)

        f.write("| Rank | Feature | AUC |\n")
        f.write("|------|---------|-----|\n")
        for i, (feat, auc) in enumerate(feature_aucs[:50], 1):
            f.write(f"| {i} | `{feat}` | {auc:.3f} |\n")

    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
