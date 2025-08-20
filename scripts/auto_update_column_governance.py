#!/usr/bin/env python3
"""
Automatic Column Governance Updater

This script automatically updates column_governance.py based on feature test results.
It adds failed features to the BLOCKLIST_PATTERNS to prevent them from being used
in training while allowing continued feature extraction and testing.

Usage:
    python scripts/auto_update_column_governance.py \
        --test-results docs/feature_development/iteration_1/discriminative_power.csv \
        --iteration 1 \
        --apply-update

For review only (no changes):
    python scripts/auto_update_column_governance.py \
        --test-results docs/feature_development/iteration_1/discriminative_power.csv \
        --iteration 1
"""

import argparse
import pandas as pd
from pathlib import Path
import re
from typing import List, Dict


def load_test_results(results_file: str) -> pd.DataFrame:
    """Load feature test results."""
    return pd.read_csv(results_file)


def identify_failed_features(
    results_df: pd.DataFrame,
    mi_threshold: float = 0.005,
    p_threshold: float = 0.1,
    zero_threshold: float = 95.0,
    missing_threshold: float = 20.0,
) -> Dict[str, List[str]]:
    """Identify features that failed quality checks."""
    print("🔍 Identifying failed features...")

    failed_categories = {
        "weak_discrimination": [],
        "high_sparsity": [],
        "high_missingness": [],
        "size_biased": [],
        "potentially_leaky": [],
    }

    for _, row in results_df.iterrows():
        feature = row["feature"]

        # Weak discrimination
        if (
            row.get("mutual_info", 0) <= mi_threshold
            or row.get("kw_pvalue", 1) >= p_threshold
        ):
            failed_categories["weak_discrimination"].append(feature)

        # High sparsity
        if row.get("zero_pct", 0) >= zero_threshold:
            failed_categories["high_sparsity"].append(feature)

        # High missingness
        if row.get("missing_pct", 0) >= missing_threshold:
            failed_categories["high_missingness"].append(feature)

    # Load additional test results if available
    results_dir = Path(results_file).parent

    # Check size bias results
    size_bias_file = results_dir / "size_bias_check.csv"
    if size_bias_file.exists():
        size_df = pd.read_csv(size_bias_file)
        size_biased = size_df[size_df.get("size_bias_flag", False) == True]
        failed_categories["size_biased"].extend(size_biased["feature"].tolist())

    # Check leakage results
    leakage_file = results_dir / "leakage_check.csv"
    if leakage_file.exists():
        leak_df = pd.read_csv(leakage_file)
        leaky = leak_df[
            (leak_df.get("leakage_flag", False) == True)
            | (leak_df.get("court_leakage_flag", False) == True)
        ]
        failed_categories["potentially_leaky"].extend(leaky["feature"].tolist())

    # Print summary
    total_failed = sum(len(features) for features in failed_categories.values())
    print(f"📊 Failed feature summary:")
    for category, features in failed_categories.items():
        if features:
            print(f"  {category}: {len(features)} features")
    print(f"  Total failed: {total_failed}")

    return failed_categories


def generate_governance_patterns(
    failed_features: Dict[str, List[str]], iteration: str
) -> List[str]:
    """Generate regex patterns for column governance."""
    patterns = []
    patterns.append(f"    # Failed features from iteration {iteration}")

    for category, features in failed_features.items():
        if not features:
            continue

        patterns.append(f"    # {category.replace('_', ' ').title()}")
        for feature in features:
            escaped_feature = feature.replace("_", r"\_")
            patterns.append(f'    r"^{escaped_feature}$",  # {category}')

    return patterns


def update_column_governance(new_patterns: List[str], backup: bool = True) -> bool:
    """Update column_governance.py with new patterns."""
    governance_file = Path(
        "src/corp_speech_risk_dataset/fully_interpretable/column_governance.py"
    )

    if not governance_file.exists():
        print(f"❌ Column governance file not found: {governance_file}")
        return False

    # Create backup
    if backup:
        backup_file = governance_file.with_suffix(
            f'.py.backup.{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
        )
        with open(governance_file, "r") as f:
            content = f.read()
        with open(backup_file, "w") as f:
            f.write(content)
        print(f"📁 Backup created: {backup_file}")

    # Read current content
    with open(governance_file, "r") as f:
        content = f.read()

    # Find the BLOCKLIST_PATTERNS section
    patterns_start = content.find("BLOCKLIST_PATTERNS = [")
    if patterns_start == -1:
        print("❌ Could not find BLOCKLIST_PATTERNS in governance file")
        return False

    patterns_end = content.find("]", patterns_start)
    if patterns_end == -1:
        print("❌ Could not find end of BLOCKLIST_PATTERNS")
        return False

    # Insert new patterns before the closing bracket
    insert_point = content.rfind("\n", patterns_start, patterns_end)

    new_content = (
        content[:insert_point] + "\n" + "\n".join(new_patterns) + content[insert_point:]
    )

    # Write updated content
    with open(governance_file, "w") as f:
        f.write(new_content)

    pattern_prefix = 'r"'
    print(
        f"✅ Updated column governance with {len([p for p in new_patterns if p.strip().startswith(pattern_prefix)])} new patterns"
    )
    return True


def preview_governance_update(new_patterns: List[str]):
    """Preview what will be added to governance without applying."""
    print("\n📋 COLUMN GOVERNANCE UPDATE PREVIEW")
    print("=" * 60)
    print("The following patterns will be added to BLOCKLIST_PATTERNS:")
    print()
    for pattern in new_patterns:
        print(pattern)
    print()
    print("This will block these features from being used in training")
    print("but allows continued extraction for testing.")


def validate_governance_update():
    """Validate that the governance update was applied correctly."""
    print("🔍 Validating governance update...")

    try:
        # Try to import and validate
        sys.path.insert(0, "src")
        from corp_speech_risk_dataset.fully_interpretable.column_governance import (
            validate_columns,
        )

        # Test with some sample columns
        test_columns = [
            "interpretable_lex_deception_norm",  # Should be allowed
            "interpretable_lr_safe_harbor_norm",  # Should be allowed
            "blocked_feature_test",  # Should be blocked if we blocked it
            "case_id",  # Should be meta
        ]

        try:
            result = validate_columns(test_columns, allow_extra=True)
            print(f"✅ Governance validation passed")
            print(f"   Interpretable features: {len(result['interpretable_features'])}")
            return True
        except Exception as e:
            print(f"❌ Governance validation failed: {e}")
            return False

    except Exception as e:
        print(f"⚠️  Could not validate governance: {e}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Automatically update column governance based on test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--test-results",
        required=True,
        help="Path to discriminative power test results CSV",
    )
    parser.add_argument(
        "--iteration", required=True, help="Iteration identifier for labeling"
    )
    parser.add_argument(
        "--apply-update",
        action="store_true",
        help="Apply the update to column_governance.py (default: preview only)",
    )
    parser.add_argument(
        "--mi-threshold",
        type=float,
        default=0.005,
        help="Mutual information threshold (default: 0.005)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.1,
        help="P-value threshold (default: 0.1)",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=95.0,
        help="Zero percentage threshold (default: 95.0)",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=20.0,
        help="Missing percentage threshold (default: 20.0)",
    )

    args = parser.parse_args()

    print("🤖 AUTOMATIC COLUMN GOVERNANCE UPDATER")
    print("=" * 60)
    print(f"Test results: {args.test_results}")
    print(f"Iteration: {args.iteration}")
    print(f"Apply update: {args.apply_update}")
    print()

    try:
        # Load test results
        if not Path(args.test_results).exists():
            print(f"❌ Test results file not found: {args.test_results}")
            return 1

        results_df = load_test_results(args.test_results)
        print(f"✓ Loaded test results for {len(results_df)} features")

        # Identify failed features
        failed_features = identify_failed_features(
            results_df,
            args.mi_threshold,
            args.p_threshold,
            args.zero_threshold,
            args.missing_threshold,
        )

        # Generate governance patterns
        new_patterns = generate_governance_patterns(failed_features, args.iteration)

        if not new_patterns or len(new_patterns) <= 2:  # Only header
            print("✅ No features failed quality checks - no governance update needed!")
            return 0

        # Preview the update
        preview_governance_update(new_patterns)

        if args.apply_update:
            print("\n🔄 Applying governance update...")
            success = update_column_governance(new_patterns, backup=True)

            if success:
                # Validate the update
                validate_governance_update()
                print("\n✅ Column governance successfully updated!")
                print("📝 Failed features will now be blocked from training")
                print("📊 You can continue adding and testing new features")
            else:
                print("\n❌ Failed to update column governance")
                return 1
        else:
            print("\n👁️  PREVIEW ONLY - no changes applied")
            print("📝 Use --apply-update to make changes")
            print("💡 Or manually copy patterns to column_governance.py")

        return 0

    except Exception as e:
        print(f"❌ Update failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
