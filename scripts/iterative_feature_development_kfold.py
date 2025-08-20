#!/usr/bin/env python3
"""
Iterative Feature Development Pipeline - K-Fold Version

This version uses the pre-stratified k-fold data which already has outcome_bin labels
and balanced class distribution. It adds new features to the existing data and tests them.

Usage:
    python scripts/iterative_feature_development_kfold.py \
        --iteration test_kfold_1 \
        --fold-dir data/final_stratified_kfold_splits_authoritative \
        --fold 3 \
        --sample-size 10000 \
        --test-class-discrimination

This avoids the binning issues and uses authoritative labels.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


def load_kfold_sample_data(
    fold_dir: str, fold: int, sample_size: int, split: str = "train"
) -> pd.DataFrame:
    """Load sample data from k-fold splits with balanced classes."""
    print(
        f"📊 Loading k-fold sample data (fold {fold}, up to {sample_size:,} records)..."
    )

    fold_path = Path(fold_dir) / f"fold_{fold}" / f"{split}.jsonl"
    if not fold_path.exists():
        raise ValueError(f"K-fold file not found: {fold_path}")

    records = []
    target_per_class = sample_size // 3  # Aim for balanced sampling
    class_counts = {0: 0, 1: 0, 2: 0}

    with open(fold_path, "r") as f:
        for line in f:
            if len(records) >= sample_size:
                break

            try:
                record = json.loads(line)
                outcome_bin = record.get("outcome_bin")

                if outcome_bin is not None and outcome_bin in class_counts:
                    # Sample more evenly across classes
                    if (
                        class_counts[outcome_bin] < target_per_class * 1.5
                    ):  # Allow some skew
                        records.append(record)
                        class_counts[outcome_bin] += 1

            except:
                continue

    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df):,} records from fold {fold}")
    print(f"  Class distribution: {dict(class_counts)}")

    return df


def extract_and_enhance_features(
    df: pd.DataFrame, iteration: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add new features to existing k-fold data and test them."""
    print(f"🔍 Adding new features to k-fold data for iteration {iteration}...")

    # Check what interpretable features are already present
    existing_features = [col for col in df.columns if col.startswith("interpretable_")]
    print(f"✓ Found {len(existing_features)} existing interpretable features")

    # Extract new features for sample of records
    extractor = InterpretableFeatureExtractor()

    enhanced_records = []
    for _, record in df.iterrows():
        text = record.get("text", "")
        context = record.get("context", "")

        try:
            # Extract ALL features (old + new)
            new_features = extractor.extract_features(text, context)

            # Create enhanced record
            enhanced_record = record.to_dict()

            # Add NEW features with prefix (skip if already present)
            for feature_name, feature_value in new_features.items():
                prefixed_name = f"interpretable_{feature_name}"
                if prefixed_name not in enhanced_record:  # Only add new ones
                    enhanced_record[prefixed_name] = float(feature_value)

            enhanced_records.append(enhanced_record)

        except Exception as e:
            print(f"Warning: Feature extraction failed for record: {e}")
            enhanced_records.append(record.to_dict())

    # Convert back to DataFrame
    enhanced_df = pd.DataFrame(enhanced_records)

    # Get all interpretable feature columns (old + new)
    all_features = [
        col for col in enhanced_df.columns if col.startswith("interpretable_")
    ]
    new_features = [col for col in all_features if col not in existing_features]

    print(f"✓ Total interpretable features: {len(all_features)}")
    print(f"✓ New features added: {len(new_features)}")

    if new_features:
        print(f"  New features include: {new_features[:5]}...")

    # Run comprehensive testing on ALL features
    test_results = run_comprehensive_feature_tests(
        enhanced_df, all_features, new_features
    )

    return enhanced_df, test_results


def run_comprehensive_feature_tests(
    df: pd.DataFrame, all_features: List[str], new_features: List[str]
) -> Dict[str, Any]:
    """Run all feature quality tests with focus on new features."""
    print("🧪 Running comprehensive feature tests...")

    results = {
        "discriminative_power": [],
        "class_0_discrimination": [],
        "size_bias_check": [],
        "leakage_check": [],
        "new_feature_focus": [],
        "summary": {},
    }

    target_col = "outcome_bin"
    if target_col not in df.columns:
        print(f"❌ No outcome_bin column found")
        return results

    # Add case size for bias testing
    if "case_size" not in df.columns and "case_id" in df.columns:
        df["case_size"] = df.groupby("case_id")["case_id"].transform("count")

    # Test all features
    for feature in all_features:
        if feature not in df.columns:
            continue

        feature_data = df[feature].dropna()
        if len(feature_data) == 0 or feature_data.std() == 0:
            continue

        is_new_feature = feature in new_features

        # 1. Discriminative power
        disc_result = test_discriminative_power(df, feature, target_col)
        disc_result["is_new_feature"] = is_new_feature
        results["discriminative_power"].append(disc_result)

        # 2. Class 0 discrimination specifically
        class0_result = test_class_0_discrimination(df, feature, target_col)
        class0_result["is_new_feature"] = is_new_feature
        results["class_0_discrimination"].append(class0_result)

        # 3. Size bias check
        if "case_size" in df.columns:
            size_result = test_size_bias(df, feature)
            size_result["is_new_feature"] = is_new_feature
            results["size_bias_check"].append(size_result)

        # 4. Basic leakage check
        leakage_result = test_basic_leakage(df, feature, target_col)
        leakage_result["is_new_feature"] = is_new_feature
        results["leakage_check"].append(leakage_result)

        # 5. Track new features separately
        if is_new_feature:
            new_feature_result = {
                "feature": feature,
                "discriminative_power": disc_result.get("mutual_info", 0),
                "class_0_auc": class0_result.get("class_0_auc", 0),
                "size_bias_flag": (
                    size_result.get("size_bias_flag", False)
                    if "case_size" in df.columns
                    else False
                ),
                "leakage_flag": leakage_result.get("leakage_flag", False),
            }
            results["new_feature_focus"].append(new_feature_result)

    # Generate summary focusing on new features
    disc_df = pd.DataFrame(results["discriminative_power"])
    class0_df = pd.DataFrame(results["class_0_discrimination"])
    new_df = pd.DataFrame(results["new_feature_focus"])

    if len(disc_df) > 0:
        # All features passing tests
        passing_features = disc_df[
            (disc_df["mutual_info"] > 0.005)
            & (disc_df["kw_pvalue"] < 0.1)
            & (disc_df["zero_pct"] < 95)
            & (disc_df["missing_pct"] < 20)
        ]

        # New features passing tests
        new_passing = passing_features[passing_features["is_new_feature"] == True]

        # Features good for class 0
        class0_good = class0_df[
            (class0_df["class_0_separation"] > 0.1)
            & (class0_df["class_0_significance"] < 0.05)
        ]

        # New features good for class 0
        new_class0_good = class0_good[class0_good["is_new_feature"] == True]

        results["summary"] = {
            "total_features": len(disc_df),
            "new_features": len(new_df),
            "passing_all_tests": len(passing_features),
            "new_features_passing": len(new_passing),
            "good_for_class_0": len(class0_good),
            "new_good_for_class_0": len(new_class0_good),
            "new_class_0_discriminators": new_class0_good["feature"].tolist(),
            "top_new_features": (
                new_df.nlargest(10, "discriminative_power")["feature"].tolist()
                if len(new_df) > 0
                else []
            ),
            "new_features_with_issues": (
                new_df[
                    (new_df["size_bias_flag"] == True)
                    | (new_df["leakage_flag"] == True)
                ]["feature"].tolist()
                if len(new_df) > 0
                else []
            ),
        }

    return results


def test_discriminative_power(
    df: pd.DataFrame, feature: str, target_col: str
) -> Dict[str, Any]:
    """Test discriminative power of a feature."""
    result = {"feature": feature}

    try:
        data = df[[feature, target_col]].dropna()
        if len(data) == 0:
            return result

        X = data[feature].values
        y = data[target_col].values

        # Basic stats
        result["mean"] = X.mean()
        result["std"] = X.std()
        result["zero_pct"] = (X == 0).mean() * 100
        result["missing_pct"] = df[feature].isna().mean() * 100

        # Mutual information
        if X.std() > 0:
            X_binned = pd.qcut(
                X, q=min(5, len(np.unique(X))), duplicates="drop", labels=False
            )
            result["mutual_info"] = mutual_info_score(y, X_binned)
        else:
            result["mutual_info"] = 0

        # Kruskal-Wallis test
        groups = [X[y == k] for k in sorted(np.unique(y))]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            kw_stat, kw_p = kruskal(*groups)
            result["kw_statistic"] = kw_stat
            result["kw_pvalue"] = kw_p
        else:
            result["kw_pvalue"] = 1.0

        # Effect sizes (Cliff's delta)
        if len(groups) >= 2:
            result["cliff_delta_01"] = cliffs_delta(groups[0], groups[1])
        if len(groups) >= 3:
            result["cliff_delta_12"] = cliffs_delta(groups[1], groups[2])

    except Exception as e:
        print(f"  Warning: Discriminative power test failed for {feature}: {e}")

    return result


def test_class_0_discrimination(
    df: pd.DataFrame, feature: str, target_col: str
) -> Dict[str, Any]:
    """Specifically test if feature can discriminate class 0 (low-risk)."""
    result = {"feature": feature}

    try:
        data = df[[feature, target_col]].dropna()
        if len(data) == 0:
            return result

        X = data[feature].values
        y = data[target_col].values

        # Class means
        class_means = {}
        for k in sorted(np.unique(y)):
            class_data = X[y == k]
            if len(class_data) > 0:
                class_means[k] = class_data.mean()

        result["class_means"] = class_means

        # Class 0 vs others separation
        if 0 in class_means and len(class_means) > 1:
            class_0_mean = class_means[0]
            other_means = [v for k, v in class_means.items() if k != 0]

            # Separation from other classes
            separations = [abs(class_0_mean - om) for om in other_means]
            result["class_0_separation"] = max(separations) if separations else 0

            # Statistical significance for class 0 vs others
            class_0_data = X[y == 0]
            other_data = X[y != 0]

            if len(class_0_data) > 1 and len(other_data) > 1:
                from scipy.stats import mannwhitneyu

                _, p_value = mannwhitneyu(
                    class_0_data, other_data, alternative="two-sided"
                )
                result["class_0_significance"] = p_value

            # Predictive power for class 0
            if len(np.unique(y)) >= 2:
                y_binary = (y == 0).astype(int)
                if len(np.unique(y_binary)) == 2 and X.std() > 0:
                    auc = roc_auc_score(y_binary, X)
                    result["class_0_auc"] = max(
                        auc, 1 - auc
                    )  # Handle inverse relationships

    except Exception as e:
        print(f"  Warning: Class 0 test failed for {feature}: {e}")

    return result


def test_size_bias(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """Test for case size bias."""
    result = {"feature": feature}

    try:
        data = df[[feature, "case_size"]].dropna()
        if len(data) == 0:
            return result

        # Correlation with case size
        corr_size, p_value = spearmanr(data[feature], data["case_size"])
        result["corr_case_size"] = corr_size
        result["size_bias_pvalue"] = p_value
        result["size_bias_flag"] = abs(corr_size) > 0.3 and p_value < 0.05

    except Exception as e:
        print(f"  Warning: Size bias test failed for {feature}: {e}")

    return result


def test_basic_leakage(
    df: pd.DataFrame, feature: str, target_col: str
) -> Dict[str, Any]:
    """Test for basic information leakage."""
    result = {"feature": feature}

    try:
        # Check direct correlation with continuous outcome
        if "final_judgement_real" in df.columns:
            data = df[[feature, "final_judgement_real"]].dropna()
            if len(data) > 10 and data[feature].std() > 0:
                corr, p_value = spearmanr(data[feature], data["final_judgement_real"])
                result["outcome_correlation"] = corr
                result["leakage_flag"] = abs(corr) > 0.8

        # Check if feature might encode court/venue (if available)
        court_cols = [
            col
            for col in df.columns
            if any(x in col.lower() for x in ["court", "venue", "district"])
        ]
        if court_cols:
            court_col = court_cols[0]
            court_data = df[[feature, court_col]].dropna()
            if len(court_data) > 20:
                # High variance across courts suggests court encoding
                court_var = court_data.groupby(court_col)[feature].mean().var()
                total_var = court_data[feature].var()
                result["court_variance_ratio"] = court_var / (total_var + 1e-8)
                result["court_leakage_flag"] = result["court_variance_ratio"] > 0.3

    except Exception as e:
        print(f"  Warning: Leakage test failed for {feature}: {e}")

    return result


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta effect size."""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0

    greater = sum(x > y for x in group1 for y in group2)
    lesser = sum(x < y for x in group1 for y in group2)

    return (greater - lesser) / (len(group1) * len(group2))


def save_iteration_results(
    results: Dict[str, Any], enhanced_df: pd.DataFrame, iteration: str, output_dir: Path
):
    """Save comprehensive results for an iteration."""
    print(f"💾 Saving iteration {iteration} results...")

    iter_dir = output_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Save test results
    for test_name, test_data in results.items():
        if isinstance(test_data, list) and test_data:
            df = pd.DataFrame(test_data)
            df.to_csv(iter_dir / f"{test_name}.csv", index=False)

    # Save summary
    with open(iter_dir / "summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2)

    # Save sample of enhanced data
    sample_df = enhanced_df.head(100)
    sample_df.to_json(
        iter_dir / "sample_enhanced_data.jsonl", orient="records", lines=True
    )

    # Create readable summary report
    summary_lines = []
    summary_lines.append(f"# K-Fold Feature Development Iteration {iteration}")
    summary_lines.append(
        f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    summary_lines.append("")

    if "summary" in results:
        summary = results["summary"]
        summary_lines.append("## Summary")
        summary_lines.append(
            f"- Total features tested: {summary.get('total_features', 0)}"
        )
        summary_lines.append(f"- New features added: {summary.get('new_features', 0)}")
        summary_lines.append(
            f"- New features passing tests: {summary.get('new_features_passing', 0)}"
        )
        summary_lines.append(
            f"- New features good for class 0: {summary.get('new_good_for_class_0', 0)}"
        )
        summary_lines.append("")

        if summary.get("new_class_0_discriminators"):
            summary_lines.append("## NEW Class 0 Discriminators")
            for feature in summary["new_class_0_discriminators"]:
                summary_lines.append(f"- {feature}")
            summary_lines.append("")

        if summary.get("top_new_features"):
            summary_lines.append("## Top NEW Features by Discriminative Power")
            for feature in summary["top_new_features"]:
                summary_lines.append(f"- {feature}")
            summary_lines.append("")

        if summary.get("new_features_with_issues"):
            summary_lines.append("## NEW Features with Issues (Size Bias/Leakage)")
            for feature in summary["new_features_with_issues"]:
                summary_lines.append(f"- {feature}")

    with open(iter_dir / f"iteration_{iteration}_summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"✓ Results saved to: {iter_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Iterative feature development using k-fold data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--iteration", required=True, help="Iteration identifier")
    parser.add_argument("--fold-dir", required=True, help="K-fold directory path")
    parser.add_argument(
        "--fold", type=int, default=3, help="Which fold to use (default: 3)"
    )
    parser.add_argument(
        "--split", default="train", help="Which split to use (train/dev)"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/feature_development_kfold",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10000, help="Sample size for testing"
    )
    parser.add_argument(
        "--test-class-discrimination",
        action="store_true",
        help="Run detailed class discrimination analysis",
    )

    args = parser.parse_args()

    print("🚀 ITERATIVE FEATURE DEVELOPMENT - K-FOLD VERSION")
    print("=" * 60)
    print(f"Iteration: {args.iteration}")
    print(f"K-fold directory: {args.fold_dir}")
    print(f"Using fold: {args.fold}")
    print(f"Sample size: {args.sample_size:,}")
    print()

    try:
        # Load k-fold sample data
        df = load_kfold_sample_data(
            args.fold_dir, args.fold, args.sample_size, args.split
        )

        # Add new features and test
        enhanced_df, test_results = extract_and_enhance_features(df, args.iteration)

        # Save results
        output_dir = Path(args.output_dir)
        save_iteration_results(test_results, enhanced_df, args.iteration, output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("ITERATION RESULTS")
        print("=" * 60)

        if "summary" in test_results:
            summary = test_results["summary"]
            print(f"✅ Total features tested: {summary.get('total_features', 0)}")
            print(f"🆕 New features added: {summary.get('new_features', 0)}")
            print(
                f"✅ New features passing tests: {summary.get('new_features_passing', 0)}"
            )
            print(
                f"🎯 New Class 0 discriminators: {summary.get('new_good_for_class_0', 0)}"
            )

            if summary.get("new_class_0_discriminators"):
                print("\n🏆 NEW FEATURES THAT HELP WITH CLASS 0:")
                for feature in summary["new_class_0_discriminators"]:
                    print(f"   - {feature}")

            if summary.get("top_new_features"):
                print(f"\n📊 TOP NEW FEATURES BY DISCRIMINATIVE POWER:")
                for feature in summary["top_new_features"][:5]:
                    print(f"   - {feature}")

            if summary.get("new_features_with_issues"):
                print(f"\n⚠️  NEW FEATURES WITH ISSUES:")
                for feature in summary["new_features_with_issues"]:
                    print(f"   - {feature}")

        print(f"\n📁 Results saved to: {output_dir}/iteration_{args.iteration}/")

        return 0

    except Exception as e:
        print(f"❌ Iteration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
