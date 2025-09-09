#!/usr/bin/env python3
"""
Iterative Feature Development Pipeline

This script provides a rapid iteration framework for developing and testing
interpretable features. It automates:

1. Feature extraction and testing
2. Class discrimination analysis (especially class 0)
3. Quality checks (size bias, leakage, temporal stability)
4. Automatic column governance updates
5. Feature performance tracking across iterations

Usage:
    python scripts/iterative_feature_development.py \
        --iteration 1 \
        --input "data/raw/doc_*_text_stage15.jsonl" \
        --sample-size 10000 \
        --test-class-discrimination

For rapid testing of specific features:
    python scripts/iterative_feature_development.py \
        --iteration test \
        --input "data/raw/doc_1000001_text_stage15.jsonl" \
        --sample-size 1000 \
        --quick-test
"""

import argparse
import json
import glob
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


def load_sample_data(file_pattern: str, sample_size: int) -> pd.DataFrame:
    """Load sample data for testing."""
    print(f"📊 Loading sample data (up to {sample_size:,} records)...")

    files = sorted(glob.glob(file_pattern))
    if not files:
        raise ValueError(f"No files found matching: {file_pattern}")

    records = []
    for file_path in files[:3]:  # Limit to first 3 files for speed
        if len(records) >= sample_size:
            break

        with open(file_path, "r") as f:
            for line in f:
                if len(records) >= sample_size:
                    break
                try:
                    records.append(json.loads(line))
                except:
                    continue

    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df):,} records from {len(files)} files")

    return df


def extract_and_test_features(
    df: pd.DataFrame, iteration: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract features and perform comprehensive testing."""
    print(f"🔍 Extracting and testing features for iteration {iteration}...")

    # Extract features for sample
    extractor = InterpretableFeatureExtractor()

    # Process sample of records to get feature matrix
    sample_records = df.head(min(5000, len(df))).to_dict("records")

    enhanced_records = []
    for record in sample_records:
        text = record.get("text", "")
        context = record.get("context", "")

        try:
            features = extractor.extract_features(text, context)
            enhanced_record = record.copy()

            # Add features with prefix
            for feature_name, feature_value in features.items():
                enhanced_record[f"interpretable_{feature_name}"] = float(feature_value)

            enhanced_records.append(enhanced_record)
        except Exception as e:
            print(f"Warning: Feature extraction failed for record: {e}")
            enhanced_records.append(record)

    # Convert back to DataFrame
    enhanced_df = pd.DataFrame(enhanced_records)

    # Get interpretable feature columns
    feature_cols = [
        col for col in enhanced_df.columns if col.startswith("interpretable_")
    ]
    print(f"✓ Extracted {len(feature_cols)} features")

    # Create target if not present
    if (
        "outcome_bin" not in enhanced_df.columns
        and "final_judgement_real" in enhanced_df.columns
    ):
        values = enhanced_df["final_judgement_real"].values
        enhanced_df["outcome_bin"] = pd.qcut(values, q=3, labels=[0, 1, 2]).astype(int)

    # Run comprehensive testing
    test_results = run_comprehensive_feature_tests(enhanced_df, feature_cols)

    return enhanced_df, test_results


def run_comprehensive_feature_tests(
    df: pd.DataFrame, features: List[str]
) -> Dict[str, Any]:
    """Run all feature quality tests."""
    print("🧪 Running comprehensive feature tests...")

    results = {
        "discriminative_power": [],
        "class_0_discrimination": [],
        "size_bias_check": [],
        "leakage_check": [],
        "stability_check": [],
        "summary": {},
    }

    target_col = "outcome_bin" if "outcome_bin" in df.columns else "y"
    if target_col not in df.columns:
        print("⚠️  No target column found, skipping discrimination tests")
        return results

    # Add case size if not present
    if "case_size" not in df.columns and "case_id" in df.columns:
        df["case_size"] = df.groupby("case_id")["case_id"].transform("count")

    for feature in features:
        if feature not in df.columns:
            continue

        feature_data = df[feature].dropna()
        if len(feature_data) == 0 or feature_data.std() == 0:
            continue

        # 1. Discriminative power
        disc_result = test_discriminative_power(df, feature, target_col)
        results["discriminative_power"].append(disc_result)

        # 2. Class 0 discrimination specifically
        class0_result = test_class_0_discrimination(df, feature, target_col)
        results["class_0_discrimination"].append(class0_result)

        # 3. Size bias check
        if "case_size" in df.columns:
            size_result = test_size_bias(df, feature)
            results["size_bias_check"].append(size_result)

        # 4. Basic leakage check
        leakage_result = test_basic_leakage(df, feature, target_col)
        results["leakage_check"].append(leakage_result)

    # Generate summary
    disc_df = pd.DataFrame(results["discriminative_power"])
    class0_df = pd.DataFrame(results["class_0_discrimination"])

    if len(disc_df) > 0:
        # Features passing all tests
        passing_features = disc_df[
            (disc_df["mutual_info"] > 0.005)
            & (disc_df["kw_pvalue"] < 0.1)
            & (disc_df["zero_pct"] < 95)
            & (disc_df["missing_pct"] < 20)
        ]

        # Features good for class 0
        class0_good = class0_df[
            (class0_df["class_0_separation"] > 0.1)
            & (class0_df["class_0_significance"] < 0.05)
        ]

        results["summary"] = {
            "total_features": len(disc_df),
            "passing_all_tests": len(passing_features),
            "good_for_class_0": len(class0_good),
            "class_0_discriminators": class0_good["feature"].tolist(),
            "top_mi_features": disc_df.nlargest(10, "mutual_info")["feature"].tolist(),
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
            X_binned = pd.qcut(X, q=5, duplicates="drop", labels=False)
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

        # Partial correlation with outcome given case size
        if "outcome_bin" in df.columns:
            full_data = df[[feature, "case_size", "outcome_bin"]].dropna()
            if len(full_data) > 10:
                # Residualize feature against case_size
                reg = LinearRegression()
                reg.fit(full_data[["case_size"]], full_data[feature])
                feature_resid = full_data[feature] - reg.predict(
                    full_data[["case_size"]]
                )

                if feature_resid.std() > 0:
                    partial_corr, _ = spearmanr(feature_resid, full_data["outcome_bin"])
                    result["partial_corr_outcome"] = partial_corr

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


def analyze_class_discrimination(
    df: pd.DataFrame, features: List[str], target_col: str
) -> pd.DataFrame:
    """Detailed analysis of class discrimination capabilities."""
    print("🎯 Analyzing class discrimination capabilities...")

    results = []

    for feature in features:
        if feature not in df.columns:
            continue

        try:
            data = df[[feature, target_col]].dropna()
            if len(data) == 0 or data[feature].std() == 0:
                continue

            X = data[feature].values
            y = data[target_col].values

            result = {"feature": feature}

            # Per-class statistics
            for class_val in sorted(np.unique(y)):
                class_data = X[y == class_val]
                if len(class_data) > 0:
                    result[f"class_{class_val}_mean"] = class_data.mean()
                    result[f"class_{class_val}_std"] = class_data.std()
                    result[f"class_{class_val}_n"] = len(class_data)

            # Overall discrimination metrics
            if len(np.unique(y)) >= 3:
                # One-vs-rest AUC for each class
                auc_scores = []
                for class_val in np.unique(y):
                    y_binary = (y == class_val).astype(int)
                    if len(np.unique(y_binary)) == 2:
                        auc = roc_auc_score(y_binary, X)
                        auc_scores.append(max(auc, 1 - auc))
                        result[f"auc_class_{class_val}"] = max(auc, 1 - auc)

                result["macro_auc"] = np.mean(auc_scores)

            # Kruskal-Wallis for overall discrimination
            groups = [X[y == k] for k in sorted(np.unique(y))]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 2:
                kw_stat, kw_p = kruskal(*groups)
                result["kw_statistic"] = kw_stat
                result["kw_pvalue"] = kw_p

            # Class 0 specific metrics
            if 0 in np.unique(y):
                class_0_mask = y == 0
                class_non0_mask = y != 0

                if class_0_mask.sum() > 0 and class_non0_mask.sum() > 0:
                    from scipy.stats import mannwhitneyu

                    _, mw_p = mannwhitneyu(
                        X[class_0_mask], X[class_non0_mask], alternative="two-sided"
                    )
                    result["class_0_vs_others_pvalue"] = mw_p

                    # Effect size for class 0
                    pooled_std = np.sqrt(
                        (X[class_0_mask].var() + X[class_non0_mask].var()) / 2
                    )
                    if pooled_std > 0:
                        cohen_d = (
                            X[class_0_mask].mean() - X[class_non0_mask].mean()
                        ) / pooled_std
                        result["class_0_effect_size"] = abs(cohen_d)

            results.append(result)

        except Exception as e:
            print(f"  Warning: Class discrimination test failed for {feature}: {e}")

    return pd.DataFrame(results)


def generate_governance_update(
    failed_features: List[str],
    reasons: Dict[str, str],
    iteration: str,
    output_dir: Path,
):
    """Generate column governance update for failed features."""
    print(f"📝 Generating column governance update for iteration {iteration}...")

    governance_file = Path(
        "src/corp_speech_risk_dataset/fully_interpretable/column_governance.py"
    )

    # Read current governance
    with open(governance_file, "r") as f:
        content = f.read()

    # Generate new patterns
    new_patterns = []
    new_patterns.append(f"    # Failed features from iteration {iteration}")

    for feature in failed_features:
        reason = reasons.get(feature, "failed_quality_checks")
        escaped_feature = feature.replace("_", r"\_")
        new_patterns.append(f'    r"^{escaped_feature}$",  # {reason}')

    # Create update file (don't modify governance automatically)
    update_content = []
    update_content.append(f"# Column Governance Update - Iteration {iteration}")
    update_content.append(
        f"# Add these patterns to BLOCKLIST_PATTERNS in column_governance.py"
    )
    update_content.append("")
    update_content.extend(new_patterns)

    update_file = output_dir / f"governance_update_iteration_{iteration}.txt"
    with open(update_file, "w") as f:
        f.write("\n".join(update_content))

    print(f"✓ Governance update saved to: {update_file}")
    print(f"  Add {len(failed_features)} patterns to column_governance.py")


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
    summary_lines.append(f"# Feature Development Iteration {iteration}")
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
        summary_lines.append(
            f"- Features passing all tests: {summary.get('passing_all_tests', 0)}"
        )
        summary_lines.append(
            f"- Features good for class 0: {summary.get('good_for_class_0', 0)}"
        )
        summary_lines.append("")

        if summary.get("class_0_discriminators"):
            summary_lines.append("## Class 0 Discriminators")
            for feature in summary["class_0_discriminators"]:
                summary_lines.append(f"- {feature}")
            summary_lines.append("")

        if summary.get("top_mi_features"):
            summary_lines.append("## Top MI Features")
            for feature in summary["top_mi_features"]:
                summary_lines.append(f"- {feature}")

    with open(iter_dir / f"iteration_{iteration}_summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"✓ Results saved to: {iter_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Iterative feature development and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--iteration", required=True, help="Iteration identifier")
    parser.add_argument("--input", required=True, help="Input file pattern")
    parser.add_argument(
        "--output-dir",
        default="docs/feature_development",
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
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick test with minimal analysis"
    )
    parser.add_argument(
        "--auto-governance-update",
        action="store_true",
        help="Automatically generate governance updates",
    )

    args = parser.parse_args()

    print("🚀 ITERATIVE FEATURE DEVELOPMENT")
    print("=" * 60)
    print(f"Iteration: {args.iteration}")
    print(f"Input pattern: {args.input}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Quick test: {args.quick_test}")
    print()

    try:
        # Load sample data
        df = load_sample_data(args.input, args.sample_size)

        # Extract and test features
        enhanced_df, test_results = extract_and_test_features(df, args.iteration)

        # Run additional analysis if requested
        if args.test_class_discrimination and not args.quick_test:
            feature_cols = [
                col for col in enhanced_df.columns if col.startswith("interpretable_")
            ]
            class_analysis = analyze_class_discrimination(
                enhanced_df, feature_cols, "outcome_bin"
            )
            test_results["detailed_class_analysis"] = class_analysis.to_dict("records")

        # Save results
        output_dir = Path(args.output_dir)
        save_iteration_results(test_results, enhanced_df, args.iteration, output_dir)

        # Generate governance update if requested
        if args.auto_governance_update and "discriminative_power" in test_results:
            disc_df = pd.DataFrame(test_results["discriminative_power"])
            failed_features = disc_df[
                (disc_df["mutual_info"] <= 0.005)
                | (disc_df["kw_pvalue"] >= 0.1)
                | (disc_df["zero_pct"] >= 95)
                | (disc_df["missing_pct"] >= 20)
            ]["feature"].tolist()

            if failed_features:
                reasons = {f: "failed_quality_checks" for f in failed_features}
                generate_governance_update(
                    failed_features, reasons, args.iteration, output_dir
                )

        # Print summary
        print("\n" + "=" * 60)
        print("ITERATION RESULTS")
        print("=" * 60)

        if "summary" in test_results:
            summary = test_results["summary"]
            print(f"✅ Total features tested: {summary.get('total_features', 0)}")
            print(f"✅ Features passing tests: {summary.get('passing_all_tests', 0)}")
            print(f"🎯 Class 0 discriminators: {summary.get('good_for_class_0', 0)}")

            if summary.get("class_0_discriminators"):
                print("\n🏆 FEATURES THAT HELP WITH CLASS 0:")
                for feature in summary["class_0_discriminators"]:
                    print(f"   - {feature}")

            if summary.get("top_mi_features"):
                print(f"\n📊 TOP MUTUAL INFO FEATURES:")
                for feature in summary["top_mi_features"][:5]:
                    print(f"   - {feature}")

        print(f"\n📁 Results saved to: {output_dir}/iteration_{args.iteration}/")

        if args.auto_governance_update:
            print("📝 Check governance update file for patterns to add")

        return 0

    except Exception as e:
        print(f"❌ Iteration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
