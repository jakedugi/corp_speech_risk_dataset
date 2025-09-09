#!/usr/bin/env python3
"""
Generate Comprehensive Feature Performance Report

Creates a detailed JSON report of feature performance based on existing validation results.
Shows which features pass which tests and identifies top performers.

Usage:
    python scripts/generate_feature_performance_report.py \
        --validation-dir docs/feature_development_kfold/iteration_comprehensive_validation_with_6_new \
        --output-file docs/feature_performance_comprehensive_report.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Set


def load_validation_results(validation_dir: Path) -> Dict[str, Any]:
    """Load all validation results from CSV files."""

    results = {
        "discriminative_power": {},
        "class_0_discrimination": {},
        "size_bias_check": {},
        "leakage_check": {},
        "new_feature_focus": {},
    }

    # Load discriminative power
    discrim_file = validation_dir / "discriminative_power.csv"
    if discrim_file.exists():
        with open(discrim_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = row["feature"].replace("interpretable_", "")

                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value and value.strip() else default
                    except (ValueError, TypeError):
                        return default

                results["discriminative_power"][feature] = {
                    "mean": safe_float(row.get("mean", 0)),
                    "std": safe_float(row.get("std", 0)),
                    "zero_pct": safe_float(row.get("zero_pct", 0)),
                    "missing_pct": safe_float(row.get("missing_pct", 0)),
                    "mutual_info": safe_float(row.get("mutual_info", 0)),
                    "kw_statistic": safe_float(row.get("kw_statistic", 0)),
                    "kw_pvalue": safe_float(row.get("kw_pvalue", 1), 1.0),
                    "cliff_delta_01": safe_float(row.get("cliff_delta_01", 0)),
                    "cliff_delta_12": safe_float(row.get("cliff_delta_12", 0)),
                    "is_new_feature": row.get("is_new_feature", "False") == "True",
                }

    # Load class 0 discrimination
    class0_file = validation_dir / "class_0_discrimination.csv"
    if class0_file.exists():
        with open(class0_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = row["feature"].replace("interpretable_", "")

                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value and value.strip() else default
                    except (ValueError, TypeError):
                        return default

                results["class_0_discrimination"][feature] = {
                    "class_0_auc": safe_float(row.get("class_0_auc", 0.5), 0.5),
                    "class_0_separation": safe_float(row.get("class_0_separation", 0)),
                    "class_0_significance": safe_float(
                        row.get("class_0_significance", 1), 1.0
                    ),
                    "class_0_mean": safe_float(row.get("class_0_mean", 0)),
                    "class_1_mean": safe_float(row.get("class_1_mean", 0)),
                    "class_2_mean": safe_float(row.get("class_2_mean", 0)),
                    "is_new_feature": row.get("is_new_feature", "False") == "True",
                }

    # Load size bias check
    size_bias_file = validation_dir / "size_bias_check.csv"
    if size_bias_file.exists():
        with open(size_bias_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = row["feature"].replace("interpretable_", "")

                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value and value.strip() else default
                    except (ValueError, TypeError):
                        return default

                results["size_bias_check"][feature] = {
                    "corr_case_size": safe_float(row.get("corr_case_size", 0)),
                    "size_bias_pvalue": safe_float(row.get("size_bias_pvalue", 1), 1.0),
                    "size_bias_flag": row.get("size_bias_flag", "False") == "True",
                    "is_new_feature": row.get("is_new_feature", "False") == "True",
                }

    # Load leakage check
    leakage_file = validation_dir / "leakage_check.csv"
    if leakage_file.exists():
        with open(leakage_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = row["feature"].replace("interpretable_", "")

                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value and value.strip() else default
                    except (ValueError, TypeError):
                        return default

                results["leakage_check"][feature] = {
                    "outcome_correlation": safe_float(
                        row.get("outcome_correlation", 0)
                    ),
                    "leakage_flag": row.get("leakage_flag", "False") == "True",
                    "is_new_feature": row.get("is_new_feature", "False") == "True",
                }

    # Load new feature focus if available
    new_feat_file = validation_dir / "new_feature_focus.csv"
    if new_feat_file.exists():
        with open(new_feat_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = row["feature"].replace("interpretable_", "")

                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value and value.strip() else default
                    except (ValueError, TypeError):
                        return default

                results["new_feature_focus"][feature] = {
                    "discriminative_power": safe_float(
                        row.get("discriminative_power", 0)
                    ),
                    "class_0_auc": safe_float(row.get("class_0_auc", 0.5), 0.5),
                    "size_bias": row.get("size_bias", "False") == "True",
                    "leakage": row.get("leakage", "False") == "True",
                }

    return results


def analyze_feature_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze comprehensive feature performance."""

    # Get all features across all test results
    all_features = set()
    for test_results in results.values():
        all_features.update(test_results.keys())

    feature_analysis = {}

    for feature in all_features:
        analysis = {
            "feature_name": feature,
            "is_new_feature": False,
            "test_results": {},
            "performance_metrics": {},
            "passes": {},
            "total_tests_passed": 0,
            "total_tests_run": 0,
            "overall_score": 0.0,
        }

        # Extract results from each test category
        for test_category, test_data in results.items():
            if feature in test_data:
                analysis["test_results"][test_category] = test_data[feature]
                analysis["is_new_feature"] = test_data[feature].get(
                    "is_new_feature", False
                )

        # Performance metrics (key indicators)
        if feature in results["class_0_discrimination"]:
            c0 = results["class_0_discrimination"][feature]
            analysis["performance_metrics"]["class_0_auc"] = c0["class_0_auc"]
            analysis["performance_metrics"]["class_0_significance"] = c0[
                "class_0_significance"
            ]

        if feature in results["discriminative_power"]:
            dp = results["discriminative_power"][feature]
            analysis["performance_metrics"]["mutual_info"] = dp["mutual_info"]
            analysis["performance_metrics"]["kw_pvalue"] = dp["kw_pvalue"]
            analysis["performance_metrics"]["zero_pct"] = dp["zero_pct"]

        # Test passes (define what constitutes "passing" each test)

        # 1. Coverage test (not too sparse)
        if feature in results["discriminative_power"]:
            dp = results["discriminative_power"][feature]
            analysis["passes"]["coverage"] = (
                dp["zero_pct"] < 95.0
            )  # Less than 95% zeros
            analysis["total_tests_run"] += 1
            if analysis["passes"]["coverage"]:
                analysis["total_tests_passed"] += 1

        # 2. Discriminative power test
        if feature in results["discriminative_power"]:
            dp = results["discriminative_power"][feature]
            analysis["passes"]["discriminative_power"] = (
                dp["kw_pvalue"] < 0.05 and dp["mutual_info"] > 0.001
            )
            analysis["total_tests_run"] += 1
            if analysis["passes"]["discriminative_power"]:
                analysis["total_tests_passed"] += 1

        # 3. Class 0 discrimination test
        if feature in results["class_0_discrimination"]:
            c0 = results["class_0_discrimination"][feature]
            analysis["passes"]["class_0_discrimination"] = (
                c0["class_0_auc"] > 0.52 and c0["class_0_significance"] < 0.05
            )
            analysis["total_tests_run"] += 1
            if analysis["passes"]["class_0_discrimination"]:
                analysis["total_tests_passed"] += 1

        # 4. Size bias test
        if feature in results["size_bias_check"]:
            sb = results["size_bias_check"][feature]
            analysis["passes"]["size_bias"] = not sb["size_bias_flag"]
            analysis["total_tests_run"] += 1
            if analysis["passes"]["size_bias"]:
                analysis["total_tests_passed"] += 1

        # 5. Leakage test
        if feature in results["leakage_check"]:
            lc = results["leakage_check"][feature]
            analysis["passes"]["leakage"] = not lc["leakage_flag"]
            analysis["total_tests_run"] += 1
            if analysis["passes"]["leakage"]:
                analysis["total_tests_passed"] += 1

        # Calculate overall score
        if analysis["total_tests_run"] > 0:
            analysis["overall_score"] = (
                analysis["total_tests_passed"] / analysis["total_tests_run"]
            )

        feature_analysis[feature] = analysis

    return feature_analysis


def identify_top_performers(feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Identify top performing features across different categories."""

    # Convert to list for sorting
    features_list = list(feature_analysis.values())

    # Overall top performers (high overall score + good class 0 AUC)
    overall_top = sorted(
        features_list,
        key=lambda x: (
            x["overall_score"],
            x["performance_metrics"].get("class_0_auc", 0.5),
            -x["performance_metrics"].get("zero_pct", 100),
        ),
        reverse=True,
    )[:20]

    # Best Class 0 discriminators
    class0_top = sorted(
        [f for f in features_list if "class_0_auc" in f["performance_metrics"]],
        key=lambda x: x["performance_metrics"]["class_0_auc"],
        reverse=True,
    )[:20]

    # Most robust (pass most tests)
    robust_top = sorted(
        features_list,
        key=lambda x: (x["total_tests_passed"], x["overall_score"]),
        reverse=True,
    )[:20]

    # Best new features
    new_features = [f for f in features_list if f["is_new_feature"]]
    new_top = sorted(
        new_features,
        key=lambda x: (
            x["overall_score"],
            x["performance_metrics"].get("class_0_auc", 0.5),
        ),
        reverse=True,
    )[:20]

    # Features that pass all available tests
    perfect_features = [
        f
        for f in features_list
        if f["overall_score"] == 1.0 and f["total_tests_run"] >= 4
    ]

    return {
        "overall_top_performers": [
            {
                "feature_name": f["feature_name"],
                "overall_score": f["overall_score"],
                "total_tests_passed": f["total_tests_passed"],
                "total_tests_run": f["total_tests_run"],
                "class_0_auc": f["performance_metrics"].get("class_0_auc", 0.5),
                "is_new_feature": f["is_new_feature"],
            }
            for f in overall_top
        ],
        "best_class0_discriminators": [
            {
                "feature_name": f["feature_name"],
                "class_0_auc": f["performance_metrics"].get("class_0_auc", 0.5),
                "class_0_significance": f["performance_metrics"].get(
                    "class_0_significance", 1.0
                ),
                "overall_score": f["overall_score"],
                "is_new_feature": f["is_new_feature"],
            }
            for f in class0_top
        ],
        "most_robust_features": [
            {
                "feature_name": f["feature_name"],
                "total_tests_passed": f["total_tests_passed"],
                "total_tests_run": f["total_tests_run"],
                "overall_score": f["overall_score"],
                "is_new_feature": f["is_new_feature"],
            }
            for f in robust_top
        ],
        "top_new_features": [
            {
                "feature_name": f["feature_name"],
                "overall_score": f["overall_score"],
                "class_0_auc": f["performance_metrics"].get("class_0_auc", 0.5),
                "total_tests_passed": f["total_tests_passed"],
            }
            for f in new_top
        ],
        "perfect_score_features": [
            {
                "feature_name": f["feature_name"],
                "total_tests_passed": f["total_tests_passed"],
                "total_tests_run": f["total_tests_run"],
                "class_0_auc": f["performance_metrics"].get("class_0_auc", 0.5),
            }
            for f in perfect_features
        ],
    }


def generate_summary_statistics(feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics across all features."""

    features_list = list(feature_analysis.values())

    # Basic counts
    total_features = len(features_list)
    new_features = len([f for f in features_list if f["is_new_feature"]])

    # Test pass rates
    test_categories = [
        "coverage",
        "discriminative_power",
        "class_0_discrimination",
        "size_bias",
        "leakage",
    ]
    pass_rates = {}

    for test in test_categories:
        passed = len([f for f in features_list if f["passes"].get(test, False)])
        total_tested = len([f for f in features_list if test in f["passes"]])
        pass_rates[test] = {
            "passed": passed,
            "total_tested": total_tested,
            "pass_rate": passed / total_tested if total_tested > 0 else 0.0,
        }

    # Overall performance distribution
    overall_scores = [f["overall_score"] for f in features_list]
    class0_aucs = [
        f["performance_metrics"].get("class_0_auc", 0.5)
        for f in features_list
        if "class_0_auc" in f["performance_metrics"]
    ]

    summary = {
        "total_features_analyzed": total_features,
        "new_features_count": new_features,
        "test_pass_rates": pass_rates,
        "performance_distribution": {
            "mean_overall_score": (
                sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            ),
            "mean_class0_auc": (
                sum(class0_aucs) / len(class0_aucs) if class0_aucs else 0.5
            ),
            "features_with_perfect_score": len(
                [f for f in features_list if f["overall_score"] == 1.0]
            ),
            "features_with_good_class0_auc": len(
                [
                    f
                    for f in features_list
                    if f["performance_metrics"].get("class_0_auc", 0.5) > 0.55
                ]
            ),
        },
        "feature_quality_tiers": {
            "tier_1_excellent": len(
                [f for f in features_list if f["overall_score"] >= 0.8]
            ),
            "tier_2_good": len(
                [f for f in features_list if 0.6 <= f["overall_score"] < 0.8]
            ),
            "tier_3_moderate": len(
                [f for f in features_list if 0.4 <= f["overall_score"] < 0.6]
            ),
            "tier_4_poor": len([f for f in features_list if f["overall_score"] < 0.4]),
        },
    }

    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive feature performance report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--validation-dir",
        required=True,
        help="Directory containing validation results CSV files",
    )
    parser.add_argument(
        "--output-file",
        default="docs/feature_performance_comprehensive_report.json",
        help="Output JSON file for the report",
    )

    args = parser.parse_args()

    print("📊 COMPREHENSIVE FEATURE PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Validation directory: {args.validation_dir}")
    print(f"Output file: {args.output_file}")
    print()

    try:
        validation_dir = Path(args.validation_dir)

        # Load validation results
        print("📋 Loading validation results...")
        results = load_validation_results(validation_dir)

        # Analyze feature performance
        print("🔍 Analyzing feature performance...")
        feature_analysis = analyze_feature_performance(results)

        # Identify top performers
        print("🏆 Identifying top performers...")
        top_performers = identify_top_performers(feature_analysis)

        # Generate summary statistics
        print("📈 Generating summary statistics...")
        summary_stats = generate_summary_statistics(feature_analysis)

        # Create comprehensive report
        comprehensive_report = {
            "metadata": {
                "analysis_date": pd.Timestamp.now().isoformat(),
                "validation_source": str(validation_dir),
                "total_features_analyzed": len(feature_analysis),
            },
            "summary_statistics": summary_stats,
            "top_performers": top_performers,
            "detailed_feature_analysis": feature_analysis,
        }

        # Save report
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(comprehensive_report, f, indent=2, default=str)

        # Print key insights
        print("\n" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)

        print(f"📊 Total features analyzed: {summary_stats['total_features_analyzed']}")
        print(f"🆕 New features: {summary_stats['new_features_count']}")
        print(
            f"⭐ Perfect score features: {summary_stats['performance_distribution']['features_with_perfect_score']}"
        )
        print(
            f"🎯 Good Class 0 discriminators: {summary_stats['performance_distribution']['features_with_good_class0_auc']}"
        )

        print(f"\n📈 TEST PASS RATES:")
        for test, stats in summary_stats["test_pass_rates"].items():
            pass_rate = stats["pass_rate"] * 100
            print(
                f"   {test.replace('_', ' ').title()}: {stats['passed']}/{stats['total_tested']} ({pass_rate:.1f}%)"
            )

        print(f"\n🏆 TOP 10 OVERALL PERFORMERS:")
        for i, perf in enumerate(top_performers["overall_top_performers"][:10], 1):
            new_flag = "🆕" if perf["is_new_feature"] else "  "
            print(
                f"{i:2d}. {new_flag} {perf['feature_name']:<40} "
                f"(Score:{perf['overall_score']:.2f}, AUC:{perf['class_0_auc']:.3f}, "
                f"Tests:{perf['total_tests_passed']}/{perf['total_tests_run']})"
            )

        print(f"\n🆕 TOP NEW FEATURES:")
        for i, perf in enumerate(top_performers["top_new_features"][:10], 1):
            print(
                f"{i:2d}. {perf['feature_name']:<40} "
                f"(Score:{perf['overall_score']:.2f}, AUC:{perf['class_0_auc']:.3f})"
            )

        print(
            f"\n✅ PERFECT SCORE FEATURES ({len(top_performers['perfect_score_features'])}):"
        )
        for perf in top_performers["perfect_score_features"]:
            print(f"   • {perf['feature_name']} (AUC:{perf['class_0_auc']:.3f})")

        print(f"\n📁 Complete report saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import pandas as pd

    exit(main())
