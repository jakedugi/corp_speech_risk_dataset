#!/usr/bin/env python3
"""Comprehensive Label Coverage Analysis

Analyzes case-level and quote-level label distribution across all folds, splits,
and datasets to diagnose prediction collapse issues.

This script provides detailed coverage analysis to understand:
- Label distribution at case level vs quote level
- Class balance across all folds and splits
- Potential data imbalance causing model collapse
- Sample weight distribution by class
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
import argparse


class LabelCoverageAnalyzer:
    """Comprehensive label coverage analyzer."""

    def __init__(
        self, data_dir: str = "data/final_stratified_kfold_splits_authoritative"
    ):
        self.data_dir = Path(data_dir)
        self.results = {}

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single JSONL file for label coverage."""
        if not file_path.exists():
            return {"error": "File not found"}

        try:
            df = pd.read_json(file_path, lines=True)

            # Basic stats
            n_rows = len(df)
            n_cases = df["case_id"].nunique() if "case_id" in df.columns else n_rows

            # Quote-level label distribution
            quote_level = {}
            if "outcome_bin" in df.columns:
                quote_counts = df["outcome_bin"].value_counts().sort_index()
                quote_level = {
                    "counts": {int(k): int(v) for k, v in quote_counts.items()},
                    "proportions": {
                        int(k): float(v / n_rows) for k, v in quote_counts.items()
                    },
                    "total": n_rows,
                }

            # Case-level label distribution
            case_level = {}
            if "case_id" in df.columns and "outcome_bin" in df.columns:
                # Get one label per case (should be consistent within case)
                case_labels = df.groupby("case_id")["outcome_bin"].first()
                case_counts = case_labels.value_counts().sort_index()
                case_level = {
                    "counts": {int(k): int(v) for k, v in case_counts.items()},
                    "proportions": {
                        int(k): float(v / n_cases) for k, v in case_counts.items()
                    },
                    "total": n_cases,
                }

                # Check for within-case label consistency
                case_consistency = df.groupby("case_id")["outcome_bin"].nunique()
                inconsistent_cases = case_consistency[case_consistency > 1]
                case_level["inconsistent_cases"] = len(inconsistent_cases)
                if len(inconsistent_cases) > 0:
                    case_level[
                        "inconsistent_case_ids"
                    ] = inconsistent_cases.index.tolist()[
                        :10
                    ]  # First 10

            # Sample weight distribution by class
            weight_by_class = {}
            if "sample_weight" in df.columns and "outcome_bin" in df.columns:
                for class_val in sorted(df["outcome_bin"].unique()):
                    class_weights = df[df["outcome_bin"] == class_val]["sample_weight"]
                    weight_by_class[int(class_val)] = {
                        "mean": float(class_weights.mean()),
                        "std": float(class_weights.std()),
                        "min": float(class_weights.min()),
                        "max": float(class_weights.max()),
                        "median": float(class_weights.median()),
                    }

            # Quotes per case distribution
            qpc_stats = {}
            if "case_id" in df.columns:
                qpc = df.groupby("case_id").size()
                qpc_stats = {
                    "mean": float(qpc.mean()),
                    "std": float(qpc.std()),
                    "min": int(qpc.min()),
                    "max": int(qpc.max()),
                    "median": float(qpc.median()),
                    "histogram": {
                        int(k): int(v) for k, v in qpc.value_counts().head(20).items()
                    },
                }

            # Continuous outcome distribution (for reference)
            continuous_stats = {}
            if "final_judgement_real" in df.columns:
                outcomes = df["final_judgement_real"].dropna()
                if len(outcomes) > 0:
                    continuous_stats = {
                        "min": float(outcomes.min()),
                        "max": float(outcomes.max()),
                        "mean": float(outcomes.mean()),
                        "median": float(outcomes.median()),
                        "std": float(outcomes.std()),
                        "q25": float(outcomes.quantile(0.25)),
                        "q75": float(outcomes.quantile(0.75)),
                    }

            return {
                "file_path": str(file_path),
                "n_rows": n_rows,
                "n_cases": n_cases,
                "quote_level": quote_level,
                "case_level": case_level,
                "weight_by_class": weight_by_class,
                "quotes_per_case": qpc_stats,
                "continuous_outcome": continuous_stats,
                "has_outcome_bin": "outcome_bin" in df.columns,
                "has_sample_weight": "sample_weight" in df.columns,
                "has_case_id": "case_id" in df.columns,
            }

        except Exception as e:
            return {"error": str(e), "file_path": str(file_path)}

    def analyze_all_data(self) -> Dict[str, Any]:
        """Analyze all folds and splits."""
        print("🔍 Analyzing label coverage across all folds and splits...")

        results = {"folds": {}, "oof_test": {}, "summary": {}}

        # Analyze each fold
        for fold_name in ["fold_0", "fold_1", "fold_2", "fold_3"]:
            fold_dir = self.data_dir / fold_name
            if not fold_dir.exists():
                continue

            print(f"\n📊 Analyzing {fold_name}...")
            fold_results = {}

            # Analyze each split file in the fold
            split_files = {
                "train": "train.jsonl",
                "val": "val.jsonl",
                "test": "test.jsonl",
                "dev": "dev.jsonl",  # For fold_3
            }

            for split_name, file_name in split_files.items():
                file_path = fold_dir / file_name
                if file_path.exists():
                    print(f"  📋 Analyzing {split_name}...")
                    split_result = self.analyze_file(file_path)
                    fold_results[split_name] = split_result

                    # Print summary
                    if "error" not in split_result:
                        quote_counts = split_result.get("quote_level", {}).get(
                            "counts", {}
                        )
                        case_counts = split_result.get("case_level", {}).get(
                            "counts", {}
                        )

                        print(
                            f"    Quotes: {split_result['n_rows']:,} | Cases: {split_result['n_cases']:,}"
                        )
                        if quote_counts:
                            quote_dist = ", ".join(
                                [
                                    f"Class {k}: {v:,}"
                                    for k, v in sorted(quote_counts.items())
                                ]
                            )
                            print(f"    Quote-level: {quote_dist}")
                        if case_counts:
                            case_dist = ", ".join(
                                [
                                    f"Class {k}: {v:,}"
                                    for k, v in sorted(case_counts.items())
                                ]
                            )
                            print(f"    Case-level:  {case_dist}")

                        # Check for severe imbalance
                        if case_counts:
                            total_cases = sum(case_counts.values())
                            proportions = {
                                k: v / total_cases for k, v in case_counts.items()
                            }
                            min_prop = min(proportions.values()) if proportions else 0
                            if min_prop < 0.05:  # Less than 5%
                                print(
                                    f"    ⚠️  SEVERE IMBALANCE: Min class has {min_prop:.1%} of cases"
                                )
                            elif min_prop < 0.15:  # Less than 15%
                                print(
                                    f"    ⚠️  MODERATE IMBALANCE: Min class has {min_prop:.1%} of cases"
                                )
                    else:
                        print(f"    ❌ Error: {split_result['error']}")

            results["folds"][fold_name] = fold_results

        # Analyze OOF test set
        oof_dir = self.data_dir / "oof_test"
        if oof_dir.exists():
            print(f"\n📊 Analyzing OOF test set...")
            oof_test_file = oof_dir / "test.jsonl"
            if oof_test_file.exists():
                oof_result = self.analyze_file(oof_test_file)
                results["oof_test"] = oof_result

                if "error" not in oof_result:
                    quote_counts = oof_result.get("quote_level", {}).get("counts", {})
                    case_counts = oof_result.get("case_level", {}).get("counts", {})

                    print(
                        f"  Quotes: {oof_result['n_rows']:,} | Cases: {oof_result['n_cases']:,}"
                    )
                    if quote_counts:
                        quote_dist = ", ".join(
                            [
                                f"Class {k}: {v:,}"
                                for k, v in sorted(quote_counts.items())
                            ]
                        )
                        print(f"  Quote-level: {quote_dist}")
                    if case_counts:
                        case_dist = ", ".join(
                            [
                                f"Class {k}: {v:,}"
                                for k, v in sorted(case_counts.items())
                            ]
                        )
                        print(f"  Case-level:  {case_dist}")

        # Generate summary statistics
        print(f"\n📈 Generating summary statistics...")
        summary = self.generate_summary(results)
        results["summary"] = summary

        return results

    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            "total_cases_by_class": defaultdict(int),
            "total_quotes_by_class": defaultdict(int),
            "class_imbalance_issues": [],
            "weight_distribution_issues": [],
            "fold_consistency_issues": [],
            "recommendations": [],
        }

        # Aggregate across all folds and splits
        all_case_distributions = {}
        all_quote_distributions = {}

        # Process folds
        for fold_name, fold_data in results.get("folds", {}).items():
            for split_name, split_data in fold_data.items():
                if "error" in split_data:
                    continue

                key = f"{fold_name}_{split_name}"

                # Case-level distributions
                case_counts = split_data.get("case_level", {}).get("counts", {})
                if case_counts:
                    all_case_distributions[key] = case_counts
                    for class_val, count in case_counts.items():
                        summary["total_cases_by_class"][class_val] += count

                # Quote-level distributions
                quote_counts = split_data.get("quote_level", {}).get("counts", {})
                if quote_counts:
                    all_quote_distributions[key] = quote_counts
                    for class_val, count in quote_counts.items():
                        summary["total_quotes_by_class"][class_val] += count

                # Check for severe imbalance
                if case_counts:
                    total_cases = sum(case_counts.values())
                    proportions = {k: v / total_cases for k, v in case_counts.items()}
                    min_prop = min(proportions.values()) if proportions else 0
                    min_class = (
                        min(proportions.keys(), key=proportions.get)
                        if proportions
                        else None
                    )

                    if min_prop < 0.05:  # Less than 5%
                        summary["class_imbalance_issues"].append(
                            {
                                "dataset": key,
                                "min_class": min_class,
                                "min_proportion": min_prop,
                                "severity": "severe",
                                "counts": case_counts,
                            }
                        )
                    elif min_prop < 0.15:  # Less than 15%
                        summary["class_imbalance_issues"].append(
                            {
                                "dataset": key,
                                "min_class": min_class,
                                "min_proportion": min_prop,
                                "severity": "moderate",
                                "counts": case_counts,
                            }
                        )

        # Process OOF test
        oof_data = results.get("oof_test", {})
        if "error" not in oof_data:
            case_counts = oof_data.get("case_level", {}).get("counts", {})
            quote_counts = oof_data.get("quote_level", {}).get("counts", {})

            if case_counts:
                all_case_distributions["oof_test"] = case_counts
                for class_val, count in case_counts.items():
                    summary["total_cases_by_class"][class_val] += count

            if quote_counts:
                all_quote_distributions["oof_test"] = quote_counts
                for class_val, count in quote_counts.items():
                    summary["total_quotes_by_class"][class_val] += count

        # Convert defaultdicts to regular dicts
        summary["total_cases_by_class"] = dict(summary["total_cases_by_class"])
        summary["total_quotes_by_class"] = dict(summary["total_quotes_by_class"])

        # Check overall imbalance
        total_cases = sum(summary["total_cases_by_class"].values())
        total_quotes = sum(summary["total_quotes_by_class"].values())

        if total_cases > 0:
            case_proportions = {
                k: v / total_cases for k, v in summary["total_cases_by_class"].items()
            }
            min_case_prop = min(case_proportions.values()) if case_proportions else 0
            min_case_class = (
                min(case_proportions.keys(), key=case_proportions.get)
                if case_proportions
                else None
            )

            summary["overall_case_balance"] = {
                "proportions": case_proportions,
                "min_class": min_case_class,
                "min_proportion": min_case_prop,
                "imbalanced": min_case_prop < 0.15,
            }

        if total_quotes > 0:
            quote_proportions = {
                k: v / total_quotes for k, v in summary["total_quotes_by_class"].items()
            }
            min_quote_prop = min(quote_proportions.values()) if quote_proportions else 0
            min_quote_class = (
                min(quote_proportions.keys(), key=quote_proportions.get)
                if quote_proportions
                else None
            )

            summary["overall_quote_balance"] = {
                "proportions": quote_proportions,
                "min_class": min_quote_class,
                "min_proportion": min_quote_prop,
                "imbalanced": min_quote_prop < 0.15,
            }

        # Generate recommendations
        if summary["class_imbalance_issues"]:
            summary["recommendations"].append(
                "CRITICAL: Severe class imbalance detected in multiple datasets"
            )

        if summary["overall_case_balance"].get("imbalanced", False):
            min_class = summary["overall_case_balance"]["min_class"]
            min_prop = summary["overall_case_balance"]["min_proportion"]
            summary["recommendations"].append(
                f"Class {min_class} is underrepresented ({min_prop:.1%} of cases)"
            )

        if len(summary["total_cases_by_class"]) < 3:
            missing_classes = set([0, 1, 2]) - set(
                summary["total_cases_by_class"].keys()
            )
            summary["recommendations"].append(
                f"Missing classes entirely: {missing_classes}"
            )

        return summary

    def create_detailed_report(self, results: Dict[str, Any]) -> str:
        """Create detailed text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE LABEL COVERAGE ANALYSIS")
        lines.append("=" * 80)
        lines.append("")

        # Overall summary
        summary = results.get("summary", {})

        lines.append("📊 OVERALL SUMMARY")
        lines.append("-" * 40)

        total_cases = summary.get("total_cases_by_class", {})
        total_quotes = summary.get("total_quotes_by_class", {})

        if total_cases:
            lines.append("Case-level distribution (all data):")
            total_case_count = sum(total_cases.values())
            for class_val in sorted(total_cases.keys()):
                count = total_cases[class_val]
                prop = count / total_case_count if total_case_count > 0 else 0
                lines.append(f"  Class {class_val}: {count:,} cases ({prop:.1%})")
            lines.append(f"  Total: {total_case_count:,} cases")

        if total_quotes:
            lines.append("\nQuote-level distribution (all data):")
            total_quote_count = sum(total_quotes.values())
            for class_val in sorted(total_quotes.keys()):
                count = total_quotes[class_val]
                prop = count / total_quote_count if total_quote_count > 0 else 0
                lines.append(f"  Class {class_val}: {count:,} quotes ({prop:.1%})")
            lines.append(f"  Total: {total_quote_count:,} quotes")

        # Class imbalance issues
        imbalance_issues = summary.get("class_imbalance_issues", [])
        if imbalance_issues:
            lines.append(
                f"\n🚨 CLASS IMBALANCE ISSUES ({len(imbalance_issues)} detected)"
            )
            lines.append("-" * 40)

            for issue in imbalance_issues:
                severity_icon = "🔴" if issue["severity"] == "severe" else "🟡"
                lines.append(
                    f"{severity_icon} {issue['dataset']}: Class {issue['min_class']} has {issue['min_proportion']:.1%}"
                )
                counts_str = ", ".join(
                    [f"Class {k}: {v}" for k, v in sorted(issue["counts"].items())]
                )
                lines.append(f"    Counts: {counts_str}")

        # Fold-by-fold breakdown
        lines.append(f"\n📋 FOLD-BY-FOLD BREAKDOWN")
        lines.append("-" * 40)

        for fold_name, fold_data in results.get("folds", {}).items():
            lines.append(f"\n{fold_name.upper()}:")

            for split_name, split_data in fold_data.items():
                if "error" in split_data:
                    lines.append(f"  {split_name}: ERROR - {split_data['error']}")
                    continue

                case_counts = split_data.get("case_level", {}).get("counts", {})
                quote_counts = split_data.get("quote_level", {}).get("counts", {})

                lines.append(f"  {split_name}:")
                lines.append(
                    f"    Rows: {split_data['n_rows']:,} | Cases: {split_data['n_cases']:,}"
                )

                if case_counts:
                    case_dist = ", ".join(
                        [f"C{k}:{v}" for k, v in sorted(case_counts.items())]
                    )
                    lines.append(f"    Case-level: {case_dist}")

                if quote_counts:
                    quote_dist = ", ".join(
                        [f"C{k}:{v}" for k, v in sorted(quote_counts.items())]
                    )
                    lines.append(f"    Quote-level: {quote_dist}")

                # Weight distribution by class
                weight_by_class = split_data.get("weight_by_class", {})
                if weight_by_class:
                    lines.append("    Sample weights by class:")
                    for class_val in sorted(weight_by_class.keys()):
                        w_stats = weight_by_class[class_val]
                        lines.append(
                            f"      Class {class_val}: mean={w_stats['mean']:.4f}, std={w_stats['std']:.4f}"
                        )

        # OOF test analysis
        oof_data = results.get("oof_test", {})
        if oof_data and "error" not in oof_data:
            lines.append(f"\nOOF TEST SET:")
            lines.append(
                f"  Rows: {oof_data['n_rows']:,} | Cases: {oof_data['n_cases']:,}"
            )

            case_counts = oof_data.get("case_level", {}).get("counts", {})
            quote_counts = oof_data.get("quote_level", {}).get("counts", {})

            if case_counts:
                case_dist = ", ".join(
                    [f"C{k}:{v}" for k, v in sorted(case_counts.items())]
                )
                lines.append(f"  Case-level: {case_dist}")

            if quote_counts:
                quote_dist = ", ".join(
                    [f"C{k}:{v}" for k, v in sorted(quote_counts.items())]
                )
                lines.append(f"  Quote-level: {quote_dist}")

        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            lines.append(f"\n💡 RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")

        # Potential causes for prediction collapse
        lines.append(f"\n🔍 POTENTIAL CAUSES FOR PREDICTION COLLAPSE")
        lines.append("-" * 40)

        if summary.get("overall_case_balance", {}).get("imbalanced", False):
            min_class = summary["overall_case_balance"]["min_class"]
            min_prop = summary["overall_case_balance"]["min_proportion"]
            lines.append(
                f"1. SEVERE CLASS IMBALANCE: Class {min_class} only {min_prop:.1%} of data"
            )
            lines.append(f"   → Model learns to ignore rare class {min_class}")

        if len(imbalance_issues) > 0:
            lines.append(
                f"2. INCONSISTENT SPLITS: {len(imbalance_issues)} splits with severe imbalance"
            )
            lines.append(
                f"   → Model sees different class distributions during training vs validation"
            )

        lines.append(f"3. POSSIBLE SOLUTIONS:")
        lines.append(f"   → Use stronger class weights for rare classes")
        lines.append(
            f"   → Increase regularization (lower C) to prevent overfitting to majority class"
        )
        lines.append(f"   → Use different hyperparameter search strategy")
        lines.append(
            f"   → Consider resampling techniques (if appropriate for temporal data)"
        )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 Results saved to: {output_path}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive label coverage analysis for POLR pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        default="data/final_stratified_kfold_splits_authoritative",
        help="Data directory to analyze",
    )

    parser.add_argument("--output", help="Save results to JSON file")

    parser.add_argument("--report", help="Save detailed text report")

    parser.add_argument(
        "--focus-fold", help="Focus analysis on specific fold (e.g., fold_0)"
    )

    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="Check for within-case label consistency",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    analyzer = LabelCoverageAnalyzer(args.data_dir)

    if args.focus_fold:
        # Analyze specific fold only
        fold_dir = analyzer.data_dir / args.focus_fold
        if not fold_dir.exists():
            print(f"❌ Fold directory not found: {fold_dir}")
            return

        print(f"🔍 Analyzing {args.focus_fold} only...")
        # Implement focused analysis if needed

    else:
        # Full analysis
        results = analyzer.analyze_all_data()

        # Generate and print report
        report = analyzer.create_detailed_report(results)
        print("\n" + report)

        # Save results if requested
        if args.output:
            analyzer.save_results(results, Path(args.output))

        if args.report:
            with open(args.report, "w") as f:
                f.write(report)
            print(f"📄 Detailed report saved to: {args.report}")

    print("\n🎯 Analysis complete!")


if __name__ == "__main__":
    main()
