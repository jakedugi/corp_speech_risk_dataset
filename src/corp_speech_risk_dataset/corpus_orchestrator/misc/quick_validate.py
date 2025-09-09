#!/usr/bin/env python3
"""Quick POLR Data Validation Script

Fast standalone script to validate POLR pipeline data integrity and weight inheritance.
This script performs essential checks without running the full pipeline.

Usage:
    python scripts/quick_validate.py
    python scripts/quick_validate.py --data-dir custom/path
    python scripts/quick_validate.py --detailed
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class QuickValidator:
    """Quick validation of POLR pipeline data."""

    def __init__(
        self, data_dir: str = "data/final_stratified_kfold_splits_authoritative"
    ):
        self.data_dir = Path(data_dir)
        self.issues = []
        self.warnings = []

    def log_issue(self, message: str):
        """Log a validation issue."""
        self.issues.append(message)
        print(f"❌ ISSUE: {message}")

    def log_warning(self, message: str):
        """Log a validation warning."""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")

    def log_success(self, message: str):
        """Log a validation success."""
        print(f"✅ {message}")

    def check_directory_structure(self) -> bool:
        """Check basic directory structure."""
        print("🔍 Checking directory structure...")

        if not self.data_dir.exists():
            self.log_issue(f"Data directory does not exist: {self.data_dir}")
            return False

        # Check required files
        required_files = ["per_fold_metadata.json", "fold_statistics.json"]

        for file_name in required_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                self.log_issue(f"Missing required file: {file_path}")
            else:
                self.log_success(f"Found {file_name}")

        # Check fold directories
        required_folds = ["fold_0", "fold_1", "fold_2", "fold_3", "oof_test"]
        for fold_name in required_folds:
            fold_path = self.data_dir / fold_name
            if not fold_path.exists():
                self.log_issue(f"Missing fold directory: {fold_path}")
            else:
                self.log_success(f"Found {fold_name}/")

        return len(self.issues) == 0

    def check_metadata_integrity(self) -> Dict[str, Any]:
        """Check metadata file integrity."""
        print("\n🔍 Checking metadata integrity...")

        metadata_path = self.data_dir / "per_fold_metadata.json"
        if not metadata_path.exists():
            self.log_issue("per_fold_metadata.json not found")
            return {}

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            self.log_issue(f"Invalid JSON in per_fold_metadata.json: {e}")
            return {}

        # Check required keys
        required_keys = ["binning", "weights"]
        for key in required_keys:
            if key not in metadata:
                self.log_issue(f"Missing metadata key: {key}")
            else:
                self.log_success(f"Found metadata key: {key}")

        # Check binning structure
        if "binning" in metadata:
            binning = metadata["binning"]
            if "fold_edges" in binning:
                fold_edges = binning["fold_edges"]
                expected_folds = ["fold_0", "fold_1", "fold_2", "fold_3"]

                for fold in expected_folds:
                    if fold not in fold_edges:
                        self.log_issue(f"Missing fold edges for {fold}")
                    else:
                        edges = fold_edges[fold]
                        if isinstance(edges, list) and len(edges) == 2:
                            self.log_success(
                                f"Valid edges for {fold}: [{edges[0]:.0f}, {edges[1]:.0f}]"
                            )
                        else:
                            self.log_issue(f"Invalid edges for {fold}: {edges}")
            else:
                self.log_issue("Missing fold_edges in binning metadata")

        # Check weights structure
        if "weights" in metadata:
            weights = metadata["weights"]
            expected_folds = ["fold_0", "fold_1", "fold_2", "fold_3"]

            for fold in expected_folds:
                if fold not in weights:
                    self.log_issue(f"Missing weights for {fold}")
                else:
                    fold_weights = weights[fold]
                    if "class_weights" in fold_weights:
                        class_weights = fold_weights["class_weights"]
                        expected_classes = ["0", "1", "2"]  # String keys in JSON

                        missing_classes = [
                            c for c in expected_classes if c not in class_weights
                        ]
                        if missing_classes:
                            self.log_issue(
                                f"Missing class weights for {fold}: {missing_classes}"
                            )
                        else:
                            weight_values = [class_weights[c] for c in expected_classes]
                            self.log_success(
                                f"Valid class weights for {fold}: {weight_values}"
                            )
                    else:
                        self.log_issue(f"Missing class_weights for {fold}")
        else:
            self.log_issue("Missing weights in metadata")

        return metadata

    def check_sample_data_integrity(self, detailed: bool = False) -> Dict[str, Any]:
        """Check sample data for precomputed fields."""
        print("\n🔍 Checking sample data integrity...")

        results = {
            "folds_checked": 0,
            "samples_checked": 0,
            "missing_outcome_bin": [],
            "missing_sample_weight": [],
            "invalid_outcome_bin": [],
            "zero_sample_weight": [],
        }

        # Check each fold
        for fold_name in ["fold_0", "fold_1", "fold_2", "fold_3"]:
            fold_dir = self.data_dir / fold_name
            if not fold_dir.exists():
                continue

            # Check train.jsonl
            train_file = fold_dir / "train.jsonl"
            if train_file.exists():
                results["folds_checked"] += 1

                try:
                    sample_count = 0
                    with open(train_file, "r") as f:
                        for i, line in enumerate(f):
                            if (
                                not detailed and i >= 10
                            ):  # Check first 10 rows unless detailed
                                break

                            try:
                                row = json.loads(line)
                                sample_count += 1

                                # Check outcome_bin
                                if "outcome_bin" not in row:
                                    results["missing_outcome_bin"].append(
                                        f"{fold_name}/train.jsonl:line{i+1}"
                                    )
                                elif not isinstance(
                                    row["outcome_bin"], (int, float)
                                ) or row["outcome_bin"] not in [0, 1, 2]:
                                    results["invalid_outcome_bin"].append(
                                        f"{fold_name}/train.jsonl:line{i+1}:value={row['outcome_bin']}"
                                    )

                                # Check sample_weight
                                if "sample_weight" not in row:
                                    results["missing_sample_weight"].append(
                                        f"{fold_name}/train.jsonl:line{i+1}"
                                    )
                                elif (
                                    isinstance(row["sample_weight"], (int, float))
                                    and row["sample_weight"] <= 0
                                ):
                                    results["zero_sample_weight"].append(
                                        f"{fold_name}/train.jsonl:line{i+1}:value={row['sample_weight']}"
                                    )

                            except json.JSONDecodeError:
                                self.log_warning(
                                    f"Invalid JSON in {fold_name}/train.jsonl at line {i+1}"
                                )

                    results["samples_checked"] += sample_count
                    self.log_success(
                        f"Checked {sample_count} samples from {fold_name}/train.jsonl"
                    )

                except Exception as e:
                    self.log_warning(f"Error reading {train_file}: {e}")

        # Check OOF test set
        oof_test_file = self.data_dir / "oof_test" / "test.jsonl"
        if oof_test_file.exists():
            try:
                with open(oof_test_file, "r") as f:
                    first_line = f.readline()
                    if first_line:
                        oof_sample = json.loads(first_line)
                        if "outcome_bin" in oof_sample:
                            self.log_success(
                                "OOF test set has precomputed outcome_bin labels"
                            )
                        else:
                            self.log_issue("OOF test set missing outcome_bin labels")
            except Exception as e:
                self.log_warning(f"Error reading OOF test file: {e}")

        # Report issues
        for issue_type, issue_list in results.items():
            if isinstance(issue_list, list) and issue_list:
                if len(issue_list) <= 5 or detailed:
                    self.log_issue(f"{issue_type}: {issue_list}")
                else:
                    self.log_issue(
                        f"{issue_type}: {len(issue_list)} instances (first 5: {issue_list[:5]})"
                    )

        return results

    def check_class_distribution(self) -> Dict[str, Any]:
        """Check class distribution across folds."""
        print("\n🔍 Checking class distribution...")

        distributions = {}

        for fold_name in ["fold_0", "fold_1", "fold_2", "fold_3"]:
            fold_dir = self.data_dir / fold_name
            train_file = fold_dir / "train.jsonl"

            if train_file.exists():
                class_counts = {0: 0, 1: 0, 2: 0}
                total_samples = 0

                try:
                    with open(train_file, "r") as f:
                        for line in f:
                            row = json.loads(line)
                            if "outcome_bin" in row:
                                outcome = int(row["outcome_bin"])
                                if outcome in class_counts:
                                    class_counts[outcome] += 1
                                total_samples += 1

                    if total_samples > 0:
                        proportions = {
                            k: v / total_samples for k, v in class_counts.items()
                        }
                        distributions[fold_name] = {
                            "counts": class_counts,
                            "proportions": proportions,
                            "total": total_samples,
                        }

                        prop_str = ", ".join(
                            [f"Class {k}: {v:.1%}" for k, v in proportions.items()]
                        )
                        self.log_success(
                            f"{fold_name}: {total_samples} samples ({prop_str})"
                        )

                        # Check for severe imbalance
                        min_prop = min(proportions.values())
                        max_prop = max(proportions.values())
                        if min_prop < 0.15:  # Less than 15% for any class
                            self.log_warning(
                                f"{fold_name}: Severe class imbalance (min: {min_prop:.1%}, max: {max_prop:.1%})"
                            )

                except Exception as e:
                    self.log_warning(
                        f"Error reading {train_file} for class distribution: {e}"
                    )

        # Check OOF distribution
        oof_test_file = self.data_dir / "oof_test" / "test.jsonl"
        if oof_test_file.exists():
            class_counts = {0: 0, 1: 0, 2: 0}
            total_samples = 0

            try:
                with open(oof_test_file, "r") as f:
                    for line in f:
                        row = json.loads(line)
                        if "outcome_bin" in row:
                            outcome = int(row["outcome_bin"])
                            if outcome in class_counts:
                                class_counts[outcome] += 1
                            total_samples += 1

                if total_samples > 0:
                    proportions = {
                        k: v / total_samples for k, v in class_counts.items()
                    }
                    distributions["oof_test"] = {
                        "counts": class_counts,
                        "proportions": proportions,
                        "total": total_samples,
                    }

                    prop_str = ", ".join(
                        [f"Class {k}: {v:.1%}" for k, v in proportions.items()]
                    )
                    self.log_success(f"OOF test: {total_samples} samples ({prop_str})")

            except Exception as e:
                self.log_warning(
                    f"Error reading OOF test file for class distribution: {e}"
                )

        return distributions

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary validation report."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "data_directory": str(self.data_dir),
            "total_issues": len(self.issues),
            "total_warnings": len(self.warnings),
            "issues": self.issues.copy(),
            "warnings": self.warnings.copy(),
            "validation_passed": len(self.issues) == 0,
        }

        return summary

    def run_validation(self, detailed: bool = False) -> Dict[str, Any]:
        """Run complete validation."""
        print("🚀 Starting POLR Data Validation")
        print("=" * 60)
        print(f"Data directory: {self.data_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Run all validation checks
        structure_ok = self.check_directory_structure()
        metadata = self.check_metadata_integrity()
        sample_results = self.check_sample_data_integrity(detailed=detailed)
        class_distributions = self.check_class_distribution()

        # Generate summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if len(self.issues) == 0:
            print("🎉 All validation checks PASSED!")
            print("✅ Data is ready for POLR pipeline execution")
        else:
            print(f"❌ Found {len(self.issues)} ISSUES that need attention:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

        if len(self.warnings) > 0:
            print(f"\n⚠️  Found {len(self.warnings)} WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        print("=" * 60)

        # Generate full report
        summary = self.generate_summary_report()
        summary.update(
            {
                "structure_ok": structure_ok,
                "metadata": metadata,
                "sample_check_results": sample_results,
                "class_distributions": class_distributions,
            }
        )

        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Quick POLR Data Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python scripts/quick_validate.py

  # Validate custom data directory
  python scripts/quick_validate.py --data-dir path/to/custom/data

  # Detailed validation (checks more samples)
  python scripts/quick_validate.py --detailed

  # Save validation report
  python scripts/quick_validate.py --output report.json
        """,
    )

    parser.add_argument(
        "--data-dir",
        default="data/final_stratified_kfold_splits_authoritative",
        help="Data directory to validate (default: data/final_stratified_kfold_splits_authoritative)",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed validation (checks more samples, slower)",
    )

    parser.add_argument("--output", help="Save validation report to JSON file")

    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output, only show final summary"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Redirect output if quiet mode
    if args.quiet:
        import io
        import sys

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    # Run validation
    validator = QuickValidator(args.data_dir)

    try:
        summary = validator.run_validation(detailed=args.detailed)

        # Restore output if quiet mode
        if args.quiet:
            sys.stdout = original_stdout

            # Print only final summary
            if summary["validation_passed"]:
                print("✅ VALIDATION PASSED - Data ready for POLR pipeline")
            else:
                print(f"❌ VALIDATION FAILED - {summary['total_issues']} issues found")

                if summary["total_issues"] <= 5:
                    for issue in summary["issues"]:
                        print(f"  - {issue}")
                else:
                    for issue in summary["issues"][:5]:
                        print(f"  - {issue}")
                    print(f"  ... and {summary['total_issues'] - 5} more issues")

        # Save report if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"📄 Validation report saved to: {output_path}")

        # Exit with error code if validation failed
        if not summary["validation_passed"]:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
