#!/usr/bin/env python3
"""
final_evaluate.py

Final evaluation script for case outcome imputation using tuned hyperparameters.
This script runs a single evaluation of the imputation pipeline using the optimal
parameters found through Bayesian optimization and reports per-case and overall
metrics (MSE, precision, recall, etc.).

Usage:
    python final_evaluate.py \
        --annotations data/gold_standard/case_outcome_amounts_hand_annotated.csv \
        --extracted-root data/extracted

The script uses these exact tuned parameters from fine-tuning:
    - min_amount: 29309.97970771781
    - context_chars: 561
    - min_features: 15
    - case_pos: 0.5423630428751168
    - docket_pos: 0.7947200838693315

And disables dismissal/patent logic by setting extreme thresholds:
    - dismissal_ratio_threshold: 200
    - strict_dismissal_threshold: 200
    - dismissal_document_type_weight: 0
    - strict_dismissal_document_type_weight: 0
    - bankruptcy_ratio_threshold: 6e22
    - patent_ratio_threshold: 6e22

Author: Jake Dugan <jake.dugan@ed.ac.uk>
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Import the harness functions
from test_case_outcome_imputer import (
    load_annotations,
    evaluate_case,
    AmountSelector,
    DEFAULT_VOTING_WEIGHTS,
)

# Import for metrics
from sklearn.metrics import precision_score, recall_score, mean_squared_error


def create_disabled_dismissal_params() -> Dict[str, float]:
    """
    Create dismissal parameters that effectively disable all dismissal,
    bankruptcy, and patent logic by setting extreme thresholds.

    Returns:
        Dict containing disabled dismissal parameters
    """
    return {
        "dismissal_ratio_threshold": 200.0,  # Exceeds any real ratio
        "strict_dismissal_threshold": 200.0,  # Exceeds any real ratio
        "dismissal_document_type_weight": 0.0,  # Zero weight = no impact
        "strict_dismissal_document_type_weight": 0.0,  # Zero weight = no impact
        "bankruptcy_ratio_threshold": 6e22,  # Enormous threshold
        "patent_ratio_threshold": 6e22,  # Enormous threshold
    }


def run_final_evaluation(
    annotations_path: Path, extracted_root: Path
) -> Dict[str, Any]:
    """
    Run final evaluation using tuned hyperparameters.

    Args:
        annotations_path: Path to hand-annotated CSV file
        extracted_root: Root directory of extracted case data

    Returns:
        Dictionary containing evaluation results and metrics
    """
    # Load the hand-annotated ground truth
    print(f"Loading annotations from: {annotations_path}")
    df = load_annotations(annotations_path)
    print(f"Loaded {len(df)} cases for evaluation")

    # Fixed hyperparameters from Bayesian optimization fine-tuning
    fixed_params = {
        "min_amount": 29309.97970771781,
        "context_chars": 561,
        "min_features": 15,
        "case_pos": 0.5423630428751168,
        "docket_pos": 0.7947200838693315,
    }

    # Disable dismissal, bankruptcy, and patent logic
    dismissal_params = create_disabled_dismissal_params()

    print("Using tuned hyperparameters:")
    for param, value in fixed_params.items():
        print(f"  {param}: {value}")

    print("\nDisabling dismissal/patent logic with extreme thresholds:")
    for param, value in dismissal_params.items():
        print(f"  {param}: {value}")

    # Loop over cases and collect per-case statistics
    per_case = []
    print(f"\nEvaluating {len(df)} cases...")

    for idx, (case_id, true_amt) in enumerate(
        zip(df["case_id"].astype(str), df["final_amount"])
    ):
        print(f"  [{idx+1}/{len(df)}] Processing case: {case_id}")

        args = (
            case_id,
            float(true_amt),
            extracted_root,
            fixed_params["min_amount"],
            fixed_params["context_chars"],
            fixed_params["min_features"],
            fixed_params["case_pos"],
            fixed_params["docket_pos"],
            DEFAULT_VOTING_WEIGHTS.to_dict(),
            dismissal_params,
        )

        result = evaluate_case(args)

        # Unpack result: (true_amt, pred_amt, case_name, raw_count, filt_count, in_raw, in_filt, status)
        (
            true_val,
            pred_val,
            case_name,
            raw_count,
            filt_count,
            in_raw,
            in_filt,
            status,
        ) = result
        error = pred_val - true_val

        per_case.append(
            {
                "case": case_name,
                "true": true_val,
                "pred": pred_val,
                "error": error,
                "sq_error": error**2,
                "abs_error": abs(error),
                "raw_candidates": raw_count,
                "filtered_candidates": filt_count,
                "true_in_raw": in_raw,
                "true_in_filtered": in_filt,
                "status": status,
            }
        )

        print(
            f"    True: ${true_val:,.2f}, Pred: ${pred_val:,.2f}, Error: ${error:,.2f}, Status: {status}"
        )

    return calculate_metrics(per_case, fixed_params, dismissal_params)


def calculate_metrics(
    per_case: List[Dict[str, Any]],
    fixed_params: Dict[str, float],
    dismissal_params: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from per-case results.

    Args:
        per_case: List of per-case evaluation results
        fixed_params: Hyperparameters used
        dismissal_params: Dismissal parameters used

    Returns:
        Dictionary containing all metrics and results
    """
    # Extract arrays for metric calculation
    errors = np.array([c["error"] for c in per_case])
    sq_errors = np.array([c["sq_error"] for c in per_case])
    abs_errors = np.array([c["abs_error"] for c in per_case])

    # Binary classification: award (>0) vs zero
    y_true = np.array([1 if c["true"] > 0 else 0 for c in per_case])
    y_pred = np.array([1 if c["pred"] > 0 else 0 for c in per_case])

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    mse = mean_squared_error(
        [c["true"] for c in per_case], [c["pred"] for c in per_case]
    )
    mae = np.mean(abs_errors)
    rmse = np.sqrt(mse)

    # Count status types
    status_counts = {}
    for case in per_case:
        status = case["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    # Calculate accuracy for amount recovery (how often we get the exact amount)
    exact_matches = sum(1 for c in per_case if abs(c["error"]) < 0.01)
    exact_accuracy = exact_matches / len(per_case) if per_case else 0

    # Calculate coverage metrics
    true_in_raw_count = sum(1 for c in per_case if c["true_in_raw"])
    true_in_filtered_count = sum(1 for c in per_case if c["true_in_filtered"])
    raw_coverage = true_in_raw_count / len(per_case) if per_case else 0
    filtered_coverage = true_in_filtered_count / len(per_case) if per_case else 0

    return {
        "per_case_results": per_case,
        "hyperparameters": fixed_params,
        "dismissal_params": dismissal_params,
        "metrics": {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "exact_accuracy": exact_accuracy,
            "raw_coverage": raw_coverage,
            "filtered_coverage": filtered_coverage,
        },
        "status_counts": status_counts,
        "summary_stats": {
            "total_cases": len(per_case),
            "exact_matches": exact_matches,
            "true_in_raw": true_in_raw_count,
            "true_in_filtered": true_in_filtered_count,
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "min_error": np.min(errors),
            "max_error": np.max(errors),
        },
    }


def print_results(results: Dict[str, Any]) -> None:
    """
    Print formatted evaluation results.

    Args:
        results: Dictionary containing evaluation results
    """
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    print("\nHYPERPARAMETERS USED:")
    for param, value in results["hyperparameters"].items():
        print(f"  {param}: {value}")

    print("\nDISMISSAL PARAMETERS (DISABLED):")
    for param, value in results["dismissal_params"].items():
        print(f"  {param}: {value}")

    print("\nPER-CASE RESULTS:")
    print(f"{'Case':<40} {'True':<12} {'Pred':<12} {'Error':<12} {'Status':<15}")
    print("-" * 95)

    for case in results["per_case_results"]:
        print(
            f"{case['case']:<40} ${case['true']:>10,.2f} ${case['pred']:>10,.2f} "
            f"${case['error']:>10,.2f} {case['status']:<15}"
        )

    print("\nOVERALL METRICS:")
    metrics = results["metrics"]
    print(f"  Mean Absolute Error (MAE):     ${metrics['mean_absolute_error']:>10,.2f}")
    print(
        f"  Root Mean Squared Error (RMSE): ${metrics['root_mean_squared_error']:>10,.2f}"
    )
    print(f"  Mean Squared Error (MSE):      ${metrics['mean_squared_error']:>10,.2f}")
    print(f"  Precision (award detection):    {metrics['precision']:>10.3f}")
    print(f"  Recall (award detection):       {metrics['recall']:>10.3f}")
    print(f"  F1 Score (award detection):     {metrics['f1_score']:>10.3f}")
    print(f"  Exact Amount Accuracy:          {metrics['exact_accuracy']:>10.3f}")
    print(f"  Raw Candidate Coverage:         {metrics['raw_coverage']:>10.3f}")
    print(f"  Filtered Candidate Coverage:    {metrics['filtered_coverage']:>10.3f}")

    print("\nSTATUS BREAKDOWN:")
    for status, count in results["status_counts"].items():
        percentage = (count / results["summary_stats"]["total_cases"]) * 100
        print(f"  {status:<20}: {count:>3} cases ({percentage:>5.1f}%)")

    print("\nSUMMARY STATISTICS:")
    stats = results["summary_stats"]
    print(f"  Total Cases Evaluated:  {stats['total_cases']}")
    print(f"  Exact Matches:          {stats['exact_matches']}")
    print(f"  True Amount in Raw:     {stats['true_in_raw']}")
    print(f"  True Amount in Filtered: {stats['true_in_filtered']}")
    print(f"  Mean Error:             ${stats['mean_error']:,.2f}")
    print(f"  Error Std Dev:          ${stats['std_error']:,.2f}")
    print(f"  Min Error:              ${stats['min_error']:,.2f}")
    print(f"  Max Error:              ${stats['max_error']:,.2f}")


def main():
    """Main entry point for final evaluation script."""
    parser = argparse.ArgumentParser(
        description="Final evaluation of case outcome imputation with tuned parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/gold_standard/case_outcome_amounts_hand_annotated.csv"),
        help="Path to hand-annotated CSV file (default: data/gold_standard/case_outcome_amounts_hand_annotated.csv)",
    )

    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("data/extracted"),
        help="Root directory of extracted case data (default: data/extracted)",
    )

    args = parser.parse_args()

    # Validate input paths
    if not args.annotations.exists():
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)

    if not args.extracted_root.exists():
        print(f"Error: Extracted root directory not found: {args.extracted_root}")
        sys.exit(1)

    print("Corporate Speech Risk Dataset - Final Evaluation")
    print("=" * 50)
    print(f"Annotations: {args.annotations}")
    print(f"Extracted Root: {args.extracted_root}")

    # Run evaluation
    try:
        results = run_final_evaluation(args.annotations, args.extracted_root)
        print_results(results)

        print(f"\nEvaluation completed successfully!")
        print(f"Processed {results['summary_stats']['total_cases']} cases")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
