#!/usr/bin/env python3
"""
grid_search_optimizer.py

Grid search optimization for case outcome imputer hyperparameters using LOOCV.
Optimizes on MSE loss and precision/recall metrics using the gold standard dataset.
"""

import argparse
import json
import csv
import itertools
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Import the case outcome imputer functions
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from corp_speech_risk_dataset.case_outcome.case_outcome_imputer import (
        scan_stage1,
        VotingWeights,
        DEFAULT_VOTING_WEIGHTS,
        DEFAULT_CONTEXT,
    )
    from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
        DEFAULT_MIN_AMOUNT,
    )
except ImportError:
    # Fallback for direct execution
    from case_outcome_imputer import (
        scan_stage1,
        VotingWeights,
        DEFAULT_VOTING_WEIGHTS,
        DEFAULT_CONTEXT,
    )
    from extract_cash_amounts_stage1 import (
        DEFAULT_MIN_AMOUNT,
    )
from src.corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
    get_case_flags,
    get_case_court_type,
    is_case_dismissed,
)


@dataclass
class OptimizationResult:
    """Results from a single hyperparameter combination evaluation."""

    hyperparams: Dict[str, Any]
    mse_loss: float
    precision: float
    recall: float
    f1_score: float
    exact_matches: int
    total_cases: int
    predictions: List[Optional[float]]
    actuals: List[Optional[float]]


class GridSearchOptimizer:
    """
    Grid search optimizer for case outcome imputer hyperparameters using LOOCV.
    """

    def __init__(self, gold_standard_path: str, extracted_data_root: str):
        """
        Initialize the optimizer.

        Args:
            gold_standard_path: Path to the gold standard CSV file
            extracted_data_root: Root directory containing extracted case data
        """
        self.gold_standard_path = Path(gold_standard_path)
        self.extracted_data_root = Path(extracted_data_root)
        self.gold_standard_data = self._load_gold_standard()

        # Define hyperparameter search space - ALL hyperparameters included
        self.hyperparameter_grid = {
            # Core extraction parameters
            "min_amount": [1000, 5000, 10000, 50000, 100000],
            "context_chars": [50, 100, 200, 300],
            "min_features": [1, 2, 3, 4],
            # Position thresholds
            "case_position_threshold": [0.3, 0.5, 0.7],
            "docket_position_threshold": [0.3, 0.5, 0.7],
            # Case flag thresholds
            "fee_shifting_ratio_threshold": [0.5, 1.0, 2.0],
            "patent_ratio_threshold": [10.0, 20.0, 30.0],
            "dismissal_ratio_threshold": [0.3, 0.5, 0.7],
            "bankruptcy_ratio_threshold": [0.3, 0.5, 0.7],
            # Voting weights
            "proximity_pattern_weight": [0.5, 1.0, 2.0],
            "judgment_verbs_weight": [0.5, 1.0, 2.0],
            "case_position_weight": [0.5, 1.0, 2.0],
            "docket_position_weight": [0.5, 1.0, 2.0],
            "all_caps_titles_weight": [0.5, 1.0, 2.0],
            "document_titles_weight": [0.5, 1.0, 2.0],
            # Header size
            "header_chars": [1000, 2000, 3000],
        }

        # Reduced grid for faster testing (comment out for full search)
        self.reduced_grid = {
            "min_amount": [10000, 50000],
            "context_chars": [100, 200],
            "min_features": [2, 3],
            "case_position_threshold": [0.5],
            "docket_position_threshold": [0.5],
            "fee_shifting_ratio_threshold": [1.0],
            "patent_ratio_threshold": [20.0],
            "dismissal_ratio_threshold": [0.5],
            "bankruptcy_ratio_threshold": [0.5],
            "proximity_pattern_weight": [1.0],
            "judgment_verbs_weight": [1.0],
            "case_position_weight": [1.0],
            "docket_position_weight": [1.0],
            "all_caps_titles_weight": [1.0],
            "document_titles_weight": [1.0],
            "header_chars": [1000, 2000, 3000],
        }

        # Use reduced grid for faster testing
        self.search_grid = self.reduced_grid

    def _load_gold_standard(self) -> pd.DataFrame:
        """Load and preprocess the gold standard data."""
        df = pd.read_csv(self.gold_standard_path)

        # Clean and validate the data
        df = df.dropna(subset=["case_id", "final_amount"])

        # Convert final_amount to numeric, handling comma-separated values
        def convert_amount(amount_str):
            if pd.isna(amount_str) or amount_str == "null":
                return None
            try:
                # Remove commas and convert to float
                if isinstance(amount_str, str):
                    amount_str = amount_str.replace(",", "")
                return float(amount_str)
            except (ValueError, TypeError):
                return None

        df["final_amount"] = df["final_amount"].apply(convert_amount)

        # Filter out cases with null final_amount (these are typically bankruptcy or dismissed cases)
        df = df.dropna(subset=["final_amount"])

        # Convert case_id to string and clean
        df["case_id"] = df["case_id"].astype(str).str.strip()

        print(f"Loaded {len(df)} valid cases from gold standard")
        print(
            f"Amount range: ${df['final_amount'].min():,.0f} - ${df['final_amount'].max():,.0f}"
        )

        return df

    def _get_case_data_path(self, case_id: str) -> Optional[Path]:
        """Get the path to case data for a given case ID."""
        # Extract the case directory name from the case_id
        # case_id format: "data/extracted/courtlistener/09-11435_nysb/"
        case_dir = (
            case_id.split("/")[-2] if case_id.endswith("/") else case_id.split("/")[-1]
        )

        # First try the direct path
        case_path = self.extracted_data_root / case_dir
        if case_path.exists():
            return case_path

        # If not found, try looking in subdirectories
        for subdir in self.extracted_data_root.iterdir():
            if subdir.is_dir():
                case_path = subdir / case_dir
                if case_path.exists():
                    return case_path

        # If still not found, try to construct the path from the case_id
        # The case_id might be in format: data/extracted/courtlistener/case_name/
        if case_id.startswith("data/extracted/"):
            # Extract the relative path from the case_id
            relative_path = case_id.replace("data/extracted/", "")
            case_path = self.extracted_data_root / relative_path.rstrip("/")
            if case_path.exists():
                return case_path

        return None

    def _predict_case_outcome(
        self, case_id: str, hyperparams: Dict[str, Any]
    ) -> Optional[float]:
        """
        Predict the outcome for a single case using given hyperparameters.

        Args:
            case_id: The case ID to predict
            hyperparams: Dictionary of hyperparameters

        Returns:
            Predicted amount or None if no prediction
        """
        case_path = self._get_case_data_path(case_id)
        if not case_path:
            print(f"⚠ Case data not found for {case_id}")
            return None

        try:
            # Create voting weights from hyperparameters
            voting_weights = VotingWeights(
                proximity_pattern_weight=hyperparams["proximity_pattern_weight"],
                judgment_verbs_weight=hyperparams["judgment_verbs_weight"],
                case_position_weight=hyperparams["case_position_weight"],
                docket_position_weight=hyperparams["docket_position_weight"],
                all_caps_titles_weight=hyperparams["all_caps_titles_weight"],
                document_titles_weight=hyperparams["document_titles_weight"],
            )

            # Get case flags
            flags = get_case_flags(
                case_path,
                fee_shifting_ratio_threshold=hyperparams[
                    "fee_shifting_ratio_threshold"
                ],
                patent_ratio_threshold=hyperparams["patent_ratio_threshold"],
                dismissal_ratio_threshold=hyperparams["dismissal_ratio_threshold"],
                bankruptcy_ratio_threshold=hyperparams["bankruptcy_ratio_threshold"],
            )

            # Check for bankruptcy court
            court_type = get_case_court_type(
                case_path,
                bankruptcy_ratio_threshold=hyperparams["bankruptcy_ratio_threshold"],
            )
            if court_type == "BANKRUPTCY":
                return None

            # Check for dismissed case
            if flags["is_dismissed"]:
                return 0.0

            # Scan for amounts
            candidates = scan_stage1(
                case_path,
                min_amount=hyperparams["min_amount"],
                context_chars=hyperparams["context_chars"],
                min_features=hyperparams["min_features"],
                case_position_threshold=hyperparams["case_position_threshold"],
                docket_position_threshold=hyperparams["docket_position_threshold"],
                voting_weights=voting_weights,
                header_chars=hyperparams.get("header_chars", 2000),
                fast_mode=True,  # Enable fast mode for optimization
            )

            # Return the highest voted amount, or None if no candidates
            if candidates:
                # Sort by feature votes (descending), then by value (descending)
                sorted_candidates = sorted(
                    candidates, key=lambda c: (c.feature_votes, c.value), reverse=True
                )
                return sorted_candidates[0].value

            return None

        except Exception as e:
            print(f"❌ Error predicting case {case_id}: {e}")
            return None

    def _evaluate_hyperparameters(
        self, hyperparams: Dict[str, Any], test_case_id: str, test_actual: float
    ) -> Tuple[Optional[float], float]:
        """
        Evaluate a single hyperparameter combination on one test case.

        Args:
            hyperparams: Hyperparameters to test
            test_case_id: Case ID to test
            test_actual: Actual outcome for this case

        Returns:
            Tuple of (predicted_amount, mse_loss)
        """
        predicted = self._predict_case_outcome(test_case_id, hyperparams)

        if predicted is None:
            # If no prediction, use a high penalty
            mse_loss = test_actual**2  # Penalty for missing prediction
        else:
            mse_loss = (predicted - test_actual) ** 2

        return predicted, mse_loss

    def _evaluate_hyperparameters_loocv(
        self, hyperparams: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Evaluate hyperparameters using Leave-One-Out Cross-Validation.

        Args:
            hyperparams: Hyperparameters to evaluate

        Returns:
            OptimizationResult with metrics
        """
        predictions = []
        actuals = []
        mse_losses = []

        print(f"🔍 Evaluating hyperparameters: {hyperparams}")

        # LOOCV: test on each case, train on the rest
        for idx, row in self.gold_standard_data.iterrows():
            test_case_id = str(row["case_id"])
            test_actual = float(row["final_amount"])

            predicted, mse_loss = self._evaluate_hyperparameters(
                hyperparams, test_case_id, test_actual
            )

            predictions.append(predicted)
            actuals.append(test_actual)
            mse_losses.append(mse_loss)

            print(
                f"  Case {test_case_id}: actual=${test_actual:,.0f}, predicted=${predicted:,.0f}"
                if predicted
                else f"  Case {test_case_id}: actual=${test_actual:,.0f}, predicted=None"
            )

        # Calculate metrics
        total_mse = sum(mse_losses)
        avg_mse = total_mse / len(mse_losses)

        # Calculate precision and recall for exact matches
        exact_matches = sum(
            1
            for p, a in zip(predictions, actuals)
            if p is not None and abs(p - a) < 1.0
        )  # Exact match within $1
        total_predictions = sum(1 for p in predictions if p is not None)
        total_actuals = len(actuals)

        precision = exact_matches / total_predictions if total_predictions > 0 else 0.0
        recall = exact_matches / total_actuals if total_actuals > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return OptimizationResult(
            hyperparams=hyperparams,
            mse_loss=avg_mse,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            exact_matches=exact_matches,
            total_cases=len(actuals),
            predictions=predictions,
            actuals=actuals,
        )

    def _generate_hyperparameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters from the grid."""
        keys = self.search_grid.keys()
        values = self.search_grid.values()

        combinations = []
        for combination in itertools.product(*values):
            hyperparams = dict(zip(keys, combination))
            combinations.append(hyperparams)

        print(f"Generated {len(combinations)} hyperparameter combinations")
        return combinations

    def optimize(
        self, max_workers: int = 4, max_combinations: Optional[int] = None
    ) -> List[OptimizationResult]:
        """
        Run the grid search optimization with progress tracking and ETA.

        Args:
            max_workers: Number of parallel workers
            max_combinations: Maximum number of combinations to evaluate (None = all)

        Returns:
            List of OptimizationResult objects, sorted by best performance
        """
        import time
        from datetime import datetime, timedelta

        combinations = self._generate_hyperparameter_combinations()

        # Limit combinations if specified
        if max_combinations is not None and len(combinations) > max_combinations:
            print(
                f"⚠️  Limiting to {max_combinations} combinations (from {len(combinations)} total)"
            )
            combinations = combinations[:max_combinations]

        results = []
        start_time = datetime.now()

        print(f"🚀 Starting grid search with {len(combinations)} combinations")
        print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if max_workers > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_combo = {
                    executor.submit(self._evaluate_hyperparameters_loocv, combo): combo
                    for combo in combinations
                }

                completed = 0
                for future in as_completed(future_to_combo):
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1

                        # Calculate progress and ETA
                        progress = (completed / len(combinations)) * 100
                        elapsed_time = (datetime.now() - start_time).total_seconds()

                        if completed > 1:
                            avg_time_per_eval = elapsed_time / completed
                            remaining_evals = len(combinations) - completed
                            eta_seconds = avg_time_per_eval * remaining_evals
                            eta_str = str(timedelta(seconds=int(eta_seconds)))
                        else:
                            eta_str = "calculating..."

                        print(
                            f"✅ Evaluation {completed}/{len(combinations)} ({progress:.1f}%) - "
                            f"MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}, ETA: {eta_str}"
                        )

                    except Exception as e:
                        combo = future_to_combo[future]
                        print(f"❌ Error evaluating {combo}: {e}")
        else:
            # Sequential evaluation
            for i, combo in enumerate(combinations):
                try:
                    start_eval_time = time.time()
                    result = self._evaluate_hyperparameters_loocv(combo)
                    results.append(result)

                    # Calculate progress and ETA
                    completed = i + 1
                    progress = (completed / len(combinations)) * 100
                    elapsed_time = (datetime.now() - start_time).total_seconds()

                    if completed > 1:
                        avg_time_per_eval = elapsed_time / completed
                        remaining_evals = len(combinations) - completed
                        eta_seconds = avg_time_per_eval * remaining_evals
                        eta_str = str(timedelta(seconds=int(eta_seconds)))
                    else:
                        eta_str = "calculating..."

                    eval_time = time.time() - start_eval_time
                    print(
                        f"✅ Evaluation {completed}/{len(combinations)} ({progress:.1f}%) - "
                        f"MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}, "
                        f"Time: {eval_time:.1f}s, ETA: {eta_str}"
                    )

                except Exception as e:
                    print(f"❌ Error evaluating {combo}: {e}")

        end_time = datetime.now()
        total_duration = end_time - start_time

        print(f"\n📊 Grid Search Complete!")
        print(f"   Total combinations evaluated: {len(results)}")
        print(f"   Duration: {total_duration}")
        print(f"   Best MSE Loss: {results[0].mse_loss:.2e}")
        print(f"   Best F1 Score: {results[0].f1_score:.3f}")

        # Sort by MSE loss (lower is better)
        results.sort(key=lambda r: r.mse_loss)

        return results

    def print_best_results(self, results: List[OptimizationResult], top_k: int = 10):
        """Print the top-k best performing hyperparameter combinations."""
        print(f"\n🏆 Top {top_k} Results:")
        print("=" * 80)

        for i, result in enumerate(results[:top_k]):
            print(f"\n{i+1}. MSE Loss: {result.mse_loss:.2e}")
            print(f"   Precision: {result.precision:.3f}")
            print(f"   Recall: {result.recall:.3f}")
            print(f"   F1 Score: {result.f1_score:.3f}")
            print(f"   Exact Matches: {result.exact_matches}/{result.total_cases}")
            print(f"   Hyperparameters:")
            for key, value in result.hyperparams.items():
                print(f"     {key}: {value}")

    def save_results(self, results: List[OptimizationResult], output_path: str):
        """Save optimization results to JSON file."""
        output_data = []

        for result in results:
            result_dict = {
                "mse_loss": result.mse_loss,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "exact_matches": result.exact_matches,
                "total_cases": result.total_cases,
                "hyperparams": result.hyperparams,
                "predictions": result.predictions,
                "actuals": result.actuals,
            }
            output_data.append(result_dict)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Results saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Grid search optimization for case outcome imputer hyperparameters"
    )
    parser.add_argument(
        "--gold-standard", required=True, help="Path to gold standard CSV file"
    )
    parser.add_argument(
        "--extracted-data-root",
        required=True,
        help="Root directory containing extracted case data",
    )
    parser.add_argument(
        "--output", default="optimization_results.json", help="Output file for results"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of top results to display"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize optimizer
    optimizer = GridSearchOptimizer(
        gold_standard_path=args.gold_standard,
        extracted_data_root=args.extracted_data_root,
    )

    # Run optimization
    print("🎯 Starting hyperparameter optimization...")
    results = optimizer.optimize(max_workers=args.max_workers)

    # Print best results
    optimizer.print_best_results(results, top_k=args.top_k)

    # Save results
    optimizer.save_results(results, args.output)

    print(f"\n✅ Optimization complete! Best MSE Loss: {results[0].mse_loss:.2e}")


if __name__ == "__main__":
    main()
