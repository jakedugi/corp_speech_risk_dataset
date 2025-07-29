#!/usr/bin/env python3
"""
bayesian_optimizer.py

Intelligent Bayesian hyperparameter optimization for case outcome imputer.
Uses scikit-optimize for efficient hyperparameter search with progress tracking,
ETA estimation, and comprehensive reporting.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import pickle

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️  scikit-optimize not available. Install with: pip install scikit-optimize")

import sys
import os

sys.path.append(os.path.dirname(__file__))

from grid_search_optimizer import GridSearchOptimizer, OptimizationResult

# Import VotingWeights from the extract_cash_amounts_stage1 module
try:
    from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
        VotingWeights,
    )
except ImportError:
    # Fallback for direct execution
    from extract_cash_amounts_stage1 import VotingWeights


@dataclass
class BayesianOptimizationResult:
    """Results from Bayesian optimization."""

    best_hyperparams: Dict[str, Any]
    best_score: float
    all_results: List[OptimizationResult]
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    max_evaluations: int
    start_time: datetime
    end_time: datetime
    progress_percentage: float
    eta_seconds: float


class BayesianOptimizer:
    """
    Intelligent Bayesian hyperparameter optimizer with progress tracking and reporting.
    """

    def __init__(
        self,
        gold_standard_path: str,
        extracted_data_root: str,
        max_evaluations: int = 100,
        random_state: int = 42,
        fast_mode: bool = False,  # New parameter for optimization speed
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            gold_standard_path: Path to gold standard CSV
            extracted_data_root: Root directory containing extracted case data
            max_evaluations: Maximum number of hyperparameter combinations to evaluate
            random_state: Random seed for reproducibility
            fast_mode: If True, reduces logging overhead for faster optimization
        """
        self.gold_standard_path = Path(gold_standard_path)
        self.extracted_data_root = Path(extracted_data_root)
        self.max_evaluations = max_evaluations
        self.random_state = random_state
        self.fast_mode = fast_mode

        # Initialize tracking variables
        self.best_score = float("inf")
        self.best_hyperparams = None
        self.best_result = None
        self.evaluation_history = []
        self.current_evaluation = 0
        self.start_time = None

        # Setup base optimizer
        self.base_optimizer = GridSearchOptimizer(
            str(self.gold_standard_path), str(self.extracted_data_root)
        )

        # Define search space
        self.search_space = self._define_search_space()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for optimization progress."""
        log_file = (
            f"logs/bayesian_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        Path("logs").mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def _define_search_space(self) -> List:
        """
        Define the hyperparameter search space for Bayesian optimization.
        Uses scikit-optimize space definitions for efficient exploration.
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")

        return [
            # Core extraction parameters
            Integer(900, 2400, name="min_amount"),  # $1k to $100k
            Integer(300, 600, name="context_chars"),  # 50 to 500 chars
            Integer(2, 7, name="min_features"),  # 1 to 5 features
            # Position thresholds (0.1 to 0.9)
            Real(0.40, 0.9, name="case_position_threshold"),
            Real(0.10, 0.9, name="docket_position_threshold"),
            # Case flag thresholds
            Real(1.0, 1.1, name="fee_shifting_ratio_threshold"),
            Real(1.0, 1.1, name="patent_ratio_threshold"),
            Real(0.4, 0.8, name="dismissal_ratio_threshold"),
            Real(0.52, 0.72, name="bankruptcy_ratio_threshold"),
            # Voting weights (0.1 to 3.0)
            Real(1.0, 3.3, name="proximity_pattern_weight"),
            Real(0.4, 1.0, name="judgment_verbs_weight"),
            Real(0.8, 2.0, name="case_position_weight"),
            Real(0.1, 3.0, name="docket_position_weight"),
            Real(1.5, 3.5, name="all_caps_titles_weight"),
            Real(1.0, 3.0, name="document_titles_weight"),
            # Header size
            Integer(1000, 1700, name="header_chars"),  # 500 to 5000 chars
        ]

    def _hyperparams_to_dict(self, x: List) -> Dict[str, Any]:
        """Convert Bayesian optimization result to hyperparameter dictionary."""
        param_names = [
            "min_amount",
            "context_chars",
            "min_features",
            "case_position_threshold",
            "docket_position_threshold",
            "fee_shifting_ratio_threshold",
            "patent_ratio_threshold",
            "dismissal_ratio_threshold",
            "bankruptcy_ratio_threshold",
            "proximity_pattern_weight",
            "judgment_verbs_weight",
            "case_position_weight",
            "docket_position_weight",
            "all_caps_titles_weight",
            "document_titles_weight",
            "header_chars",
        ]

        return dict(zip(param_names, x))

    def _objective_function(self, x: List) -> float:
        """
        Objective function for Bayesian optimization.
        Returns negative MSE loss (minimization problem).
        """
        self.current_evaluation += 1

        # Convert to hyperparameter dictionary
        hyperparams = self._hyperparams_to_dict(x)

        # Create voting weights
        voting_weights = VotingWeights(
            proximity_pattern_weight=hyperparams["proximity_pattern_weight"],
            judgment_verbs_weight=hyperparams["judgment_verbs_weight"],
            case_position_weight=hyperparams["case_position_weight"],
            docket_position_weight=hyperparams["docket_position_weight"],
            all_caps_titles_weight=hyperparams["all_caps_titles_weight"],
            document_titles_weight=hyperparams["document_titles_weight"],
        )

        # Update hyperparams with voting weights
        hyperparams["voting_weights"] = voting_weights

        # Evaluate this hyperparameter combination
        try:
            result = self._evaluate_hyperparameters_loocv(hyperparams)

            # Track evaluation
            evaluation_record = {
                "evaluation": self.current_evaluation,
                "hyperparams": hyperparams,
                "mse_loss": result.mse_loss,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "exact_matches": result.exact_matches,
                "total_cases": result.total_cases,
                "timestamp": datetime.now().isoformat(),
            }
            self.evaluation_history.append(evaluation_record)

            # Check if this is the best result so far
            if result.mse_loss < self.best_score:
                self.best_score = result.mse_loss
                self.best_hyperparams = hyperparams.copy()
                self.best_result = result

                # Log the new best parameters (reduced in fast mode)
                if not self.fast_mode:
                    self.logger.info(
                        f"🏆 NEW BEST! Evaluation {self.current_evaluation}/20 - MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}"
                    )
                    self.logger.info(
                        f"   Best hyperparams: min_amount={hyperparams['min_amount']}, context_chars={hyperparams['context_chars']}, min_features={hyperparams['min_features']}"
                    )
                    self.logger.info(
                        f"   Position thresholds: case={hyperparams['case_position_threshold']:.3f}, docket={hyperparams['docket_position_threshold']:.3f}"
                    )
                    self.logger.info(
                        f"   Voting weights: proximity={hyperparams['proximity_pattern_weight']:.3f}, judgment={hyperparams['judgment_verbs_weight']:.3f}"
                    )
                    self.logger.info(
                        f"   Exact matches: {result.exact_matches}/{result.total_cases}"
                    )
                else:
                    # Minimal logging in fast mode
                    self.logger.info(
                        f"🏆 NEW BEST! E{self.current_evaluation}/20 - MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}"
                    )

            # Log progress (reduced in fast mode)
            elapsed_time = time.time() - self.start_time.timestamp()
            progress = (self.current_evaluation / self.max_evaluations) * 100

            if self.current_evaluation > 1:
                avg_time_per_eval = elapsed_time / (self.current_evaluation - 1)
                remaining_evals = self.max_evaluations - self.current_evaluation
                eta_seconds = avg_time_per_eval * remaining_evals
                eta_str = str(timedelta(seconds=int(eta_seconds)))

                if not self.fast_mode:
                    self.logger.info(
                        f"Evaluation {self.current_evaluation}/{self.max_evaluations} ({progress:.1f}%) - "
                        f"MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}, ETA: {eta_str}"
                    )
                else:
                    # Minimal logging in fast mode
                    self.logger.info(
                        f"E{self.current_evaluation}/{self.max_evaluations} ({progress:.1f}%) - "
                        f"MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}, ETA: {eta_str}"
                    )
            else:
                if not self.fast_mode:
                    self.logger.info(
                        f"Evaluation {self.current_evaluation}/{self.max_evaluations} ({progress:.1f}%) - "
                        f"MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}, ETA: calculating..."
                    )
                else:
                    # Minimal logging in fast mode
                    self.logger.info(
                        f"E{self.current_evaluation}/{self.max_evaluations} ({progress:.1f}%) - "
                        f"MSE: {result.mse_loss:.2e}, F1: {result.f1_score:.3f}, ETA: calculating..."
                    )

            # Return negative MSE for minimization (scikit-optimize minimizes)
            return result.mse_loss

        except Exception as e:
            self.logger.error(f"Error in evaluation {self.current_evaluation}: {e}")
            # Return a high penalty for failed evaluations
            return 1e12

    def _evaluate_hyperparameters_loocv(
        self, hyperparams: Dict[str, Any]
    ) -> OptimizationResult:
        """Evaluate hyperparameters using LOOCV (reuse from grid search optimizer)."""
        # This reuses the LOOCV logic from the grid search optimizer
        return self.base_optimizer._evaluate_hyperparameters_loocv(hyperparams)

    def optimize(self) -> BayesianOptimizationResult:
        """
        Run Bayesian hyperparameter optimization.

        Returns:
            BayesianOptimizationResult with optimization results
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")

        self.start_time = datetime.now()
        self.current_evaluation = 0

        self.logger.info(
            f"🚀 Starting Bayesian optimization with {self.max_evaluations} evaluations"
        )
        self.logger.info(f"Search space: {len(self.search_space)} hyperparameters")

        # Run Bayesian optimization
        result = gp_minimize(
            func=self._objective_function,
            dimensions=self.search_space,
            n_calls=self.max_evaluations,
            random_state=self.random_state,
            verbose=True,
        )

        end_time = datetime.now()

        # Find best result from evaluation history
        best_eval = min(self.evaluation_history, key=lambda x: x["mse_loss"])

        # Create optimization result
        optimization_result = BayesianOptimizationResult(
            best_hyperparams=best_eval["hyperparams"],
            best_score=best_eval["mse_loss"],
            all_results=[],  # Will be populated below
            optimization_history=self.evaluation_history,
            total_evaluations=len(self.evaluation_history),
            max_evaluations=self.max_evaluations,
            start_time=self.start_time,
            end_time=end_time,
            progress_percentage=100.0,
            eta_seconds=0.0,
        )

        # Convert evaluation history to OptimizationResult objects
        for eval_record in self.evaluation_history:
            opt_result = OptimizationResult(
                hyperparams=eval_record["hyperparams"],
                mse_loss=eval_record["mse_loss"],
                precision=eval_record["precision"],
                recall=eval_record["recall"],
                f1_score=eval_record["f1_score"],
                exact_matches=eval_record["exact_matches"],
                total_cases=eval_record["total_cases"],
                predictions=[],  # Not tracked in Bayesian optimization
                actuals=[],  # Not tracked in Bayesian optimization
            )
            optimization_result.all_results.append(opt_result)

        # Sort results by MSE loss
        optimization_result.all_results.sort(key=lambda r: r.mse_loss)

        # Print best parameters summary
        self.print_best_parameters_summary()

        return optimization_result

    def print_optimization_report(self, result: BayesianOptimizationResult):
        """Print comprehensive optimization report."""
        print("\n" + "=" * 80)
        print("🎯 BAYESIAN OPTIMIZATION REPORT")
        print("=" * 80)

        # Summary statistics
        print(f"\n📊 Optimization Summary:")
        print(
            f"   Total evaluations: {result.total_evaluations}/{result.max_evaluations}"
        )
        print(f"   Duration: {result.end_time - result.start_time}")
        print(f"   Best MSE Loss: {result.best_score:.2e}")

        # Best hyperparameters
        print(f"\n🏆 Best Hyperparameters:")
        for key, value in result.best_hyperparams.items():
            if key != "voting_weights":
                print(f"   {key}: {value}")

        # Voting weights
        voting_weights = result.best_hyperparams.get("voting_weights")
        if voting_weights:
            print(f"\n⚖️  Best Voting Weights:")
            for key, value in voting_weights.to_dict().items():
                print(f"   {key}: {value}")

        # Top 5 results
        print(f"\n🥇 Top 5 Results:")
        for i, opt_result in enumerate(result.all_results[:5]):
            print(
                f"   {i+1}. MSE: {opt_result.mse_loss:.2e}, "
                f"F1: {opt_result.f1_score:.3f}, "
                f"Precision: {opt_result.precision:.3f}, "
                f"Recall: {opt_result.recall:.3f}"
            )

        # Hyperparameter importance analysis
        self._analyze_hyperparameter_importance(result)

        # Progress analysis
        self._analyze_optimization_progress(result)

    def _analyze_hyperparameter_importance(self, result: BayesianOptimizationResult):
        """Analyze the importance of different hyperparameters."""
        self.logger.info("🔍 Hyperparameter Impact Analysis:")

        # Analyze hyperparameter importance based on evaluation history
        for eval_record in result.optimization_history:
            for param_name, param_value in eval_record["hyperparams"].items():
                if param_name == "voting_weights":
                    continue

                if param_name not in param_analysis:
                    param_analysis[param_name] = []

                param_analysis[param_name].append(
                    {"value": param_value, "mse_loss": eval_record["mse_loss"]}
                )

        # Find most impactful parameters
        impact_scores = {}
        for param_name, values in param_analysis.items():
            if len(values) > 1:
                # Calculate correlation between parameter value and MSE loss
                param_values = [v["value"] for v in values]
                mse_losses = [v["mse_loss"] for v in values]

                # Simple impact measure: range of MSE losses for this parameter
                mse_range = max(mse_losses) - min(mse_losses)
                impact_scores[param_name] = mse_range

        # Sort by impact
        sorted_impact = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"   Most impactful hyperparameters:")
        for param_name, impact in sorted_impact[:5]:
            print(f"     {param_name}: {impact:.2e}")

    def _analyze_optimization_progress(self, result: BayesianOptimizationResult):
        """Analyze optimization progress and convergence."""
        print(f"\n📈 Optimization Progress Analysis:")

        # Plot progress over time
        mse_losses = [
            eval_record["mse_loss"] for eval_record in result.optimization_history
        ]
        evaluations = list(range(1, len(mse_losses) + 1))

        # Calculate convergence metrics
        best_mse = min(mse_losses)
        initial_mse = mse_losses[0]
        improvement = (initial_mse - best_mse) / initial_mse * 100

        print(f"   Initial MSE: {initial_mse:.2e}")
        print(f"   Best MSE: {best_mse:.2e}")
        print(f"   Improvement: {improvement:.1f}%")

        # Find when best result was found
        best_eval_idx = mse_losses.index(best_mse)
        print(f"   Best result found at evaluation {best_eval_idx + 1}")

        # Convergence analysis
        if len(mse_losses) > 10:
            recent_improvement = (mse_losses[-10] - best_mse) / mse_losses[-10] * 100
            print(f"   Recent improvement (last 10 evals): {recent_improvement:.1f}%")

    def save_results(self, result: BayesianOptimizationResult, output_path: str):
        """Save optimization results to JSON file."""
        # Convert to serializable format
        output_data = {
            "best_hyperparams": result.best_hyperparams,
            "best_score": result.best_score,
            "optimization_history": result.optimization_history,
            "total_evaluations": result.total_evaluations,
            "max_evaluations": result.max_evaluations,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "progress_percentage": result.progress_percentage,
            "eta_seconds": result.eta_seconds,
        }

        # Convert voting weights to dict for serialization
        if "voting_weights" in output_data["best_hyperparams"]:
            output_data["best_hyperparams"]["voting_weights"] = output_data[
                "best_hyperparams"
            ]["voting_weights"].to_dict()

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Results saved to {output_path}")

    def get_current_best_parameters(self) -> Dict[str, Any]:
        """Get the current best performing parameters."""
        if self.best_hyperparams is None:
            return {}

        return {
            "hyperparams": self.best_hyperparams,
            "mse_loss": self.best_score,
            "f1_score": self.best_result.f1_score if self.best_result else 0.0,
            "precision": self.best_result.precision if self.best_result else 0.0,
            "recall": self.best_result.recall if self.best_result else 0.0,
            "exact_matches": self.best_result.exact_matches if self.best_result else 0,
            "total_cases": self.best_result.total_cases if self.best_result else 0,
        }

    def print_best_parameters_summary(self):
        """Print a summary of the best performing parameters found."""
        if self.best_hyperparams is None:
            self.logger.warning(
                "No best parameters found - optimization may have failed"
            )
            return

        self.logger.info("=" * 80)
        self.logger.info("🏆 BEST PERFORMING PARAMETERS SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Best MSE Loss: {self.best_score:.2e}")
        self.logger.info(f"Best F1 Score: {self.best_result.f1_score:.3f}")
        self.logger.info(f"Best Precision: {self.best_result.precision:.3f}")
        self.logger.info(f"Best Recall: {self.best_result.recall:.3f}")
        self.logger.info(
            f"Exact Matches: {self.best_result.exact_matches}/{self.best_result.total_cases}"
        )
        self.logger.info("")
        self.logger.info("📊 Best Hyperparameters:")
        self.logger.info(f"  Core Parameters:")
        self.logger.info(f"    min_amount: {self.best_hyperparams['min_amount']:,}")
        self.logger.info(f"    context_chars: {self.best_hyperparams['context_chars']}")
        self.logger.info(f"    min_features: {self.best_hyperparams['min_features']}")
        self.logger.info(f"    header_chars: {self.best_hyperparams['header_chars']}")
        self.logger.info(f"  Position Thresholds:")
        self.logger.info(
            f"    case_position_threshold: {self.best_hyperparams['case_position_threshold']:.3f}"
        )
        self.logger.info(
            f"    docket_position_threshold: {self.best_hyperparams['docket_position_threshold']:.3f}"
        )
        self.logger.info(f"  Case Flag Thresholds:")
        self.logger.info(
            f"    fee_shifting_ratio_threshold: {self.best_hyperparams['fee_shifting_ratio_threshold']:.3f}"
        )
        self.logger.info(
            f"    patent_ratio_threshold: {self.best_hyperparams['patent_ratio_threshold']:.3f}"
        )
        self.logger.info(
            f"    dismissal_ratio_threshold: {self.best_hyperparams['dismissal_ratio_threshold']:.3f}"
        )
        self.logger.info(
            f"    bankruptcy_ratio_threshold: {self.best_hyperparams['bankruptcy_ratio_threshold']:.3f}"
        )
        self.logger.info(f"  Voting Weights:")
        self.logger.info(
            f"    proximity_pattern_weight: {self.best_hyperparams['proximity_pattern_weight']:.3f}"
        )
        self.logger.info(
            f"    judgment_verbs_weight: {self.best_hyperparams['judgment_verbs_weight']:.3f}"
        )
        self.logger.info(
            f"    case_position_weight: {self.best_hyperparams['case_position_weight']:.3f}"
        )
        self.logger.info(
            f"    docket_position_weight: {self.best_hyperparams['docket_position_weight']:.3f}"
        )
        self.logger.info(
            f"    all_caps_titles_weight: {self.best_hyperparams['all_caps_titles_weight']:.3f}"
        )
        self.logger.info(
            f"    document_titles_weight: {self.best_hyperparams['document_titles_weight']:.3f}"
        )
        self.logger.info("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for case outcome imputer"
    )
    parser.add_argument(
        "--gold-standard",
        default="data/gold_standard/case_outcome_amounts_hand_annotated.csv",
        help="Path to gold standard CSV file",
    )
    parser.add_argument(
        "--extracted-data-root",
        default="data/extracted",
        help="Root directory containing extracted case data",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=50,
        help="Maximum number of hyperparameter combinations to evaluate",
    )
    parser.add_argument(
        "--output",
        default="bayesian_optimization_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    """Main function for Bayesian optimization."""
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter optimization")
    parser.add_argument(
        "--gold-standard", required=True, help="Path to gold standard CSV"
    )
    parser.add_argument(
        "--extracted-data", required=True, help="Path to extracted data root"
    )
    parser.add_argument(
        "--max-evaluations", type=int, default=100, help="Maximum evaluations"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fast-mode", action="store_true", help="Enable fast mode for optimization"
    )
    parser.add_argument("--output", help="Output path for results")

    args = parser.parse_args()

    # Create optimizer with fast mode
    optimizer = BayesianOptimizer(
        gold_standard_path=args.gold_standard,
        extracted_data_root=args.extracted_data,
        max_evaluations=args.max_evaluations,
        random_state=args.random_state,
        fast_mode=args.fast_mode,  # Enable fast mode
    )

    # Run optimization
    result = optimizer.optimize()

    # Print results
    optimizer.print_optimization_report(result)

    # Save results if output path provided
    if args.output:
        optimizer.save_results(result, args.output)


if __name__ == "__main__":
    main()
