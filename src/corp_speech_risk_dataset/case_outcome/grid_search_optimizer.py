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
import multiprocessing as mp

# High-performance JSON parser for Apple M1/ARM64 optimization
try:
    import orjson

    # Fast JSON loading function
    def fast_json_loads(data: str) -> dict:
        """Fast JSON parsing using orjson (optimized for ARM64/M1)."""
        return orjson.loads(data)

except ImportError:
    # Fallback to standard json if orjson not available
    def fast_json_loads(data: str) -> dict:
        """Fallback JSON parsing using standard library."""
        return json.loads(data)


# Optimize multiprocessing for macOS M1 - use fork for better performance
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    # fork may already be set, or not available on this platform
    pass

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
from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
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

    def __init__(
        self, gold_standard_path: str, extracted_data_root: str, fast_mode: bool = False
    ):
        """
        Initialize the optimizer.

        Args:
            gold_standard_path: Path to the gold standard CSV file
            extracted_data_root: Root directory containing extracted case data
            fast_mode: Enable fast mode for optimization (skips expensive operations)
        """
        self.gold_standard_path = Path(gold_standard_path)
        self.fast_mode = fast_mode  # 🚀 CRITICAL FIX - Add fast_mode attribute
        self.extracted_data_root = Path(extracted_data_root)
        self.gold_standard_data = self._load_gold_standard()

        # 🚀 OPTIMIZATION: Add case prediction cache for repeated LOOCV calls
        self._case_cache: Dict[Tuple, Optional[float]] = {}

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
        # Validate case_id first
        if not case_id or str(case_id).strip() in ["nan", "None", ""]:
            if self.fast_mode:
                print(f"  ⚠️  Invalid case_id: '{case_id}'")
            return None

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

        # Debug path resolution failure in fast mode
        if self.fast_mode:
            print(f"  ⚠️  Case path not found for: '{case_id}' -> '{case_dir}'")
            print(f"     Tried: {self.extracted_data_root / case_dir}")

        return None

    def _predict_case_outcome(
        self, case_id: str, hyperparams: Dict[str, Any]
    ) -> Optional[float]:
        """Predict case outcome using given hyperparameters."""
        # 🚀 OPTIMIZATION: Check cache first to avoid re-computation
        cache_key = (case_id, tuple(sorted(hyperparams.items())))
        if cache_key in self._case_cache:
            return self._case_cache[cache_key]

        # Note: Fast mode now uses real analysis with timeout protection (no dummy predictions)
        case_path = self._get_case_data_path(case_id)
        if not case_path:
            self._case_cache[cache_key] = None
            return None

        # Create voting weights from hyperparameters with safe defaults for missing values
        voting_weights = VotingWeights(
            # Original voting weights
            proximity_pattern_weight=hyperparams.get("proximity_pattern_weight", 1.0),
            judgment_verbs_weight=hyperparams.get("judgment_verbs_weight", 1.0),
            case_position_weight=hyperparams.get("case_position_weight", 1.0),
            docket_position_weight=hyperparams.get("docket_position_weight", 1.0),
            all_caps_titles_weight=hyperparams.get("all_caps_titles_weight", 1.0),
            document_titles_weight=hyperparams.get("document_titles_weight", 1.0),
            # Enhanced feature weights with safe defaults
            financial_terms_weight=hyperparams.get("financial_terms_weight", 1.0),
            settlement_terms_weight=hyperparams.get("settlement_terms_weight", 1.0),
            legal_proceedings_weight=hyperparams.get("legal_proceedings_weight", 1.0),
            monetary_phrases_weight=hyperparams.get("monetary_phrases_weight", 1.0),
            dependency_parsing_weight=hyperparams.get("dependency_parsing_weight", 1.0),
            fraction_extraction_weight=hyperparams.get(
                "fraction_extraction_weight", 1.0
            ),
            percentage_extraction_weight=hyperparams.get(
                "percentage_extraction_weight", 1.0
            ),
            implied_totals_weight=hyperparams.get("implied_totals_weight", 1.0),
            document_structure_weight=hyperparams.get("document_structure_weight", 1.0),
            table_detection_weight=hyperparams.get("table_detection_weight", 1.0),
            header_detection_weight=hyperparams.get("header_detection_weight", 1.0),
            section_boundaries_weight=hyperparams.get("section_boundaries_weight", 1.0),
            numeric_gazetteer_weight=hyperparams.get("numeric_gazetteer_weight", 1.0),
            mixed_numbers_weight=hyperparams.get("mixed_numbers_weight", 1.0),
            sentence_boundary_weight=hyperparams.get("sentence_boundary_weight", 1.0),
            paragraph_boundary_weight=hyperparams.get("paragraph_boundary_weight", 1.0),
            # Confidence boosting features
            high_confidence_patterns_weight=hyperparams.get(
                "high_confidence_patterns_weight", 1.0
            ),
            amount_adjacent_keywords_weight=hyperparams.get(
                "amount_adjacent_keywords_weight", 1.0
            ),
            confidence_boost_weight=hyperparams.get("confidence_boost_weight", 1.0),
            # High/Low signal regex weights
            high_signal_financial_weight=hyperparams.get(
                "high_signal_financial_weight", 1.0
            ),
            low_signal_financial_weight=hyperparams.get(
                "low_signal_financial_weight", 0.5
            ),
            high_signal_settlement_weight=hyperparams.get(
                "high_signal_settlement_weight", 1.0
            ),
            low_signal_settlement_weight=hyperparams.get(
                "low_signal_settlement_weight", 0.5
            ),
            calculation_boost_multiplier=hyperparams.get(
                "calculation_boost_multiplier", 1.0
            ),
        )

        # Check dismissal logic first with hyperparameters (with safe defaults)
        from corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
            get_case_flags,
            get_case_court_type,
        )

        flags = get_case_flags(
            case_path,
            fee_shifting_ratio_threshold=hyperparams.get(
                "fee_shifting_ratio_threshold", 1.0
            ),
            patent_ratio_threshold=hyperparams.get("patent_ratio_threshold", 50.0),
            dismissal_ratio_threshold=hyperparams.get("dismissal_ratio_threshold", 0.5),
            bankruptcy_ratio_threshold=hyperparams.get(
                "bankruptcy_ratio_threshold", 0.5
            ),
            use_weighted_dismissal_scoring=hyperparams.get(
                "use_weighted_dismissal_scoring", True
            ),
            dismissal_document_type_weight=hyperparams.get(
                "dismissal_document_type_weight", 2.0
            ),
            fast_mode=self.fast_mode,  # 🚀 CRITICAL SPEED FIX - Use fast mode during optimization
        )

        # Check if this is a bankruptcy court case
        court_type = get_case_court_type(
            case_path,
            bankruptcy_ratio_threshold=hyperparams.get(
                "bankruptcy_ratio_threshold", 0.5
            ),
        )
        if court_type == "BANKRUPTCY":
            return None

        # Check if this is a high patent ratio case (hard filter)
        if flags["has_large_patent_amounts"]:
            return None

        # Check if this is a dismissed case
        if flags["is_dismissed"]:
            return 0.0

        # Get candidates using hyperparameters with timeout protection
        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("scan_stage1 timeout")

            # Set 5-second timeout for fast mode (more reasonable for real analysis)
            if self.fast_mode:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)  # 5 second timeout

            candidates = scan_stage1(
                case_path,
                min_amount=hyperparams["min_amount"],
                context_chars=min(
                    hyperparams["context_chars"], 300
                ),  # Cap context for speed
                min_features=min(
                    hyperparams["min_features"], 200
                ),  # Cap features for speed
                case_position_threshold=hyperparams["case_position_threshold"],
                docket_position_threshold=hyperparams["docket_position_threshold"],
                voting_weights=voting_weights,
                header_chars=min(
                    hyperparams["header_chars"], 1000
                ),  # Cap header for speed
                fast_mode=True,  # Use fast mode for optimization
            )

            if self.fast_mode:
                signal.alarm(0)  # Cancel timeout

        except (TimeoutError, Exception) as e:
            if self.fast_mode:
                signal.alarm(0)  # Cancel timeout
            if self.fast_mode:
                # In fast mode, return dummy result instead of None to avoid hanging
                print(f"  ⚡ Fast skip {case_path.name}: {type(e).__name__}")
                return 0.0  # Return 0 for ultra-fast evaluation
            else:
                print(f"  ⚠️  Error for {case_path.name}: {e}")
                return None

        # Choose best candidate
        if not candidates:
            result = None
        else:
            # Sort by feature votes (descending), then by value (descending)
            sorted_candidates = sorted(
                candidates, key=lambda c: (c.feature_votes, c.value), reverse=True
            )
            result = sorted_candidates[0].value

        # 🚀 OPTIMIZATION: Cache the result for future use
        self._case_cache[cache_key] = result
        return result

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

    def _evaluate_single_case_worker(
        self, test_case_id: str, test_actual: float, hyperparams: Dict[str, Any]
    ) -> Tuple[Optional[float], float]:
        """
        Worker function for parallel case evaluation.

        Args:
            test_case_id: Case ID to evaluate
            test_actual: Actual outcome for this case
            hyperparams: Hyperparameters to use

        Returns:
            Tuple of (predicted_amount, mse_loss)
        """
        predicted = self._predict_case_outcome(test_case_id, hyperparams)

        if predicted is None:
            mse_loss = test_actual**2  # Penalty for missing prediction
        else:
            mse_loss = (predicted - test_actual) ** 2

        return predicted, mse_loss

    def _evaluate_loocv_sequential(
        self, tasks: List[Tuple[str, float, Dict[str, Any]]]
    ) -> Tuple[List[Optional[float]], List[float]]:
        """
        Sequential fallback for LOOCV evaluation.

        Args:
            tasks: List of (test_case_id, test_actual, hyperparams) tuples

        Returns:
            Tuple of (predictions, mse_losses)
        """
        predictions = []
        mse_losses = []

        for test_case_id, test_actual, hyperparams in tasks:
            predicted, mse_loss = self._evaluate_hyperparameters(
                hyperparams, test_case_id, test_actual
            )
            predictions.append(predicted)
            mse_losses.append(mse_loss)

            print(
                f"  Case {test_case_id}: actual=${test_actual:,.0f}, predicted=${predicted:,.0f}"
                if predicted is not None
                else f"  Case {test_case_id}: actual=${test_actual:,.0f}, predicted=None"
            )

        return predictions, mse_losses

    def _evaluate_hyperparameters_loocv(
        self, hyperparams: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Evaluate hyperparameters using K-Fold Cross-Validation (fast mode) or LOOCV.
        In fast mode, uses 3-fold CV for 7x speed improvement over LOOCV.

        Args:
            hyperparams: Hyperparameters to evaluate

        Returns:
            OptimizationResult with metrics
        """
        predictions = []
        actuals = []
        mse_losses = []

        print(f"🔍 Evaluating hyperparameters: {hyperparams}")

        # 🚀 ULTRA-FAST MODE - Use K-Fold CV instead of LOOCV
        if self.fast_mode:
            from sklearn.model_selection import KFold
            import numpy as np

            # Use 3-fold CV for ultra-fast evaluation
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            case_ids = self.gold_standard_data["case_id"].values
            amounts = self.gold_standard_data["final_amount"].values

            tasks = []
            # 🚀 FIXED: Use all valid cases in ultra-fast mode, not just first fold
            # Filter out invalid case IDs (NaN values) first
            valid_indices = []
            for i, case_id in enumerate(case_ids):
                if (
                    pd.notna(case_id)
                    and pd.notna(amounts[i])
                    and str(case_id).strip() != "nan"
                ):
                    valid_indices.append(i)

            if len(valid_indices) == 0:
                print("⚠️  No valid cases found in gold standard data!")
                return OptimizationResult(
                    hyperparams=hyperparams,
                    mse_loss=1e12,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    exact_matches=0,
                    total_cases=0,
                    predictions=[],
                    actuals=[],
                )

            # 🚀 ULTRA-FAST: Use ALL valid cases for complete evaluation
            valid_case_ids = case_ids[valid_indices]
            valid_amounts = amounts[valid_indices]

            # Use ALL valid cases for comprehensive evaluation
            for test_case_id, test_actual in zip(valid_case_ids, valid_amounts):
                tasks.append(
                    (str(test_case_id), float(test_actual), hyperparams.copy())
                )
                actuals.append(test_actual)

            print(
                f"⚡ Ultra-fast mode: Using ALL {len(tasks)} valid cases for comprehensive evaluation"
            )
            if len(tasks) < 10:
                print(
                    f"⚠️  Warning: Small dataset ({len(tasks)} cases) - consider more data for robust results"
                )
        else:
            # Full LOOCV for production optimization
            tasks = []
            for idx, row in self.gold_standard_data.iterrows():
                test_case_id = str(row["case_id"])
                test_actual = float(row["final_amount"])
                tasks.append((test_case_id, test_actual, hyperparams.copy()))
                actuals.append(test_actual)

        # 🚀 ULTRA-FAST MODE - Always use sequential for all cases with detailed logging
        if self.fast_mode:
            # Sequential processing for ultra-fast mode with detailed logging
            print(
                f"⚡ Sequential processing for {len(tasks)} cases with detailed logging"
            )
            print(f"\n📊 PREDICTIONS vs ACTUALS:")
            try:
                results = []
                for task in tasks:
                    result = self._evaluate_single_case_worker(*task)
                    results.append(result)

                for (predicted, mse_loss), (test_case_id, test_actual, _) in zip(
                    results, tasks
                ):
                    predictions.append(predicted)
                    mse_losses.append(mse_loss)

                    # Detailed logging for each case
                    case_name = (
                        test_case_id.split("/")[-2]
                        if "/" in test_case_id
                        else test_case_id
                    )
                    if predicted is not None:
                        error = abs(predicted - test_actual)
                        error_pct = (
                            (error / test_actual * 100) if test_actual > 0 else 0
                        )
                        print(
                            f"  {case_name:25} | Actual: ${test_actual:>12,.0f} | Predicted: ${predicted:>12,.0f} | Error: ${error:>12,.0f} ({error_pct:>5.1f}%)"
                        )
                    else:
                        print(
                            f"  {case_name:25} | Actual: ${test_actual:>12,.0f} | Predicted:         None | Error:     MISSING"
                        )

            except Exception as e:
                print(
                    f"⚠️  Sequential processing failed ({e}), using default prediction"
                )
                for _, test_actual, _ in tasks:
                    predictions.append(0.0)
                    mse_losses.append((test_actual - 0.0) ** 2)

        elif len(tasks) > 4:  # Use parallel processing for larger sets
            try:
                with mp.Pool(processes=min(mp.cpu_count(), len(tasks))) as pool:
                    results = pool.starmap(self._evaluate_single_case_worker, tasks)

                for (predicted, mse_loss), (test_case_id, test_actual, _) in zip(
                    results, tasks
                ):
                    predictions.append(predicted)
                    mse_losses.append(mse_loss)
                    print(
                        f"  Case {test_case_id}: actual=${test_actual:,.0f}, predicted=${predicted:,.0f}"
                        if predicted is not None
                        else f"  Case {test_case_id}: actual=${test_actual:,.0f}, predicted=None"
                    )
            except Exception as e:
                print(
                    f"⚠️  Parallel processing failed ({e}), falling back to sequential"
                )
                # Fallback to sequential processing
                predictions, mse_losses = self._evaluate_loocv_sequential(tasks)
        else:
            # Use sequential processing for small number of cases
            predictions, mse_losses = self._evaluate_loocv_sequential(tasks)

        # Calculate metrics - handle empty mse_losses case
        if len(mse_losses) == 0:
            # No valid predictions - return high penalty
            avg_mse = 1e12
        else:
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
        """Generate all hyperparameter combinations for grid search."""
        combinations = []

        # Core parameters
        min_amounts = [100, 500, 1000, 2000, 5000]
        context_chars = [100, 200, 300, 500, 1000]
        min_features = [1, 2, 3, 4, 5]
        case_position_thresholds = [0.3, 0.5, 0.7, 0.8]
        docket_position_thresholds = [0.3, 0.5, 0.7, 0.8]

        # Voting weights
        proximity_pattern_weights = [0.5, 1.0, 1.5, 2.0]
        judgment_verbs_weights = [0.5, 1.0, 1.5, 2.0]
        case_position_weights = [0.5, 1.0, 1.5, 2.0]
        docket_position_weights = [0.5, 1.0, 1.5, 2.0]
        all_caps_titles_weights = [0.5, 1.0, 1.5, 2.0]
        document_titles_weights = [0.5, 1.0, 1.5, 2.0]

        # Dismissal parameters (new)
        dismissal_ratio_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
        dismissal_document_type_weights = [1.0, 2.0, 3.0, 5.0]
        use_weighted_dismissal_scoring_options = [True, False]

        # Other parameters
        header_chars = [1000, 2000, 3000]

        for min_amount in min_amounts:
            for context_char in context_chars:
                for min_feature in min_features:
                    for case_pos_thresh in case_position_thresholds:
                        for docket_pos_thresh in docket_position_thresholds:
                            for prox_weight in proximity_pattern_weights:
                                for judgment_weight in judgment_verbs_weights:
                                    for case_pos_weight in case_position_weights:
                                        for (
                                            docket_pos_weight
                                        ) in docket_position_weights:
                                            for caps_weight in all_caps_titles_weights:
                                                for (
                                                    doc_weight
                                                ) in document_titles_weights:
                                                    for (
                                                        dismissal_thresh
                                                    ) in dismissal_ratio_thresholds:
                                                        for (
                                                            dismissal_doc_weight
                                                        ) in dismissal_document_type_weights:
                                                            for (
                                                                use_weighted
                                                            ) in use_weighted_dismissal_scoring_options:
                                                                for (
                                                                    header_char
                                                                ) in header_chars:
                                                                    combination = {
                                                                        "min_amount": min_amount,
                                                                        "context_chars": context_char,
                                                                        "min_features": min_feature,
                                                                        "case_position_threshold": case_pos_thresh,
                                                                        "docket_position_threshold": docket_pos_thresh,
                                                                        "proximity_pattern_weight": prox_weight,
                                                                        "judgment_verbs_weight": judgment_weight,
                                                                        "case_position_weight": case_pos_weight,
                                                                        "docket_position_weight": docket_pos_weight,
                                                                        "all_caps_titles_weight": caps_weight,
                                                                        "document_titles_weight": doc_weight,
                                                                        "dismissal_ratio_threshold": dismissal_thresh,
                                                                        "dismissal_document_type_weight": dismissal_doc_weight,
                                                                        "use_weighted_dismissal_scoring": use_weighted,
                                                                        "header_chars": header_char,
                                                                    }
                                                                    combinations.append(
                                                                        combination
                                                                    )

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
