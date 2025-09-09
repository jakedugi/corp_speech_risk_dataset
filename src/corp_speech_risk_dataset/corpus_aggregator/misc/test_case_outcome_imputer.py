#!/usr/bin/env python3
"""
test_case_outcome_imputer.py

Hyperparameter tuning harness for case outcome imputation.
Loads hand-annotated final amounts, converts them to floats,
runs the imputation pipeline over each case, computes F1 score
(classification: award vs zero) and MSE, and uses Gaussian-process
Bayesian optimization to find optimal pipeline hyperparameters.

Usage:
    python test_case_outcome_imputer.py \
        --annotations case_outcome_amounts_hand_annotated.csv \
        --tokenized-root /path/to/tokenized/root \
        --extracted-root /path/to/extracted/root

Requires:
    pip install scikit-optimize scikit-learn pandas numpy
"""
import argparse
import csv
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from concurrent.futures import ProcessPoolExecutor

# Import your imputation functions
from corp_speech_risk_dataset.corpus_extractors.case_outcome_imputer import (
    scan_stage1,
    AmountSelector,
    DEFAULT_VOTING_WEIGHTS,
)
from corp_speech_risk_dataset.corpus_extractors.extract_cash_amounts_stage1 import (
    VotingWeights,
    is_case_dismissed,
    is_case_definitively_dismissed,
    has_large_patent_amounts,
)


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Load hand-annotated CSV with columns ['case_id', 'final_amount', ...],
    convert 'final_amount' from string to float.
    """
    import polars as pl

    # Read CSV without strict schema to handle extra columns and null values
    df = (
        (
            pl.read_csv(csv_path, truncate_ragged_lines=True)
            .with_columns(
                [
                    pl.col("final_amount")
                    .str.replace_all(",", "")
                    .str.replace("null", "0")  # Replace "null" strings with "0"
                    .cast(pl.Float64, strict=False)  # Use non-strict casting
                ]
            )
            .filter(
                pl.col("final_amount").is_not_null()
            )  # Remove any remaining null rows
        )
        .select(["case_id", "final_amount"])
        .to_pandas()
    )
    return df


def impute_pipeline(
    case_id: str,
    tokenized_root: Path,
    extracted_root: Path,
    min_amount: float,
    context_chars: int,
    min_features: int,
    case_position_threshold: float,
    docket_position_threshold: float,
    return_stats: bool = False,
    disable_spacy: bool = False,
    disable_spelled: bool = False,
    disable_usd: bool = False,
    disable_calcs: bool = False,
    disable_regex: bool = False,
) -> Union[float, tuple[float, int, float]]:
    """
    For a single case, run scan_stage1 and select the final amount.
    Returns predicted float amount (or 0.0 if None).
    If return_stats=True, returns (amount, candidate_count, avg_top5_votes).
    """
    case_root_extracted = extracted_root / case_id
    # Scan for candidate amounts
    candidates = scan_stage1(
        case_root_extracted,
        min_amount=min_amount,
        context_chars=context_chars,
        min_features=min_features,
        case_position_threshold=case_position_threshold,
        docket_position_threshold=docket_position_threshold,
        voting_weights=DEFAULT_VOTING_WEIGHTS,
        disable_spacy=disable_spacy,
        disable_spelled=disable_spelled,
        disable_usd=disable_usd,
        disable_calcs=disable_calcs,
        disable_regex=disable_regex,
    )
    selector = AmountSelector()
    chosen = selector.choose(candidates)
    predicted_amount = float(chosen) if chosen is not None else 0.0

    if return_stats:
        candidate_count = len(candidates)
        # Calculate average top-5 vote counts
        if candidates:
            top5_votes = [c.feature_votes for c in candidates[:5]]
            avg_top5_votes = sum(top5_votes) / len(top5_votes) if top5_votes else 0.0
        else:
            avg_top5_votes = 0.0
        return predicted_amount, candidate_count, avg_top5_votes

    return predicted_amount


def evaluate_case(args):
    """
    Helper function for parallel case evaluation.
    Returns (true_amt, predicted_amt, case_name, raw_cand_count, filtered_cand_count, in_raw, in_filtered)
    """
    (
        case_id,
        true_amt,
        extracted_root,
        min_amount,
        context_chars,
        min_features,
        case_pos,
        docket_pos,
        voting_weights_dict,
        dismissal_params,
    ) = args
    voting_weights = (
        VotingWeights.from_dict(voting_weights_dict)
        if voting_weights_dict
        else DEFAULT_VOTING_WEIGHTS
    )

    # Check for dismissal
    if dismissal_params:
        # Extract just the case name from the full path (same logic as below)
        if "courtlistener/" in case_id:
            case_name = case_id.split("courtlistener/")[-1].rstrip("/")
        else:
            case_name = case_id

        case_path = extracted_root / case_name

        # Check if we can find any stage1 files
        stage1_files = (
            list(case_path.rglob("*_stage1.jsonl")) if case_path.exists() else []
        )

        if stage1_files:
            is_dismissed_flag = is_case_dismissed(
                case_path,
                dismissal_ratio_threshold=dismissal_params.get(
                    "dismissal_ratio_threshold", 0.5
                ),
                use_weighted_scoring=True,
                document_type_weight=dismissal_params.get(
                    "dismissal_document_type_weight", 2.0
                ),
            )

            is_definitively_dismissed_flag = is_case_definitively_dismissed(
                case_path,
                strict_dismissal_threshold=dismissal_params.get(
                    "strict_dismissal_threshold", 0.8
                ),
                document_type_weight=dismissal_params.get(
                    "strict_dismissal_document_type_weight", 3.0
                ),
            )

            if is_dismissed_flag or is_definitively_dismissed_flag:
                return (true_amt, 0.0, case_name, 0, 0, False, False, "DISMISSED")

            # Check for large patent amounts
            has_large_patent_amounts_flag = has_large_patent_amounts(
                case_path,
                patent_ratio_threshold=dismissal_params.get(
                    "patent_ratio_threshold", 50.0
                ),
            )

            if has_large_patent_amounts_flag:
                return (true_amt, 0.0, case_name, 0, 0, False, False, "PATENT")

    # Skip if true_amt is NaN
    if pd.isna(true_amt):
        return None

    # Extract just the case name from the full path
    if "courtlistener/" in case_id:
        case_name = case_id.split("courtlistener/")[-1].rstrip("/")
    else:
        case_name = case_id

    extracted_case_root = extracted_root / case_name

    # Check if data exists
    if not extracted_case_root.exists():
        return (true_amt, 0.0, case_name, 0, 0, False, False, "MISSING_DATA")

    stage1_files_found = list(extracted_case_root.rglob("*_stage1.jsonl"))
    if not stage1_files_found:
        return (true_amt, 0.0, case_name, 0, 0, False, False, "NO_STAGE1_FILES")

    # Run unfiltered scan to get total candidate count
    raw_cands = scan_stage1(
        extracted_case_root,
        min_amount,
        context_chars,
        min_features=0,
        case_position_threshold=case_pos,
        docket_position_threshold=docket_pos,
        voting_weights=voting_weights,
        disable_spacy=False,
        disable_spelled=False,
        disable_usd=False,
        disable_calcs=False,
        disable_regex=False,
    )

    # Run filtered scan with actual min_features
    filtered_cands = scan_stage1(
        extracted_case_root,
        min_amount,
        context_chars,
        min_features=min_features,
        case_position_threshold=case_pos,
        docket_position_threshold=docket_pos,
        voting_weights=voting_weights,
        disable_spacy=False,
        disable_spelled=False,
        disable_usd=False,
        disable_calcs=False,
        disable_regex=False,
    )

    # Pick the predicted amount
    predicted_amt = AmountSelector().choose(filtered_cands)
    predicted_amt = float(predicted_amt) if predicted_amt is not None else 0.0

    # Check if actual amount is in candidate lists
    in_raw = any(
        math.isclose(c.value, true_amt, rel_tol=1e-4, abs_tol=0.01) for c in raw_cands
    )
    in_filtered = any(
        math.isclose(c.value, true_amt, rel_tol=1e-4, abs_tol=0.01)
        for c in filtered_cands
    )

    return (
        true_amt,
        predicted_amt,
        case_name,
        len(raw_cands),
        len(filtered_cands),
        in_raw,
        in_filtered,
        "SUCCESS",
    )


def evaluate(
    df: pd.DataFrame,
    tokenized_root: Path,
    extracted_root: Path,
    params: list[float | int],
    ann_map: dict[str, float] | None = None,
    voting_weights: VotingWeights = DEFAULT_VOTING_WEIGHTS,
    dismissal_params: dict | None = None,
) -> float:
    """
    Evaluate objective for Bayesian optimization.
    params = [min_amount, context_chars, min_features, case_pos_thresh, docket_pos_thresh]
    Returns objective: -F1 + (MSE / max_true)
    """
    min_amount, context_chars, min_features, case_pos, docket_pos = params
    # Convert to proper types
    context_chars = int(context_chars)
    min_features = int(min_features)

    preds = []
    trues = []
    max_true = df["final_amount"].max() if df["final_amount"].max() > 0 else 1.0

    # Print header for detailed output during optimization
    print("\n--- Optimization Iteration Case Details ---")
    print(
        f"Params: min_amount={min_amount:.0f}, context_chars={context_chars}, min_features={min_features}, case_pos={case_pos:.2f}, docket_pos={docket_pos:.2f}"
    )
    print("------------------------------------------")
    print(
        f"{'Case ID':<30} {'Actual':>12} {'Predicted':>12} {'Cands':>7} {'InRaw':>6} {'InFilt':>7}"
    )

    # Prepare tasks for parallel processing
    tasks = [
        (
            case_id,
            true_amt,
            extracted_root,
            min_amount,
            context_chars,
            min_features,
            case_pos,
            docket_pos,
            voting_weights.to_dict(),
            dismissal_params,
        )
        for case_id, true_amt in zip(df["case_id"].astype(str), df["final_amount"])
        if not pd.isna(true_amt)
    ]

    # Process cases in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_case, tasks))

    # Filter out None results and process outputs
    for result in results:
        if result is None:
            continue

        (
            true_amt,
            predicted_amt,
            case_name,
            raw_count,
            filtered_count,
            in_raw,
            in_filtered,
            status,
        ) = result

        if status == "MISSING_DATA":
            print(
                f"▶ {case_name:<30} actual={true_amt!r:>12}  MISSING DATA (extracted root does not exist)"
            )
        elif status == "NO_STAGE1_FILES":
            print(f"▶ {case_name:<30} actual={true_amt!r:>12}  NO STAGE1 FILES FOUND")
        else:
            print(
                f"▶ {case_name:<30}"
                f" {true_amt:>12.0f}"
                f" {predicted_amt if predicted_amt is not None else 'None':>12}"
                f" {filtered_count:>3}/{raw_count:<3}"
                f" {in_raw!s:<6} {in_filtered!s:<7}"
            )

        preds.append(predicted_amt)
        trues.append(float(true_amt))

    # Binary labels: award (>0) vs zero
    y_true = np.array([1 if t > 0 else 0 for t in trues])
    y_pred = np.array([1 if p > 0 else 0 for p in preds])

    # Calculate precision, recall, and F1
    if y_true.sum() == 0 and y_pred.sum() == 0:
        precision = recall = f1 = 1.0
    else:
        precision = precision_score(y_true, y_pred, zero_division="warn")
        recall = recall_score(y_true, y_pred, zero_division="warn")
        f1 = f1_score(y_true, y_pred, zero_division="warn")

    mse = mean_squared_error(trues, preds)

    # Objective: minimize MSE (normalized by max_true for scale)
    obj = mse / max_true

    print(
        f"Params(min_amount={min_amount}, context_chars={context_chars}, "
        f"min_features={min_features}, case_pos={case_pos:.2f}, docket_pos={docket_pos:.2f}) "
        f"→ Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, MSE={mse:.3f}, Obj={obj:.3f}"
    )
    return obj


def detailed_analysis(
    df: pd.DataFrame,
    tokenized_root: Path,
    extracted_root: Path,
    best_params: list[float | int],
) -> None:
    """
    Run detailed per-case analysis with best parameters.
    """
    min_amount, context_chars, min_features, case_pos, docket_pos = best_params
    context_chars = int(context_chars)
    min_features = int(min_features)

    print("\n" + "=" * 80)
    print("DETAILED PER-CASE ANALYSIS")
    print("=" * 80)
    print(
        f"{'Case ID':<20} {'True':<12} {'Pred':<12} {'Candidates':<12} {'Avg Top5 Votes':<15}"
    )
    print("-" * 80)

    all_coverages = []
    all_votes = []
    all_trues = []
    all_preds = []

    for case_id, true_amt in zip(df["case_id"].astype(str), df["final_amount"]):
        if pd.isna(true_amt):
            continue

        # Extract just the case name from the full path
        if "courtlistener/" in case_id:
            case_name = case_id.split("courtlistener/")[-1].rstrip("/")
        else:
            case_name = case_id

        result = impute_pipeline(
            case_name,
            tokenized_root,
            extracted_root,
            min_amount,
            context_chars,
            min_features,
            case_pos,
            docket_pos,
            return_stats=True,
        )
        if isinstance(result, tuple):
            pred_amt, candidate_count, avg_top5_votes = result
        else:
            # Fallback if stats not returned properly
            pred_amt = result
            candidate_count = 0
            avg_top5_votes = 0.0

        print(
            f"{case_id:<20} {true_amt:<12.0f} {pred_amt:<12.0f} {candidate_count:<12} {avg_top5_votes:<15.2f}"
        )

        all_coverages.append(candidate_count)
        all_votes.append(avg_top5_votes)
        all_trues.append(float(true_amt))
        all_preds.append(pred_amt)

    # Summary statistics
    y_true = np.array([1 if t > 0 else 0 for t in all_trues])
    y_pred = np.array([1 if p > 0 else 0 for p in all_preds])

    if y_true.sum() == 0 and y_pred.sum() == 0:
        overall_precision = overall_recall = overall_f1 = 1.0
    else:
        overall_precision = precision_score(y_true, y_pred, zero_division="warn")
        overall_recall = recall_score(y_true, y_pred, zero_division="warn")
        overall_f1 = f1_score(y_true, y_pred, zero_division="warn")

    overall_mse = mean_squared_error(all_trues, all_preds)
    avg_coverage = np.mean(all_coverages) if all_coverages else 0
    avg_votes = np.mean(all_votes) if all_votes else 0

    print("-" * 80)
    print("SUMMARY STATISTICS:")
    print(f"  Average candidate coverage: {avg_coverage:.2f}")
    print(f"  Average top-5 vote counts:  {avg_votes:.2f}")
    print(f"  Overall Precision:          {overall_precision:.3f}")
    print(f"  Overall Recall:             {overall_recall:.3f}")
    print(f"  Overall F1 score:           {overall_f1:.3f}")
    print(f"  Overall MSE:                {overall_mse:.3f}")
    print("=" * 80)


def run_optimization(
    df: pd.DataFrame,
    tokenized_root: Path,
    extracted_root: Path,
    n_calls: int = 50,
    ann_map: dict[str, float] | None = None,
) -> None:
    """
    Set up search space and run Gaussian-Process Bayesian optimization.
    """
    space = [
        Real(500, 50000, name="min_amount"),
        Integer(400, 700, name="context_chars"),
        Integer(0, 20, name="min_features"),
        Real(0.5, 0.9, name="case_position_threshold"),
        Real(0.7, 0.9, name="docket_position_threshold"),
    ]

    result = gp_minimize(
        func=lambda params: evaluate(
            df, tokenized_root, extracted_root, params, ann_map=ann_map
        ),
        dimensions=space,
        n_calls=n_calls,
        random_state=0,
        verbose=True,
    )

    print("\nOptimization completed.")
    if result and hasattr(result, "x") and hasattr(result, "fun"):
        print(f"Best parameters: {result.x}")
        print(f"Best objective value: {result.fun:.3f}")

        # Run detailed analysis with best parameters
        detailed_analysis(df, tokenized_root, extracted_root, result.x)
    else:
        print("Optimization failed to converge.")


def run_weight_optimization(df, tokenized_root, extracted_root, n_calls=30):
    # fixed pipeline params:
    fixed = {
        "min_amount": 58200.0,
        "context_chars": 561,
        "min_features": 1,
        "case_pos": 0.55,
        "docket_pos": 0.74,
    }

    # the 15 new voting-weight dims
    weight_names = [
        "legal_proceedings_weight",
        "monetary_phrases_weight",
        "dependency_parsing_weight",
        "document_structure_weight",
        "table_detection_weight",
        "header_detection_weight",
        "section_boundaries_weight",
        "numeric_gazetteer_weight",
        "mixed_numbers_weight",
        "sentence_boundary_weight",
        "paragraph_boundary_weight",
        "high_confidence_patterns_weight",
        "amount_adjacent_keywords_weight",
        "confidence_boost_weight",
        "calculation_boost_multiplier",
    ]
    space = [Real(0.0, 2.0, name=n) for n in weight_names]

    def objective(weight_vals):
        # merge defaults + overrides
        wdict = DEFAULT_VOTING_WEIGHTS.to_dict()
        wdict.update(dict(zip(weight_names, weight_vals)))
        voting_weights = VotingWeights.from_dict(wdict)

        # call your existing evaluate(), but pass in our fixed pipeline params
        return evaluate(
            df,
            tokenized_root,
            extracted_root,
            [
                fixed["min_amount"],
                fixed["context_chars"],
                fixed["min_features"],
                fixed["case_pos"],
                fixed["docket_pos"],
            ],
            ann_map=None,
            voting_weights=voting_weights,
        )

    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=0,
        verbose=True,
    )
    if res and hasattr(res, "x") and hasattr(res, "fun"):
        print("Best weights:")
        for name, val in zip(weight_names, res.x):
            print(f"  {name} = {val:.3f}")
        print(f"Obj = {res.fun:.5f}")
    else:
        print("Weight optimization failed to converge.")


def run_dismissal_optimization(df, tokenized_root, extracted_root, n_calls=30):
    # fixed pipeline params and weights
    fixed_params = {
        "min_amount": 29309.97970771781,
        "context_chars": 561,
        "min_features": 15,
        "case_pos": 0.5423630428751168,
        "docket_pos": 0.7947200838693315,
    }

    space = [
        Real(
            100.0, 200.0, name="dismissal_ratio_threshold"
        ),  # 1.95 - 2.5 is meaningful range maybe 1.7 to 3.0?
        Real(
            100,
            200,
            name="strict_dismissal_threshold",  # 10-40% weighted confidence (allows for mixed signals)
        ),  # 0.1 - 0.5 is meaningful range, not working
        Real(
            0.0, 0.000000000000000001, name="dismissal_document_type_weight"
        ),  # Document type weighting
        Real(
            0.0, 0.000000000000000001, name="strict_dismissal_document_type_weight"
        ),  # Document type weighting
        Real(
            600000000000000.0, 60000000000000000000.0, name="bankruptcy_ratio_threshold"
        ),  # Bankruptcy court filtering
        Real(
            6000000000000000000.0,
            60000000000000000000000.0,
            name="patent_ratio_threshold",
        ),  # 50-150 patent mentions per doc (very high bar)
    ]

    def objective(dismissal_params):
        # Create a dictionary of the dismissal parameters
        dismissal_dict = dict(zip([d.name for d in space], dismissal_params))

        # This is a placeholder for where you would pass these to your evaluate function
        # For now, we'll just print them to show they're being used.
        print(f"Testing dismissal params: {dismissal_dict}")

        return evaluate(
            df,
            tokenized_root,
            extracted_root,
            list(fixed_params.values()),
            ann_map=None,
            voting_weights=DEFAULT_VOTING_WEIGHTS,  # Using the fully optimized weights
            dismissal_params=dismissal_dict,
        )

    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=0,
        verbose=True,
    )
    if res and hasattr(res, "x") and hasattr(res, "fun"):
        print("Best dismissal params:")
        for name, val in zip([d.name for d in space], res.x):
            print(f"  {name} = {val:.3f}")
        print(f"Obj = {res.fun:.5f}")
    else:
        print("Dismissal optimization failed to converge.")


def compare_mode(
    ann_map: dict[str, float],
    tokenized_root: Path,
    extracted_root: Path,
    min_amount: float = 1000.0,
    context_chars: int = 200,
    min_features: int = 2,
    case_position_threshold: float = 0.5,
    docket_position_threshold: float = 0.5,
    disable_spacy: bool = False,
    disable_spelled: bool = False,
    disable_usd: bool = False,
    disable_calcs: bool = False,
    disable_regex: bool = False,
) -> None:
    """
    Simple comparison mode: show actual vs predicted amounts with candidate counts.
    """
    selector = AmountSelector()

    print("\nCASE COMPARISON: ACTUAL vs PREDICTED\n" + "=" * 80)

    # Debug: Print loaded annotations map
    print(f"[DEBUG] Loaded {len(ann_map)} annotations: {list(ann_map.keys())[:5]}...")

    for case_name in sorted(ann_map.keys()):
        case_root = tokenized_root / case_name
        extracted_case_root = extracted_root / case_name

        # Debug: Print paths being checked
        print(
            f"[DEBUG] Checking case: {case_name}, Extracted Path: {extracted_case_root}"
        )

        if not extracted_case_root.exists():
            print(
                f"▶ {case_name:<30} actual={ann_map[case_name]!r:>12}  MISSING DATA (extracted root does not exist)"
            )
            continue

        # Check for stage1 files in the extracted_case_root
        stage1_files_found = list(extracted_case_root.rglob("*_stage1.jsonl"))
        if not stage1_files_found:
            print(
                f"▶ {case_name:<30} actual={ann_map[case_name]!r:>12}  NO STAGE1 FILES FOUND"
            )
            continue

        # Run unfiltered scan to get total candidate count
        raw_cands = scan_stage1(
            extracted_case_root,
            min_amount,
            context_chars,
            min_features=0,
            case_position_threshold=case_position_threshold,
            docket_position_threshold=docket_position_threshold,
            voting_weights=DEFAULT_VOTING_WEIGHTS,
            disable_spacy=disable_spacy,
            disable_spelled=disable_spelled,
            disable_usd=disable_usd,
            disable_calcs=disable_calcs,
            disable_regex=disable_regex,
        )

        # Run filtered scan with actual min_features
        filtered_cands = scan_stage1(
            extracted_case_root,
            min_amount,
            context_chars,
            min_features=min_features,
            case_position_threshold=case_position_threshold,
            docket_position_threshold=docket_position_threshold,
            voting_weights=DEFAULT_VOTING_WEIGHTS,
            disable_spacy=disable_spacy,
            disable_spelled=disable_spelled,
            disable_usd=disable_usd,
            disable_calcs=disable_calcs,
            disable_regex=disable_regex,
        )

        # Pick the predicted amount
        predicted = selector.choose(filtered_cands)
        actual = ann_map[case_name]

        # Check if actual amount is in candidate lists
        in_raw = any(
            math.isclose(c.value, actual, rel_tol=1e-4, abs_tol=0.01) for c in raw_cands
        )
        in_filtered = any(
            math.isclose(c.value, actual, rel_tol=1e-4, abs_tol=0.01)
            for c in filtered_cands
        )

        print(
            f"▶ {case_name:<30}"
            f" actual={actual!r:>12}"
            f"  predicted={predicted!r:>12}"
            f"  candidates={len(filtered_cands):>3}/{len(raw_cands):<3}"
            f"  in_raw={in_raw!s:<5} in_filtered={in_filtered!s:<5}"
        )


def main():
    p = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning for case outcome imputer"
    )
    p.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to hand-annotated CSV with 'case_id' and 'final_amount'",
    )
    p.add_argument(
        "--tokenized-root",
        type=Path,
        required=True,
        help="Root directory of tokenized (stage4) case folders",
    )
    p.add_argument(
        "--extracted-root",
        type=Path,
        required=True,
        help="Root directory where stage1 JSONL files live",
    )
    p.add_argument(
        "--calls",
        type=int,
        default=50,
        help="Number of optimization calls (default: 50)",
    )
    p.add_argument(
        "--mode",
        choices=["optimize", "compare"],
        default="optimize",
        help="Mode: 'optimize' for Bayesian optimization, 'compare' for simple comparison",
    )
    p.add_argument(
        "--optimize-weights",
        action="store_true",
        help="Hold (min_amount=58200, context_chars=561, min_features=1, case_pos=0.55, docket_pos=0.74) and Bayesian-optimize the new voting weights",
    )
    p.add_argument(
        "--optimize-dismissal",
        action="store_true",
        help="Hold all other parameters and weights to optimize dismissal logic",
    )
    p.add_argument(
        "--min-amount",
        type=float,
        default=1000.0,
        help="Minimum amount threshold (default: 1000.0)",
    )
    p.add_argument(
        "--context-chars",
        type=int,
        default=200,
        help="Context characters around amounts (default: 200)",
    )
    p.add_argument(
        "--min-features",
        type=int,
        default=2,
        help="Minimum feature votes required (default: 2)",
    )
    p.add_argument(
        "--case-position-threshold",
        type=float,
        default=0.5,
        help="Case position threshold (default: 0.5)",
    )
    p.add_argument(
        "--docket-position-threshold",
        type=float,
        default=0.5,
        help="Docket position threshold (default: 0.5)",
    )
    # Disable flags for pipeline stages
    p.add_argument(
        "--disable-spacy",
        action="store_true",
        help="Turn off all spaCy-based extraction",
    )
    p.add_argument(
        "--disable-spelled",
        action="store_true",
        help="Turn off spelled-out-number extraction",
    )
    p.add_argument(
        "--disable-usd", action="store_true", help="Turn off USD-prefix extraction"
    )
    p.add_argument(
        "--disable-calcs",
        action="store_true",
        help="Turn off any calculation-based extraction (fractions, sums, etc.)",
    )
    p.add_argument(
        "--disable-regex",
        action="store_true",
        help="Turn off the standard AMOUNT_REGEX pass",
    )
    args = p.parse_args()

    if args.optimize_weights:
        df = load_annotations(args.annotations)
        run_weight_optimization(
            df, args.tokenized_root, args.extracted_root, n_calls=args.calls
        )
        return
    elif args.optimize_dismissal:
        df = load_annotations(args.annotations)
        run_dismissal_optimization(
            df, args.tokenized_root, args.extracted_root, n_calls=args.calls
        )
        return

    # Load annotations into a dict
    ann_map: dict[str, float] = {}
    if args.annotations:
        with open(args.annotations, newline="", encoding="utf8") as f:
            # Read the header and strip BOM from keys
            header = [
                h.strip().replace("\ufeff", "") for h in f.readline().split(",")
            ]  # Read first line for header
            f.seek(0)  # Reset file pointer to beginning
            reader = csv.DictReader(f, fieldnames=header)
            # Skip header row if it's the first line
            next(reader)  # Skip the header line as it's already processed

            for row in reader:
                # Debug: Print raw row to inspect contents
                # print(f"[DEBUG] Processing row: {row}")
                try:
                    # Use .get() with a default of '' to prevent KeyError if a column is truly missing
                    case_id_raw = row.get("case_id", "")
                    final_amount_raw = row.get("final_amount", "")

                    if not case_id_raw or not final_amount_raw:
                        # print(f"[DEBUG] Skipping row due to missing case_id or final_amount: {row}")
                        continue

                    case = case_id_raw.rstrip("/").split("/")[-1]
                    final_amount = final_amount_raw.replace(",", "")

                    if final_amount.lower() in ("null", "none", ""):
                        # print(f"[DEBUG] Skipping row due to null/none final_amount: {row}")
                        continue
                    ann_map[case] = float(final_amount)
                except (ValueError, KeyError, AttributeError) as e:
                    print(f"[DEBUG] Skipping row due to error: {e} in row: {row}")
                    continue

    if args.mode == "compare":
        compare_mode(
            ann_map,
            args.tokenized_root,
            args.extracted_root,
            args.min_amount,
            args.context_chars,
            args.min_features,
            args.case_position_threshold,
            args.docket_position_threshold,
            args.disable_spacy,
            args.disable_spelled,
            args.disable_usd,
            args.disable_calcs,
            args.disable_regex,
        )
    else:
        df = load_annotations(args.annotations)
        run_optimization(
            df, args.tokenized_root, args.extracted_root, args.calls, ann_map=ann_map
        )


if __name__ == "__main__":
    main()
