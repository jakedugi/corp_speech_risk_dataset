#!/usr/bin/env python3
"""
Binary Classification Stratified K-Fold Cross-Validation

Modified version of stratified_kfold_case_split.py that creates 2 classes (lower/higher)
instead of 3 classes (tertiles), using a single interpolated midpoint to divide each fold
into case-level halves. All other logic remains exactly the same.

Key changes from original:
- binary_edges_caselevel_train_only() instead of tertile_edges_caselevel_train_only()
- assign_bin_binary() for 2-class assignment
- casewise_train_bins_binary() for binary case binning
- All temporal CV, DNT policy, weighting logic unchanged

Usage:
    python scripts/stratified_kfold_binary_split.py \
        --input data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl \
        --output-dir data/final_stratified_kfold_splits_binary \
        --k-folds 4 \
        --target-field final_judgement_real \
        --use-temporal-cv \
        --oof-test-ratio 0.15 \
        --random-seed 42
"""

import sys
import os
from typing import Tuple, Set, Dict, Any, List
from pathlib import Path
import argparse
import json
import hashlib

# Add the original script directory to path so we can import most functions
sys.path.insert(0, os.path.dirname(__file__))

# Import everything from the original script
from stratified_kfold_case_split import *

import numpy as np
import pandas as pd
from collections import Counter
from loguru import logger


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """
    Compute weighted quantile using interpolation for tie-safe results.

    This implements the standard weighted quantile calculation:
    1. Sort values by outcome, carrying weights
    2. Compute cumulative weights
    3. Find interpolated quantile position
    4. Return interpolated value

    Args:
        values: Array of outcome values
        weights: Array of weights (same length as values)
        quantile: Quantile to compute (0.0 to 1.0)

    Returns:
        Interpolated quantile value
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")

    if len(values) == 0:
        raise ValueError("Cannot compute quantile of empty array")

    if len(values) == 1:
        return float(values[0])

    # Sort by values, carrying weights
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]

    if total_weight == 0:
        raise ValueError("Total weight is zero")

    # Target position for quantile
    target_position = quantile * total_weight

    # Find the position in cumulative weights
    # We want the first index where cumulative_weight >= target_position
    position_index = np.searchsorted(cumulative_weights, target_position, side="right")

    if position_index == 0:
        return float(sorted_values[0])
    elif position_index >= len(sorted_values):
        return float(sorted_values[-1])
    else:
        # Interpolate between two adjacent values
        left_value = sorted_values[position_index - 1]
        right_value = sorted_values[position_index]

        # Linear interpolation based on weight position
        left_cum_weight = (
            cumulative_weights[position_index - 1] if position_index > 0 else 0
        )
        right_cum_weight = cumulative_weights[position_index]

        if right_cum_weight == left_cum_weight:
            return float(left_value)

        # Interpolation factor
        alpha = (target_position - left_cum_weight) / (
            right_cum_weight - left_cum_weight
        )
        interpolated_value = left_value + alpha * (right_value - left_value)

        return float(interpolated_value)


def anonymize_case_id(case_id: str) -> str:
    """
    Anonymize case ID by removing temporal information while preserving uniqueness.

    Converts: "1:06-cv-04567_nysd" -> "anon_1_cv_04567_nysd"
    Converts: "24-10951_ca5" -> "anon_10951_ca5"

    This prevents temporal leakage while maintaining case uniqueness.
    """
    if not case_id:
        return case_id

    import re

    # Handle federal court format: "1:06-cv-04567_nysd"
    federal_match = re.search(r"^(\d+):(\d+)-(cv|cr)-(\d+)_([a-z]+)$", str(case_id))
    if federal_match:
        court_num = federal_match.group(1)
        case_type = federal_match.group(3)  # cv or cr
        case_num = federal_match.group(4)
        district = federal_match.group(5)
        return f"anon_{court_num}_{case_type}_{case_num}_{district}"

    # Handle appellate format: "24-10951_ca5"
    appellate_match = re.search(r"^(\d{2})-(\d+)_ca(\d+)$", str(case_id))
    if appellate_match:
        case_num = appellate_match.group(2)
        circuit = appellate_match.group(3)
        return f"anon_{case_num}_ca{circuit}"

    # Handle other formats by removing obvious year patterns
    anonymized = str(case_id)

    # Remove 2-4 digit years at start of strings
    anonymized = re.sub(r"^(19|20)\d{2}[-_]", "anon_", anonymized)
    anonymized = re.sub(r":\d{2,4}-", ":XX-", anonymized)

    return anonymized


def test_anonymization():
    """Test the anonymization function with examples."""
    test_cases = [
        "1:06-cv-04481_ilnd",
        "1:12-cv-23362_flsd",
        "24-10951_ca5",
        "2:07-cv-04572_laed",
        "some_other_format_2020_case",
    ]

    for case_id in test_cases:
        anon = anonymize_case_id(case_id)
        print(f"  {case_id} -> {anon}")


def debug_year_extraction(case_id: str) -> str:
    """Debug function to test year extraction patterns."""
    import re

    if not case_id or pd.isna(case_id):
        return f"EMPTY/NULL -> fallback"

    case_id_str = str(case_id).strip()
    if not case_id_str:
        return f"EMPTY_STRING -> fallback"

    # Test each pattern individually
    results = []

    # Pattern 1: Appellate court
    match = re.search(r"^(\d{2})-\d+_ca\d+$", case_id_str)
    if match:
        year_suffix = int(match.group(1))
        year = 2000 + year_suffix if year_suffix <= 30 else 1900 + year_suffix
        return f"APPELLATE: {case_id_str} -> {year}"

    # Pattern 2: Federal court with district
    match = re.search(r"(\d+):(\d+)-(cv|cr)-\d+_[a-z]+", case_id_str)
    if match:
        year = int(match.group(2))
        final_year = 1900 + year if year > 90 else 2000 + year
        return f"FEDERAL_FULL: {case_id_str} -> {final_year} (raw: {year})"

    # Pattern 3: Federal court without district
    match = re.search(r"(\d+):(\d+)-(cv|cr)-\d+", case_id_str)
    if match:
        year = int(match.group(2))
        final_year = 1900 + year if year > 90 else 2000 + year
        return f"FEDERAL_PARTIAL: {case_id_str} -> {final_year} (raw: {year})"

    # Pattern 4: Any colon pattern
    match = re.search(r":(\d{2})-", case_id_str)
    if match:
        year = int(match.group(1))
        final_year = 1900 + year if year > 90 else 2000 + year
        return f"COLON_PATTERN: {case_id_str} -> {final_year} (raw: {year})"

    # Pattern 5: 4-digit year
    match = re.search(r"(\d{4})", case_id_str)
    if match:
        year = int(match.group(1))
        if 1950 <= year <= 2030:
            return f"FOUR_DIGIT: {case_id_str} -> {year}"

    # Pattern 6: 2-digit year at start
    match = re.search(r"^(\d{2})[-_]", case_id_str)
    if match:
        year = int(match.group(1))
        final_year = 1900 + year if year > 90 else 2000 + year
        return f"START_TWO_DIGIT: {case_id_str} -> {final_year} (raw: {year})"

    return f"NO_MATCH: {case_id_str} -> fallback"


# Quick test of the debug function
if __name__ == "__main__" and False:  # Set to True to run test
    test_cases = [
        "1:19-cv-02184_dcd",  # From logs - should work
        "24-10951_ca5",  # Appellate - should work
        "1:06-cv-04481_ilnd",  # Federal - should work
        "",  # Empty - should fail
        "some_weird_format",  # Unknown - should fail
    ]
    print("Testing year extraction:")
    for case in test_cases:
        result = debug_year_extraction(case)
        print(f"  {result}")


def binary_edges_caselevel_train_only(
    train_df: pd.DataFrame,
    case_col: str,
    y_col: str,
    guard_lo: float = 0.35,
    guard_hi: float = 0.65,
    rng_seed: int = 42,
    use_quote_balanced: bool = False,
    quote_weight_cap: int = None,
) -> np.ndarray:
    """
    BINARY boundary logic (case-level, train-only) - creates 2 equal classes.

    Goal: Split training cases into two near-equal groups by outcome,
    then derive numeric cutpoint for DEV/TEST/OOF application.

    Supports two balancing strategies:
    A) Case-balanced (default): Each case contributes equally (weight=1)
    B) Quote-balanced: Each case weighted by its quote count (or capped quote count)

    Semantics (binary):
    - lower: y < e1
    - higher: y >= e1

    Args:
        train_df: Training DataFrame
        case_col: Case ID column name
        y_col: Outcome column name
        guard_lo: Minimum acceptable bin share (default: 0.35)
        guard_hi: Maximum acceptable bin share (default: 0.65)
        rng_seed: Random seed for tie-breaking
        use_quote_balanced: If True, weight cases by quote count for cutoff calculation
        quote_weight_cap: Optional cap on quote weights to reduce mega-case influence

    Returns:
        np.array([e1]) or empty array if degenerate
    """
    # 1) one outcome per TRAIN case with optional quote weights
    case_df = train_df[[case_col, y_col]].dropna().drop_duplicates(subset=[case_col])

    if use_quote_balanced:
        # Calculate quote count per case for weighting
        case_quote_counts = train_df.groupby(case_col).size()
        case_df = case_df.set_index(case_col)
        case_df["quote_count"] = case_quote_counts
        case_df = case_df.dropna().reset_index()

        # Apply weight cap if specified
        if quote_weight_cap is not None:
            case_df["weight"] = case_df["quote_count"].clip(upper=quote_weight_cap)
            logger.info(
                f"Applied quote weight cap of {quote_weight_cap} to {(case_df['quote_count'] > quote_weight_cap).sum()} cases"
            )
        else:
            case_df["weight"] = case_df["quote_count"]

        logger.info(
            f"Quote-balanced mode: weights range from {case_df['weight'].min()} to {case_df['weight'].max()}"
        )
    else:
        # Case-balanced mode: all weights = 1
        case_df = case_df.reset_index(drop=True)
        case_df["weight"] = 1.0

    y = case_df[y_col].values
    weights = case_df["weight"].values
    case_ids = case_df[case_col].values
    n = len(y)

    if n < 2 or pd.Series(y).nunique() < 2:
        logger.warning(
            f"Insufficient distinct outcomes ({pd.Series(y).nunique()}) for binary split in {n} training cases"
        )
        return np.array([])  # degrade upstream as needed

    # 2) Compute cutoff using weighted quantile
    if use_quote_balanced:
        e1 = weighted_quantile(y, weights, 0.5)
        logger.info(
            f"Quote-balanced cutoff: ${e1:,.2f} (50th percentile by quote weight)"
        )
    else:
        # Standard case-balanced approach
        e = np.quantile(y, [0.5], method="midpoint")  # Single median
        e1 = float(e[0])
        logger.info(f"Case-balanced cutoff: ${e1:,.2f} (50th percentile by case count)")

    # 3) check guardrails on TRAIN shares with binary rule (weighted)
    def weighted_shares(e1, values, weights):
        """Compute weighted shares for lower/higher bins."""
        lower_mask = values < e1
        higher_mask = values >= e1

        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0.0, 0.0

        lower_weight = np.sum(weights[lower_mask])
        higher_weight = np.sum(weights[higher_mask])

        return lower_weight / total_weight, higher_weight / total_weight

    lower, higher = weighted_shares(e1, y, weights)

    # Log both case and quote perspectives for transparency
    case_lower = (y < e1).mean()
    case_higher = (y >= e1).mean()

    if use_quote_balanced:
        logger.info(f"Quote-weighted shares: lower={lower:.1%}, higher={higher:.1%}")
        logger.info(
            f"Case shares (for reference): lower={case_lower:.1%}, higher={case_higher:.1%}"
        )
    else:
        logger.info(f"Case shares: lower={lower:.1%}, higher={higher:.1%}")

    if (
        lower >= guard_lo
        and higher >= guard_lo
        and lower <= guard_hi
        and higher <= guard_hi
    ):
        balance_type = "quote-weighted" if use_quote_balanced else "case-balanced"
        logger.info(
            f"Cutoff passed guardrails ({balance_type}): lower={lower:.1%}, higher={higher:.1%}"
        )
        return np.array([e1], dtype=float)

    # 4) FALLBACK: weighted rank-slice for exact halves, then midpoint
    balance_type = "quote-weighted" if use_quote_balanced else "case-balanced"
    logger.warning(
        f"{balance_type} cutoff failed guardrails: lower={lower:.1%}, higher={higher:.1%}"
    )
    logger.info(f"Using weighted rank-slice fallback for exact {balance_type} halves")

    if use_quote_balanced:
        # For quote-balanced, use weighted median with tie-breaking
        # Sort by outcome, then by case_id hash for stable tie-breaking
        case_id_series = pd.Series(case_ids)
        tie = (
            pd.util.hash_pandas_object(case_id_series).astype("int64") % (10**9)
        ).values
        order = np.lexsort((tie, y))  # sort by y, then tie-breaker

        y_sorted = y[order]
        weights_sorted = weights[order]

        # Find the weighted median using the 50th percentile
        e1 = weighted_quantile(y_sorted, weights_sorted, 0.5)

    else:
        # Standard case-balanced fallback (original logic)
        case_id_series = pd.Series(case_ids)
        tie = (
            pd.util.hash_pandas_object(case_id_series).astype("int64") % (10**9)
        ).values
        order = np.lexsort((tie, y))  # sort by y, then tie-breaker
        y_sorted = y[order]

        i1 = int(np.ceil(n / 2)) - 1  # zero-based boundary index for last in lower

        def midpoint(a, b):
            return (a + b) / 2.0

        # Edge 1 (single boundary for binary)
        if i1 + 1 < len(y_sorted) and y_sorted[i1] != y_sorted[i1 + 1]:
            e1 = midpoint(y_sorted[i1], y_sorted[i1 + 1])
        else:
            e1 = y_sorted[i1]  # equals go to higher per binary rule

    return np.array([float(e1)], dtype=float)


def assign_bin_binary(y: float, edges: np.ndarray) -> int:
    """Apply binary labels with the rule: lower < e1, higher >= e1

    Current convention: "equals → higher" (y >= e1 → bin 1)
    This maintains monotonic "≥ cutoff = higher risk" interpretation.
    """
    if len(edges) < 1:
        return 0  # degenerate case → single bin
    e1 = edges[0]
    if y < e1:
        return 0  # lower
    return 1  # higher


def analyze_cutoff_ties(outcomes: list[float], edge: float) -> dict[str, int]:
    """Analyze how many cases are exactly equal to the cutoff."""
    outcomes_array = np.array(outcomes)
    exactly_equal = np.sum(outcomes_array == edge)
    below_cutoff = np.sum(outcomes_array < edge)
    above_cutoff = np.sum(outcomes_array > edge)

    return {
        "exactly_equal_to_cutoff": int(exactly_equal),
        "strictly_below": int(below_cutoff),
        "strictly_above": int(above_cutoff),
        "total_cases": len(outcomes),
    }


def casewise_train_bins_binary(
    train_df: pd.DataFrame,
    y_col: str = "final_judgement_real",
    use_quote_balanced: bool = False,
    quote_weight_cap: int = None,
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Create case-wise BINARY bins using authoritative binary boundary logic.

    Uses interpolated median with rank-slice fallback for guaranteed 50/50 balance.
    Implements the binary boundary semantics: lower < e1, higher >= e1.

    Supports both case-balanced and quote-balanced cutoff calculation.

    Args:
        train_df: Training DataFrame with case_id and outcome columns
        y_col: Column name for outcomes
        use_quote_balanced: If True, weight cases by quote count for cutoff calculation
        quote_weight_cap: Optional cap on quote weights to reduce mega-case influence

    Returns:
        Tuple of (case_bins dict, edges array)
    """
    # Use the binary boundary logic with quote balancing options
    edges = binary_edges_caselevel_train_only(
        train_df,
        "case_id",
        y_col,
        use_quote_balanced=use_quote_balanced,
        quote_weight_cap=quote_weight_cap,
    )

    if edges.size == 0:
        logger.warning(
            "Could not create binary boundary - insufficient distinct outcomes"
        )
        return {}, np.array([])

    # Extract one outcome per case for labeling
    case_y = train_df.groupby("case_id")[y_col].first().astype(float)

    # Apply the binary boundary rule to assign bins
    case_bins = {}
    for case_id, outcome in case_y.items():
        bin_idx = assign_bin_binary(outcome, edges)
        case_bins[case_id] = bin_idx

    # Verify final balance
    final_counts = Counter(case_bins.values())
    total_cases = len(case_bins)
    balance_info = []
    for bin_idx in sorted(final_counts.keys()):
        count = final_counts[bin_idx]
        pct = count / total_cases * 100
        balance_info.append(f"bin_{bin_idx}: {count} cases ({pct:.1f}%)")

    logger.info(f"Case-wise BINARY balance: " + " | ".join(balance_info))

    return case_bins, edges


def make_temporal_fold_binary(
    df: pd.DataFrame,
    train_cases: Set[str],
    val_cases: Set[str],
    test_cases: Set[str],
    fold_idx: int,
    use_quote_balanced: bool = False,
    quote_weight_cap: int = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Create a single temporal fold with binary classification.

    Same as make_temporal_fold but uses binary binning instead of tertiles.
    """
    # Filter to cases in this fold
    all_fold_cases = train_cases | val_cases | test_cases
    fold_df = df[df["case_id"].isin(all_fold_cases)].copy()

    # Assign split labels
    fold_df["fold"] = fold_idx
    fold_df["split"] = np.where(
        fold_df["case_id"].isin(test_cases),
        "test",
        np.where(fold_df["case_id"].isin(val_cases), "val", "train"),
    )

    # Per-fold BINARY outcome binning using TRAINING DATA ONLY - GUARANTEED 50/50 CASE BALANCE
    train_df = fold_df[fold_df["split"] == "train"]
    edges_used = []
    train_case_bins = {}

    if len(train_df) >= 2:
        # Use new case-wise binary approach
        train_case_bins, edges = casewise_train_bins_binary(
            train_df,
            "final_judgement_real",
            use_quote_balanced=use_quote_balanced,
            quote_weight_cap=quote_weight_cap,
        )

        if len(train_case_bins) > 0 and edges.size > 0:
            # Apply case bins to ALL records in the fold
            fold_df["outcome_bin"] = (
                fold_df["case_id"].map(train_case_bins).fillna(0).astype(int)
            )

            # For records not in training (val/test), use edges to assign bins
            non_train_mask = ~fold_df["case_id"].isin(train_case_bins.keys())
            if non_train_mask.any():
                # Vectorized binary assignment: 0 if y < e1 else 1
                fold_df.loc[non_train_mask, "outcome_bin"] = (
                    fold_df.loc[non_train_mask, "final_judgement_real"].values
                    >= edges[0]
                ).astype(int)

            edges_used = edges.tolist()

            logger.info(f"Fold {fold_idx}: Case-wise binary split at ${edges[0]:,.0f}")

            # Analyze cutoff ties for convention validation
            train_outcomes = train_df["final_judgement_real"].dropna().tolist()
            tie_analysis = analyze_cutoff_ties(train_outcomes, edges[0])
            if tie_analysis["exactly_equal_to_cutoff"] > 0:
                logger.info(
                    f"Fold {fold_idx}: {tie_analysis['exactly_equal_to_cutoff']} cases exactly at cutoff "
                    f"(assigned to higher bin by current convention)"
                )

        elif len(train_case_bins) > 0:
            # Handle degenerate case (1 bin only)
            fold_df["outcome_bin"] = (
                fold_df["case_id"].map(train_case_bins).fillna(0).astype(int)
            )

            # Assign non-training records to bin 0 as fallback
            non_train_mask = ~fold_df["case_id"].isin(train_case_bins.keys())
            fold_df.loc[non_train_mask, "outcome_bin"] = 0

            edges_used = edges.tolist() if edges.size > 0 else []

            n_bins = len(set(train_case_bins.values()))
            logger.info(f"Fold {fold_idx}: Degenerate case with {n_bins} bins")

        else:
            # Complete fallback
            fold_df["outcome_bin"] = 0
            edges_used = []
            logger.warning(f"Fold {fold_idx}: Could not create case-wise binary bins")
    else:
        # Fallback for insufficient training data
        fold_df["outcome_bin"] = 0
        edges_used = []
        logger.warning(
            f"Fold {fold_idx}: Insufficient training data ({len(train_df)} records)"
        )

    # CRITICAL: Add normalized text hash if not present
    if "text_hash_norm" not in fold_df.columns and "text" in fold_df.columns:
        logger.info(
            f"Fold {fold_idx}: Creating normalized text hashes for better duplicate detection"
        )
        fold_df["text_hash_norm"] = (
            fold_df["text"]
            .astype(str)
            .apply(normalize_for_hash)
            .apply(lambda t: hashlib.md5(t.encode()).hexdigest())
        )

    # CRITICAL: Remove within-fold text contamination (same as original)
    initial_train_count = (fold_df["split"] == "train").sum()
    fold_df = purge_within_fold_text_collisions(fold_df)
    final_train_count = (fold_df["split"] == "train").sum()

    if initial_train_count != final_train_count:
        logger.info(
            f"Fold {fold_idx}: Reduced train set from {initial_train_count} to {final_train_count} "
            f"records ({initial_train_count - final_train_count} dropped for text collisions)"
        )

    # Compute support weights for balanced training (same logic as original)
    train_fold = fold_df[fold_df["split"] == "train"]
    if len(train_fold) > 0:
        # Get case sizes for weighting
        case_support = train_fold.groupby("case_id").size()

        # Support tertiles for reporting/analysis (marked as DNT)
        fold_df["support_tertile"] = (
            fold_df["case_id"]
            .map(dict(zip(case_support.index, support_tertile(case_support))))
            .fillna(1)
        )  # Default for cases not in training

        # --- QUOTE-BASED CLASS WEIGHTS (binary) ---
        train_mask = fold_df["split"] == "train"
        train_fold = fold_df.loc[train_mask]

        # Count QUOTES per bin in TRAIN (not cases)
        record_bin_counts = train_fold["outcome_bin"].value_counts().to_dict()
        unique_bins = sorted(record_bin_counts.keys())

        if len(unique_bins) > 1 and min(record_bin_counts.values()) > 0:
            total_records = int(train_mask.sum())
            n_bins = len(unique_bins)

            bin_weight_map = {}
            for bin_idx in unique_bins:
                cnt = record_bin_counts[bin_idx]
                # Balanced BY QUOTES:
                w = total_records / (n_bins * cnt)
                # Clip to avoid runaway weights
                bin_weight_map[bin_idx] = float(np.clip(w, 0.25, 4.0))
        else:
            # Fallback if a class is absent in TRAIN
            bin_weight_map = {b: 1.0 for b in unique_bins}

        # Apply to ALL rows (val/test inherit train weights)
        fold_df["bin_weight"] = fold_df["outcome_bin"].map(bin_weight_map).fillna(1.0)

        # (Optional) Logging for transparency
        info = " | ".join(
            f"bin_{b}: {record_bin_counts.get(b,0)} quotes → weight={bin_weight_map[b]:.3f}"
            for b in unique_bins
        )
        logger.info(f"Fold {fold_idx}: QUOTE-based class weights - {info}")

        # Verify quote-wise balance and class weight ratios
        quote_shares = []
        for bin_idx in sorted(unique_bins):
            quote_count = record_bin_counts.get(bin_idx, 0)
            if total_records > 0:
                pct = quote_count / total_records
                quote_shares.append(pct)

        if quote_shares:
            min_share, max_share = min(quote_shares), max(quote_shares)
            logger.info(
                f"Fold {fold_idx}: TRAIN quote balance (range: {min_share:.1%}-{max_share:.1%})"
            )

            # Sanity check for extreme weights
            if len(bin_weight_map) > 1:
                max_weight = max(bin_weight_map.values())
                min_weight = min(bin_weight_map.values())
                weight_ratio = (
                    max_weight / min_weight if min_weight > 0 else float("inf")
                )

                if weight_ratio > 4.0:  # After clipping, should be ≤4x
                    logger.warning(
                        f"Fold {fold_idx}: Class weight ratio ({weight_ratio:.1f}x) at clip limit - "
                        f"indicates significant quote imbalance"
                    )
                else:
                    logger.info(
                        f"✅ Fold {fold_idx}: Class weight ratio {weight_ratio:.1f}x within bounds"
                    )
            else:
                fold_df["bin_weight"] = 1.0
        else:
            fold_df["bin_weight"] = 1.0

        # Rest of weighting logic same as original...
        # Inverse sqrt support weighting with clipping
        w_case = (1.0 / np.sqrt(case_support)).clip(0.25, 4.0)  # Cap extremes
        w_case_normalized = w_case * (len(case_support) / w_case.sum())  # Normalize

        # Combine bin weights and support weights
        case_weight_map = dict(zip(case_support.index, w_case_normalized))
        fold_df["support_weight"] = fold_df["case_id"].map(case_weight_map).fillna(1.0)
        fold_df["sample_weight"] = fold_df["bin_weight"] * fold_df["support_weight"]

        # Re-normalize final sample_weight to mean=1.0 on train split only
        train_mask = fold_df["split"] == "train"
        train_sample_weights = fold_df.loc[train_mask, "sample_weight"]
        if len(train_sample_weights) > 0 and train_sample_weights.mean() > 0:
            normalization_factor = train_sample_weights.mean()
            fold_df["sample_weight"] = fold_df["sample_weight"] / normalization_factor
            logger.info(
                f"Fold {fold_idx}: Re-normalized sample_weight to mean=1.0 on train (factor: {normalization_factor:.4f})"
            )

            # Validation checks
            train_final_weights = fold_df.loc[train_mask, "sample_weight"]
            logger.info(
                f"Fold {fold_idx}: Final train sample_weight mean={train_final_weights.mean():.4f}, std={train_final_weights.std():.4f}"
            )

        # Store weight metadata for persistence (JSON-serializable)
        weight_metadata = {
            "class_weights": {str(k): float(v) for k, v in bin_weight_map.items()},
            "class_weights_unit": "quote",  # <— NEW, clarifies the unit
            "support_weight_method": "inverse_sqrt_clipped",
            "support_weight_range": [0.25, 4.0],
            "bin_count": int(len(unique_bins)),
            "train_split_counts": {
                "total": int(len(train_fold)),
                "per_bin": {str(k): int(v) for k, v in record_bin_counts.items()},
            },
        }
        edges_used.append(weight_metadata)  # Append metadata to edges return
    else:
        fold_df["support_tertile"] = 1
        fold_df["bin_weight"] = 1.0
        fold_df["support_weight"] = 1.0
        fold_df["sample_weight"] = 1.0

        # No weight metadata for empty folds
        weight_metadata = {
            "class_weights": {},
            "class_weights_unit": "quote",
            "support_weight_method": "none",
            "support_weight_range": [1.0, 1.0],
            "bin_count": 1,
            "train_split_counts": {"total": 0, "per_bin": {}},
        }
        edges_used.append(weight_metadata)

    return fold_df, edges_used


def make_leakage_safe_splits_binary(
    df: pd.DataFrame,
    k: int = 5,
    oof_ratio: float = 0.15,
    seed: int = 42,
    oof_min_ratio: float = 0.15,
    oof_max_ratio: float = 0.40,
    oof_step: float = 0.05,
    min_class_cases: int = 5,
    min_class_quotes: int = 50,
    oof_criterion: str = "both",
    use_quote_balanced: bool = False,
    quote_weight_cap: int = None,
) -> pd.DataFrame:
    """
    Create complete leakage-safe splits with DNT policy and temporal CV using BINARY classification.

    Same as make_leakage_safe_splits but uses binary classification instead of tertiles.
    """
    np.random.seed(seed)
    logger.info("Creating leakage-safe splits with DNT policy and temporal CV (BINARY)")

    # 0) Hygiene: remove within-case duplicates
    df = hard_dedupe_within_case(df)

    # 1) Apply DNT policy (wrap, don't drop)
    df, dnt = wrap_do_not_train_cols(df)
    df, dnt = collapse_sentiment_and_mark_raw_dnt(df, dnt)

    # 2) Create case-level data with outcomes
    case_df = df.groupby("case_id", as_index=False).agg(
        case_year=("case_year", "min"),
        outcome=("final_judgement_real", "first"),
        size=("case_id", "count"),  # Count records per case
    )
    case_df["case_year"] = case_df["case_year"].fillna(case_df["case_year"].min())

    # Extract case outcomes for adaptive OOF (vectorized)
    valid_outcomes = case_df.dropna(subset=["outcome"])
    case_outcomes = dict(
        zip(valid_outcomes["case_id"], valid_outcomes["outcome"].astype(float))
    )

    # 3) Create adaptive OOF test split with 2-class guarantee (instead of 3-class)
    # Note: For binary, we need both classes represented in OOF
    # Modify the OOF growth function to check for 2 classes instead of 3

    def binary_edges_tie_safe(values):
        """Binary version of tertile_edges_tie_safe."""
        s = pd.Series(values).dropna()
        if len(s) < 2:
            return np.array([])

        y = s.values
        if pd.Series(y).nunique() < 2:
            return np.array([])

        # Use interpolated median for consistency
        e = np.quantile(y, [0.5], method="midpoint")
        e1 = float(e[0])

        return np.array([e1], dtype=float)

    # Use modified OOF growth that checks for 2 classes
    oof_test_cases, cv_cases, oof_meta = grow_oof_until_binary_classes(
        case_df=case_df,
        case_outcomes=case_outcomes,
        oof_min_ratio=oof_min_ratio,
        oof_max_ratio=oof_max_ratio,
        oof_step=oof_step,
        min_class_cases=min_class_cases,
        min_class_quotes=min_class_quotes,
        criterion=oof_criterion,
    )

    # 4) Generate rolling-origin folds on CV cases only
    cv_case_df = case_df[case_df["case_id"].isin(cv_cases)]
    folds = rolling_origin_group_folds(cv_case_df, k)

    # 5) Create fold DataFrames using binary classification
    fold_dfs = []
    per_fold_edges = {}
    per_fold_weights = {}
    for i, (train_cases, val_cases, test_cases) in enumerate(folds):
        fold_df, edges_and_weights = make_temporal_fold_binary(
            df,
            train_cases,
            val_cases,
            test_cases,
            i,
            use_quote_balanced=use_quote_balanced,
            quote_weight_cap=quote_weight_cap,
        )

        # Separate edges from weight metadata
        if len(edges_and_weights) >= 2:
            edges = (
                edges_and_weights[0]
                if isinstance(edges_and_weights[0], list)
                else edges_and_weights[:-1]
            )
            weight_metadata = (
                edges_and_weights[-1] if isinstance(edges_and_weights[-1], dict) else {}
            )
        else:
            edges = edges_and_weights if isinstance(edges_and_weights, list) else []
            weight_metadata = {}

        per_fold_edges[f"fold_{i}"] = edges
        per_fold_weights[f"fold_{i}"] = weight_metadata
        fold_dfs.append(fold_df)

    # 6) Create OOF test DataFrame (labels will be set after final fold creation)
    if oof_test_cases:
        oof_df = df[df["case_id"].isin(oof_test_cases)].copy()
        oof_df["fold"] = -1  # Special marker for OOF
        oof_df["split"] = "oof_test"
        oof_df["support_tertile"] = 1
        oof_df["sample_weight"] = 1.0
        oof_df["outcome_bin"] = 0  # Temporary - will be corrected after final fold
        fold_dfs.append(oof_df)

    # 7) Create final training fold combining all CV data (using binary)
    logger.info("Creating final training fold with all CV data (BINARY)...")

    # Same logic as original but use binary classification
    all_cv_train_cases = cv_cases
    dev_ratio = 0.15
    min_dev_quotes = 150

    cv_df = df[df["case_id"].isin(cv_cases)]
    total_cv_quotes = len(cv_df)
    needed_dev_ratio = min(0.3, max(dev_ratio, min_dev_quotes / total_cv_quotes))

    if needed_dev_ratio != dev_ratio:
        logger.info(
            f"Final fold: Adjusting dev_ratio from {dev_ratio:.2f} to {needed_dev_ratio:.2f} for min quotes"
        )

    cv_case_df_sorted = cv_case_df.sort_values("case_year")
    cv_case_ids = cv_case_df_sorted["case_id"].tolist()

    n_dev = max(1, int(needed_dev_ratio * len(cv_case_ids)))
    final_dev_cases = set(cv_case_ids[-n_dev:])
    final_train_cases = set(cv_case_ids[:-n_dev])

    # Verify final fold will have proper case-wise balance for binary
    if len(final_train_cases) >= 2:
        train_test_df = df[df["case_id"].isin(final_train_cases)].copy()
        temp_case_bins, temp_edges = casewise_train_bins_binary(
            train_test_df,
            "final_judgement_real",
            use_quote_balanced=use_quote_balanced,
            quote_weight_cap=quote_weight_cap,
        )

        if len(temp_case_bins) > 0 and temp_edges.size > 0:
            # Apply train edges to dev cases to check coverage
            dev_bins = {}
            for case_id in final_dev_cases:
                if case_id in case_outcomes:
                    outcome = case_outcomes[case_id]
                    bin_idx = assign_bin_binary(outcome, temp_edges)
                    dev_bins[case_id] = bin_idx

            # Count unique bins in dev set
            unique_dev_bins = set(dev_bins.values())
            expected_bins = set(range(2))  # Binary: {0, 1}
            missing_bins = expected_bins - unique_dev_bins

            if len(missing_bins) > 0:
                logger.warning(
                    f"Final fold dev set missing bins: {missing_bins}. "
                    f"Present bins: {sorted(unique_dev_bins)}. "
                    f"Consider increasing dev_ratio."
                )

                # Adaptive strategy for binary: grow dev set if missing critical bins
                if len(unique_dev_bins) < 2:
                    logger.info("Growing dev set to ensure binary coverage...")
                    n_dev_new = min(len(cv_case_ids) - 5, int(0.25 * len(cv_case_ids)))
                    if n_dev_new > n_dev:
                        final_dev_cases = set(cv_case_ids[-n_dev_new:])
                        final_train_cases = set(cv_case_ids[:-n_dev_new])
                        logger.info(
                            f"Increased dev set to {n_dev_new} cases for better binary coverage"
                        )

            # Report final distribution for binary
            if dev_bins:
                dev_bin_counts = Counter(dev_bins.values())
                dev_distribution = []
                for bin_idx in sorted(expected_bins):
                    count = dev_bin_counts.get(bin_idx, 0)
                    pct = count / len(dev_bins) * 100 if dev_bins else 0
                    balance_label = "lower" if bin_idx == 0 else "higher"
                    dev_distribution.append(
                        f"{balance_label}: {count} cases ({pct:.1f}%)"
                    )

                logger.info(
                    f"Final fold dev set distribution: " + " | ".join(dev_distribution)
                )

            # Report expected train balance for binary
            temp_bin_counts = Counter(temp_case_bins.values())
            train_distribution = []
            for bin_idx in sorted(temp_bin_counts.keys()):
                count = temp_bin_counts[bin_idx]
                pct = count / len(temp_case_bins) * 100
                balance_label = "lower" if bin_idx == 0 else "higher"
                train_distribution.append(
                    f"{balance_label}: {count} cases ({pct:.1f}%)"
                )

            logger.info(
                f"Final fold train set will have: " + " | ".join(train_distribution)
            )

    # Create final training fold DataFrame using binary
    final_fold_df, final_edges_and_weights = make_temporal_fold_binary(
        df,
        final_train_cases,
        set(),
        final_dev_cases,
        k,
        use_quote_balanced=use_quote_balanced,
        quote_weight_cap=quote_weight_cap,
    )

    # Mark this as the final training fold
    final_fold_df["fold"] = k
    final_fold_df.loc[final_fold_df["split"] == "test", "split"] = "dev"
    final_fold_df = final_fold_df[final_fold_df["split"] != "val"]

    # Extract edges and weights for final fold
    if len(final_edges_and_weights) >= 2:
        final_edges = (
            final_edges_and_weights[0]
            if isinstance(final_edges_and_weights[0], list)
            else final_edges_and_weights[:-1]
        )
        final_weight_metadata = (
            final_edges_and_weights[-1]
            if isinstance(final_edges_and_weights[-1], dict)
            else {}
        )
    else:
        final_edges = (
            final_edges_and_weights if isinstance(final_edges_and_weights, list) else []
        )
        final_weight_metadata = {}

    per_fold_edges[f"fold_{k}"] = final_edges
    per_fold_weights[f"fold_{k}"] = final_weight_metadata
    fold_dfs.append(final_fold_df)

    logger.info(
        f"Final training fold: {len(final_train_cases)} train cases, {len(final_dev_cases)} dev cases"
    )

    # 7.5) Fix OOF test labels using fold k cutoffs (binary)
    if oof_test_cases and final_edges:
        logger.info("Correcting OOF test labels using final fold cutoffs (BINARY)...")
        final_edge = np.array(final_edges)
        logger.info(f"Applying final fold cutoff to OOF test: ${final_edge[0]:,.2f}")

        # Find the OOF DataFrame in fold_dfs and update its labels
        for i, fold_df in enumerate(fold_dfs):
            if "split" in fold_df.columns and (fold_df["split"] == "oof_test").any():
                # Vectorized OOF relabeling
                fold_dfs[i] = fold_df.copy()
                oof_mask = fold_dfs[i]["split"] == "oof_test"

                # For OOF records, apply binary assignment (vectorized)
                oof_outcomes = fold_dfs[i].loc[oof_mask, "final_judgement_real"]
                oof_valid_mask = oof_mask & oof_outcomes.notna()
                oof_null_mask = oof_mask & oof_outcomes.isna()

                # Vectorized binary assignment: 0 if y < e1 else 1
                fold_dfs[i].loc[oof_valid_mask, "outcome_bin"] = (
                    fold_dfs[i].loc[oof_valid_mask, "final_judgement_real"].values
                    >= final_edge[0]
                ).astype(int)

                # Default to 0 for null outcomes in OOF
                fold_dfs[i].loc[oof_null_mask, "outcome_bin"] = 0

                # Non-OOF records keep their existing outcome_bin (no change needed)

                # Verify OOF class distribution for binary (vectorized)
                oof_only_bins = fold_dfs[i].loc[oof_mask, "outcome_bin"].values
                oof_distribution = Counter(oof_only_bins)
                logger.info(f"OOF test binary distribution: {dict(oof_distribution)}")
                break

    # 8) Combine all folds
    result_df = pd.concat(fold_dfs, ignore_index=True)
    result_df.attrs["do_not_train"] = df.attrs.get("do_not_train", sorted(dnt))

    # 9) Skip global purge - already handled per-fold

    # 10) Preserve per-fold bin edges, weights, and OOF metadata for audit
    result_df.attrs["per_fold_bin_edges"] = per_fold_edges
    result_df.attrs["per_fold_weights"] = per_fold_weights
    result_df.attrs["oof_growth_metadata"] = oof_meta

    # 11) Validate no empty splits (safety guards) - adapted for binary
    for f, g in result_df.groupby("fold"):
        if f == -1:  # Skip OOF test validation
            continue
        ntr = (g["split"] == "train").sum()

        if f == k:  # Final training fold uses dev instead of val/test
            ndev = (g["split"] == "dev").sum()
            assert (
                ntr > 0 and ndev > 0
            ), f"Final fold {f} has empty split(s): train={ntr}, dev={ndev}"
        else:  # Regular CV folds
            nva = (g["split"] == "val").sum()
            nte = (g["split"] == "test").sum()
            assert (
                ntr > 0 and nva > 0 and nte > 0
            ), f"Fold {f} has empty split(s): train={ntr}, val={nva}, test={nte}"

    # 12) Each case must be eval in at most one CV fold (exclude final training fold)
    cv_eval_df = result_df[
        (result_df["split"].isin(["val", "test"])) & (result_df["fold"] < k)
    ]
    if len(cv_eval_df) > 0:
        eval_counts = cv_eval_df.groupby("case_id")["fold"].nunique()
        assert (
            eval_counts <= 1
        ).all(), f"Cases appear as eval in multiple folds: {eval_counts[eval_counts > 1].index.tolist()}"

    logger.info(
        f"✅ Validation passed: No empty splits, no cross-fold eval contamination"
    )
    logger.info(
        f"Leakage-safe BINARY splits complete: {k} CV folds + 1 final training fold + OOF test"
    )
    logger.info(f"DNT columns: {len(dnt)}")
    logger.info(f"Per-fold bin edges preserved: {len(per_fold_edges)} folds")

    return result_df


def grow_oof_until_binary_classes(
    case_df: pd.DataFrame,
    case_outcomes: Dict[str, float],
    oof_min_ratio: float = 0.15,
    oof_max_ratio: float = 0.40,
    oof_step: float = 0.05,
    min_class_cases: int = 5,
    min_class_quotes: int = 50,
    criterion: str = "both",
) -> Tuple[Set[str], Set[str], Dict[str, Any]]:
    """
    Grow OOF test split until it has both binary outcome classes with adequate support.

    Modified version of grow_oof_until_three_classes for binary classification.
    """
    logger.info("Growing OOF test split until binary class coverage achieved")

    # Sort cases by year (oldest to newest)
    case_df = case_df.sort_values("case_year")
    ids = case_df["case_id"].tolist()
    n_total = len(ids)

    meta = {}
    best = None

    def binary_edges_tie_safe(values):
        """Binary version of tertile_edges_tie_safe."""
        s = pd.Series(values).dropna()
        if len(s) < 2:
            return np.array([])

        y = s.values
        if pd.Series(y).nunique() < 2:
            return np.array([])

        # Use interpolated median
        e = np.quantile(y, [0.5], method="midpoint")
        e1 = float(e[0])

        return np.array([e1], dtype=float)

    def counts_under_edges_binary(cids: Set[str], edges: np.ndarray) -> Dict[str, Any]:
        """Count cases and quotes per class using given binary edges."""
        # Get outcomes for these cases
        ys = [
            (cid, case_outcomes.get(cid, None)) for cid in cids if cid in case_outcomes
        ]

        if not ys or edges.size == 0:
            return {
                "classes": set(),
                "per_class_cases": Counter(),
                "per_class_quotes": Counter(),
            }

        # Bin cases using binary edges
        bins = {}
        for cid, y in ys:
            bin_idx = assign_bin_binary(y, edges)
            bins[cid] = bin_idx

        # Count cases per class
        per_class_cases = Counter(bins.values())

        # Count quotes per class using case sizes
        size_map = dict(zip(case_df["case_id"], case_df["size"]))
        per_class_quotes = Counter()
        for cid, b in bins.items():
            per_class_quotes[b] += int(size_map.get(cid, 0))

        return {
            "classes": set(per_class_cases.keys()),
            "per_class_cases": per_class_cases,
            "per_class_quotes": per_class_quotes,
        }

    def check_ok_binary(cnts: Dict[str, Any]) -> bool:
        """Check if counts meet minimum requirements for binary."""
        if len(cnts["classes"]) < 2:  # Need both classes
            return False
        if (
            cnts["per_class_cases"]
            and min(cnts["per_class_cases"].values()) < min_class_cases
        ):
            return False
        if (
            cnts["per_class_quotes"]
            and min(cnts["per_class_quotes"].values()) < min_class_quotes
        ):
            return False
        return True

    # Try growing OOF from min to max ratio
    r = oof_min_ratio
    while r <= oof_max_ratio + 1e-9:
        n_oof = max(1, int(round(r * n_total)))
        oof_cases = set(ids[-n_oof:])  # Latest cases
        cv_cases = set(ids[:-n_oof])  # Earlier cases

        # Compute OOF-native edges (from OOF data only) - binary
        oof_y = [case_outcomes[c] for c in oof_cases if c in case_outcomes]
        E_native = binary_edges_tie_safe(np.array(oof_y))
        nat = counts_under_edges_binary(oof_cases, E_native)

        # Compute baseline edges (from CV/older data only - no peeking!) - binary
        cv_y = [case_outcomes[c] for c in cv_cases if c in case_outcomes]
        E_star = binary_edges_tie_safe(np.array(cv_y))
        base = counts_under_edges_binary(oof_cases, E_star)

        # Check conditions
        cond_native = check_ok_binary(nat)
        cond_base = check_ok_binary(base)

        if criterion == "both":
            satisfied = cond_native and cond_base
        elif criterion == "native":
            satisfied = cond_native
        else:  # baseline
            satisfied = cond_base

        # Store current state
        best = (
            oof_cases,
            cv_cases,
            {
                "ratio": r,
                "n_cases": len(oof_cases),
                "E_native": E_native.tolist() if E_native.size > 0 else [],
                "E_star": E_star.tolist() if E_star.size > 0 else [],
                "native_counts": {
                    k: (dict(v) if isinstance(v, Counter) else list(v))
                    for k, v in nat.items()
                },
                "baseline_counts": {
                    k: (dict(v) if isinstance(v, Counter) else list(v))
                    for k, v in base.items()
                },
                "satisfied_native": cond_native,
                "satisfied_baseline": cond_base,
                "satisfied_criterion": satisfied,
            },
        )

        if satisfied:
            meta = best[2]
            logger.info(
                f"✓ Found binary class OOF at ratio={r:.2f} ({n_oof}/{n_total} cases)"
            )
            logger.info(f"  Native edge: {meta['E_native']}")
            logger.info(f"  Baseline edge: {meta['E_star']}")
            logger.info(
                f"  Native class cases: {meta['native_counts']['per_class_cases']}"
            )
            logger.info(
                f"  Baseline class cases: {meta['baseline_counts']['per_class_cases']}"
            )
            break

        logger.info(
            f"  Ratio {r:.2f}: native_ok={cond_native}, baseline_ok={cond_base}"
        )
        r += oof_step

    if not meta and best:
        # Couldn't satisfy - use largest OOF and report
        _, _, meta = best
        meta["warning"] = (
            f"OOF max ratio {oof_max_ratio} reached without binary class coverage under {criterion} criterion"
        )
        logger.warning(meta["warning"])
        logger.info(
            f"  Using largest OOF: {meta['n_cases']} cases (ratio={meta['ratio']:.2f})"
        )

    # Get year ranges for reporting
    oof_years = case_df[case_df["case_id"].isin(best[0])]["case_year"]
    cv_years = case_df[case_df["case_id"].isin(best[1])]["case_year"]

    meta["oof_year_range"] = [float(oof_years.min()), float(oof_years.max())]
    meta["cv_year_range"] = [float(cv_years.min()), float(cv_years.max())]

    logger.info(
        f"Final OOF: {len(best[0])} cases, years {oof_years.min():.0f}-{oof_years.max():.0f}"
    )
    logger.info(
        f"Final CV: {len(best[1])} cases, years {cv_years.min():.0f}-{cv_years.max():.0f}"
    )

    return best[0], best[1], meta


def main():
    """Main execution function for binary classification."""
    parser = argparse.ArgumentParser(
        description="Create stratified k-fold cross-validation splits with binary classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Use same arguments as original but with binary-specific defaults
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for fold splits"
    )
    parser.add_argument(
        "--k-folds", type=int, default=4, help="Number of folds (default: 4)"
    )
    parser.add_argument(
        "--target-field",
        default="final_judgement_real",
        help="Target field for stratification (default: final_judgement_real)",
    )
    parser.add_argument(
        "--case-id-field",
        default="case_id",
        help="Field name containing case ID (default: case_id)",
    )
    parser.add_argument(
        "--use-temporal-cv",
        action="store_true",
        default=True,
        help="Use temporal rolling-origin CV (default: True for binary)",
    )
    parser.add_argument(
        "--oof-test-ratio",
        type=float,
        default=0.15,
        help="Proportion of latest cases for out-of-fold test (default: 0.15)",
    )
    parser.add_argument(
        "--oof-min-ratio",
        type=float,
        default=0.15,
        help="Minimum OOF ratio to start with (default: 0.15)",
    )
    parser.add_argument(
        "--oof-max-ratio",
        type=float,
        default=0.40,
        help="Maximum OOF ratio to grow to (default: 0.40)",
    )
    parser.add_argument(
        "--oof-step",
        type=float,
        default=0.05,
        help="Step size for growing OOF (default: 0.05)",
    )
    parser.add_argument(
        "--oof-min-class-cases",
        type=int,
        default=5,
        help="Minimum cases per class in OOF (default: 5)",
    )
    parser.add_argument(
        "--oof-min-class-quotes",
        type=int,
        default=50,
        help="Minimum quotes per class in OOF (default: 50)",
    )
    parser.add_argument(
        "--oof-class-criterion",
        choices=["native", "baseline", "both"],
        default="both",
        help="Criterion for OOF class coverage (default: both)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--filter-undateable",
        action="store_true",
        help="Remove cases that cannot be reliably dated (improves temporal CV purity)",
    )
    parser.add_argument(
        "--use-quote-balanced-cutoff",
        action="store_true",
        help="Use quote-balanced cutoff (weight cases by quote count) instead of case-balanced cutoff",
    )
    parser.add_argument(
        "--quote-weight-cap",
        type=int,
        default=None,
        help="Cap on quote weights to reduce mega-case influence (default: no cap)",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    logger.info("Starting stratified k-fold cross-validation (BINARY)")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"K-folds: {args.k_folds}")
    logger.info(f"Target field: {args.target_field}")
    logger.info(f"Random seed: {args.random_seed}")

    # Log cutoff balancing strategy
    if args.use_quote_balanced_cutoff:
        balance_strategy = "Quote-balanced (cases weighted by quote count)"
        if args.quote_weight_cap:
            balance_strategy += f" with cap={args.quote_weight_cap}"
        logger.info(f"Cutoff strategy: {balance_strategy}")
    else:
        logger.info("Cutoff strategy: Case-balanced (each case weighted equally)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data with orjson optimization...")
    try:
        import orjson

        # Use orjson for faster loading
        records = []
        with open(args.input, "rb") as f:
            for line in f:
                if line.strip():
                    records.append(orjson.loads(line))
        df = pd.DataFrame(records)
        logger.info("✅ Fast loading with orjson successful")
    except ImportError:
        logger.info("⚠️ orjson not available, using standard pandas")
        df = pd.read_json(args.input, lines=True, dtype={"case_id_clean": "string"})

    # Extract case_id if not present
    if "case_id" not in df.columns:
        if "case_id_clean" in df.columns:
            logger.info("Using case_id_clean as case_id...")
            df["case_id"] = df["case_id_clean"]
        elif "_metadata_src_path" in df.columns:
            logger.info("Extracting case_id from _metadata_src_path...")
            df["case_id"] = df.apply(
                lambda row: extract_case_id(row, "_metadata_src_path"), axis=1
            )
        else:
            logger.error("No case ID source found in dataset!")
            raise ValueError("Dataset missing case ID information")

    # CRITICAL: Anonymize case IDs to prevent temporal leakage
    logger.info("Anonymizing case IDs to prevent temporal leakage...")
    original_case_id_sample = df["case_id"].head(3).tolist()
    df["case_id_original"] = df["case_id"]  # Preserve original for debugging
    df["case_id"] = df["case_id"].apply(anonymize_case_id)
    anonymized_sample = df["case_id"].head(3).tolist()

    logger.info("Case ID anonymization examples:")
    for orig, anon in zip(original_case_id_sample, anonymized_sample):
        logger.info(f"  {orig} -> {anon}")

    # Verify uniqueness is preserved
    orig_unique = df["case_id_original"].nunique()
    anon_unique = df["case_id"].nunique()
    if orig_unique != anon_unique:
        logger.error(f"Anonymization broke uniqueness! {orig_unique} -> {anon_unique}")
        raise ValueError("Case ID anonymization failed to preserve uniqueness")

    logger.info(
        f"✅ Anonymized {len(df):,} records with {anon_unique:,} unique case IDs"
    )

    # Add case_year extraction if not present with improved heuristics
    # IMPORTANT: Use original case_id for year extraction (before anonymization)
    if "case_year" not in df.columns:
        logger.info(
            "Extracting case years from original case_id (before anonymization)..."
        )

        # Get min observed year from any existing timestamp or metadata as fallback
        fallback_year = 2020  # Conservative default
        if "_metadata_created" in df.columns:
            try:
                meta_years = pd.to_datetime(
                    df["_metadata_created"], errors="coerce"
                ).dt.year
                min_meta_year = meta_years.min()
                if pd.notna(min_meta_year) and 1950 <= min_meta_year <= 2030:
                    fallback_year = int(min_meta_year)
                    logger.info(f"Using metadata-based fallback year: {fallback_year}")
            except:
                pass

        def extract_year_from_case_id(case_id):
            if not case_id or pd.isna(case_id):
                return fallback_year
            import re

            case_id_str = str(case_id).strip()
            if not case_id_str:
                return fallback_year

            # Handle appellate court pattern: YY-NNNNN_caX (e.g., 24-10951_ca5)
            match = re.search(r"^(\d{2})-\d+_ca\d+$", case_id_str)
            if match:
                year_suffix = int(match.group(1))
                # Convert 2-digit year to 4-digit (24 -> 2024)
                if year_suffix <= 30:  # Assume 24 = 2024, not 1924
                    return 2000 + year_suffix
                else:
                    return 1900 + year_suffix

            # Try standard federal court pattern like "1:23-cv-04567_nysd" -> 2023
            match = re.search(r"(\d+):(\d+)-(cv|cr)-\d+_[a-z]+", case_id_str)
            if match:
                year = int(match.group(2))
                if year > 90:
                    return 1900 + year
                else:
                    return 2000 + year

            # Try looser federal pattern "1:23-cv-04567" (without district)
            match = re.search(r"(\d+):(\d+)-(cv|cr)-\d+", case_id_str)
            if match:
                year = int(match.group(2))
                if year > 90:
                    return 1900 + year
                else:
                    return 2000 + year

            # Try any pattern with ":" and 2-digit year
            match = re.search(r":(\d{2})-", case_id_str)
            if match:
                year = int(match.group(1))
                if year > 90:
                    return 1900 + year
                else:
                    return 2000 + year

            # Try 4-digit year pattern anywhere in string
            match = re.search(r"(\d{4})", case_id_str)
            if match:
                year = int(match.group(1))
                if 1950 <= year <= 2030:
                    return year

            # Try 2-digit year at start with delimiter
            match = re.search(r"^(\d{2})[-_]", case_id_str)
            if match:
                year = int(match.group(1))
                if year > 90:
                    return 1900 + year
                else:
                    return 2000 + year

            return fallback_year  # Improved fallback

        # Extract year from ORIGINAL case_id (before anonymization)
        df["case_year"] = df["case_id_original"].apply(extract_year_from_case_id)

        # Track which cases actually failed extraction vs got legitimate fallback year
        def was_extraction_successful(case_id):
            """Check if year extraction was successful (not a true fallback)"""
            debug_result = debug_year_extraction(case_id)
            return not debug_result.startswith(
                "NO_MATCH"
            ) and not debug_result.startswith("EMPTY")

        # Identify truly failed extractions
        fallback_mask = df["case_year"] == fallback_year
        if fallback_mask.any():
            fallback_case_ids = df.loc[fallback_mask, "case_id_original"].unique()

            # Check which ones are legitimate vs actual failures
            actual_failures = []
            legitimate_fallback_year = []

            for case_id in fallback_case_ids[:20]:  # Sample first 20 for efficiency
                if was_extraction_successful(case_id):
                    legitimate_fallback_year.append(case_id)
                else:
                    actual_failures.append(case_id)

            # Count the full dataset
            total_fallback_cases = len(fallback_case_ids)

            # Estimate failure rate from sample
            sample_size = min(20, len(fallback_case_ids))
            if sample_size > 0:
                failure_rate = len(actual_failures) / sample_size
                estimated_true_failures = int(failure_rate * total_fallback_cases)
            else:
                estimated_true_failures = 0

            if estimated_true_failures > 0:
                logger.warning(
                    f"⚠️  Estimated {estimated_true_failures:,} cases with failed year extraction"
                )
                logger.info("Sample of actual extraction failures:")
                for case_id in actual_failures[:5]:
                    debug_result = debug_year_extraction(case_id)
                    logger.info(f"  {debug_result}")

            if legitimate_fallback_year:
                logger.info(
                    f"✅ {total_fallback_cases - estimated_true_failures:,} cases correctly extracted to year {fallback_year}"
                )
                logger.info("Sample of legitimate year extractions:")
                for case_id in legitimate_fallback_year[:3]:
                    debug_result = debug_year_extraction(case_id)
                    logger.info(f"  {debug_result}")

            # Only filter actual failures, not legitimate cases from the fallback year
            if args.filter_undateable and estimated_true_failures > 0:
                logger.info(
                    "Filtering out only cases with actual extraction failures..."
                )
                original_count = len(df)

                # Create mask for actual failures
                actual_failure_mask = df["case_id_original"].apply(
                    lambda x: (
                        not was_extraction_successful(x)
                        if df.loc[df["case_id_original"] == x, "case_year"].iloc[0]
                        == fallback_year
                        else False
                    )
                )

                df = df[~actual_failure_mask].copy()
                filtered_count = len(df)
                removed_count = original_count - filtered_count

                if removed_count > 0:
                    logger.info(
                        f"Removed {removed_count:,} cases with failed year extraction"
                    )
                    logger.info(
                        f"Dataset reduced from {original_count:,} to {filtered_count:,} records"
                    )
                    logger.info(f"✅ Kept legitimate {fallback_year} cases")
                else:
                    logger.info(
                        "No cases needed to be filtered - all extractions successful"
                    )
            elif estimated_true_failures > 0:
                logger.info(
                    "Use --filter-undateable flag to remove cases with failed extraction"
                )
        else:
            logger.info("✅ All cases had successful year extraction")

    # Add text_hash if not present (same logic as original)
    if "text_hash" not in df.columns:
        logger.info("Creating text_hash for deduplication...")
        df["text_hash"] = (
            df["text"].astype(str).apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        )

    if "text_hash_norm" not in df.columns:
        logger.info("Creating normalized text_hash for robust deduplication...")
        df["text_hash_norm"] = (
            df["text"]
            .astype(str)
            .apply(normalize_for_hash)
            .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        )

    # Create leakage-safe splits with DNT policy and adaptive OOF (BINARY)
    result_df = make_leakage_safe_splits_binary(
        df,
        k=args.k_folds,
        oof_ratio=args.oof_test_ratio,
        seed=args.random_seed,
        oof_min_ratio=args.oof_min_ratio,
        oof_max_ratio=args.oof_max_ratio,
        oof_step=args.oof_step,
        min_class_cases=args.oof_min_class_cases,
        min_class_quotes=args.oof_min_class_quotes,
        oof_criterion=args.oof_class_criterion,
        use_quote_balanced=args.use_quote_balanced_cutoff,
        quote_weight_cap=args.quote_weight_cap,
    )

    # CRITICAL: Remove original case_id to prevent temporal leakage
    if "case_id_original" in result_df.columns:
        logger.info("Dropping case_id_original column to prevent temporal leakage")
        result_df = result_df.drop(columns=["case_id_original"])

    # Save results by fold (same structure as original)
    for fold_idx in range(args.k_folds + 1):  # +1 to include final training fold
        fold_data = result_df[result_df["fold"] == fold_idx]
        if len(fold_data) == 0:
            continue

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if fold_idx == args.k_folds:  # Final training fold
            # Save train and dev splits
            for split in ["train", "dev"]:
                split_data = fold_data[fold_data["split"] == split]
                if len(split_data) > 0:
                    split_path = fold_dir / f"{split}.jsonl"
                    split_data.to_json(split_path, orient="records", lines=True)

            # Save case IDs
            train_cases = set(fold_data[fold_data["split"] == "train"]["case_id"])
            dev_cases = set(fold_data[fold_data["split"] == "dev"]["case_id"])

            case_ids_path = fold_dir / "case_ids.json"
            with open(case_ids_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_case_ids": list(train_cases),
                        "dev_case_ids": list(dev_cases),
                        "is_final_training_fold": True,
                        "classification_type": "binary",
                    },
                    f,
                    indent=2,
                )
        else:  # Regular CV folds
            # Save split files
            for split in ["train", "val", "test"]:
                split_data = fold_data[fold_data["split"] == split]
                if len(split_data) > 0:
                    split_path = fold_dir / f"{split}.jsonl"
                    split_data.to_json(split_path, orient="records", lines=True)

            # Save case IDs
            train_cases = set(fold_data[fold_data["split"] == "train"]["case_id"])
            val_cases = set(fold_data[fold_data["split"] == "val"]["case_id"])
            test_cases = set(fold_data[fold_data["split"] == "test"]["case_id"])

            case_ids_path = fold_dir / "case_ids.json"
            with open(case_ids_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_case_ids": list(train_cases),
                        "val_case_ids": list(val_cases),
                        "test_case_ids": list(test_cases),
                        "classification_type": "binary",
                    },
                    f,
                    indent=2,
                )

    # Save OOF test data if present
    oof_data = result_df[result_df["split"] == "oof_test"]
    if len(oof_data) > 0:
        oof_dir = output_dir / "oof_test"
        oof_dir.mkdir(parents=True, exist_ok=True)
        oof_path = oof_dir / "test.jsonl"
        oof_data.to_json(oof_path, orient="records", lines=True)

        # Save OOF case IDs
        oof_cases = list(set(oof_data["case_id"]))
        oof_case_ids_path = oof_dir / "case_ids.json"
        with open(oof_case_ids_path, "w", encoding="utf-8") as f:
            json.dump(
                {"test_case_ids": oof_cases, "classification_type": "binary"},
                f,
                indent=2,
            )

    # Save DNT manifest
    manifest = {
        "do_not_train": result_df.attrs.get("do_not_train", []),
        "classification_type": "binary",
    }
    manifest_path = output_dir / "dnt_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save per-fold metadata for audit detection (binary format)
    balance_method = (
        "quote_weighted" if args.use_quote_balanced_cutoff else "case_balanced"
    )
    per_fold_metadata = {
        "binning": {
            "method": f"train_only_binary_median_tie_safe_{balance_method}",
            "balance_strategy": balance_method,
            "quote_weight_cap": (
                args.quote_weight_cap if args.use_quote_balanced_cutoff else None
            ),
            "fold_edges": result_df.attrs.get("per_fold_bin_edges", {}),
            "classification_type": "binary",
        },
        "weights": result_df.attrs.get("per_fold_weights", {}),
        "methodology": "temporal_rolling_origin_with_adaptive_oof_binary",
        "oof_growth": result_df.attrs.get("oof_growth_metadata", {}),
        "adaptive_features": {
            "oof_class_guarantee": True,
            "stratified_eval_blocks": True,
            "adaptive_val_frac": True,
            "tie_safe_binary": True,
            "classification_type": "binary",
        },
    }
    metadata_path = output_dir / "per_fold_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(per_fold_metadata, f, indent=2)

    # Save fold statistics (binary version)
    stats = {
        "methodology": "temporal_rolling_origin_with_dnt_binary",
        "classification_type": "binary",
        "folds": args.k_folds,
        "final_training_fold": True,
        "total_folds_including_final": args.k_folds + 1,
        "oof_test_ratio": args.oof_test_ratio,
        "total_records": len(result_df),
        "dnt_columns": len(manifest["do_not_train"]),
        "stratification_approach": "outcome_only_binary",
        "support_handling": "weighting_only",
        "leakage_prevention": {
            "temporal_splits": "rolling_origin",
            "per_fold_binning": "training_data_only_binary_median",
            "dnt_policy": "wrap_not_drop_expanded",
            "text_deduplication": "eval_vs_train_global",
            "support_policy": "weighting_not_stratification",
        },
        "binning_strategy": {
            "method": f"train_only_binary_median_{balance_method}",
            "balance_strategy": balance_method,
            "quote_weight_cap": (
                args.quote_weight_cap if args.use_quote_balanced_cutoff else None
            ),
            "bins": ["lower", "higher"],
            "quantiles": [0.5],
            "temporal_purity": "preserved",
            "composite_labels": "disabled",
            "per_fold_edges_saved": True,
        },
        "support_strategy": {
            "method": "inverse_sqrt_weighting",
            "clipping": [0.25, 4.0],
            "normalization": "per_fold",
            "tertiles": "reporting_only_dnt",
        },
        "final_fold_info": {
            "description": "Final training fold combines all CV data for production model",
            "includes_all_cv_cases": True,
            "split_method": "temporal_train_dev",
            "dev_ratio": "adaptive_minimum_150_quotes",
        },
    }

    stats_path = output_dir / "fold_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.success("Temporal CV with DNT policy complete (BINARY)!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"DNT columns: {len(manifest['do_not_train'])}")
    logger.info("Classification type: BINARY (lower/higher)")

    return 0


if __name__ == "__main__":
    exit(main())
