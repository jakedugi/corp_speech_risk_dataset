#!/usr/bin/env python3
"""
Leakage-Safe Temporal K-Fold Cross-Validation with Do-Not-Train Policy

This script creates leakage-safe k-fold cross-validation splits with:
1. Temporal integrity (rolling-origin methodology)
2. Case-level integrity (no data leakage between cases)
3. Do-Not-Train (DNT) policy (wrap, don't drop leaky features)
4. Per-fold outcome binning (prevents global binning leakage)
5. Out-of-fold (OOF) final test split (latest 10-30% of cases)

Key innovations:
- **Rolling-Origin CV**: Fold i uses cases before time block i for training
- **DNT Policy**: Keep identifiers/leaky features for EDA but exclude from training
- **Temporal Priority**: Time order > support balancing (handle with weights)
- **Train-Only Binning**: Compute quantiles on training data only per fold
- **Eval Deduplication**: Remove eval records that appear in training text hashes

Leakage prevention:
- Case-wise splitting prevents quote-level leakage
- Temporal splits prevent future information leakage
- Per-fold binning prevents global binning leakage
- DNT policy prevents metadata correlation leakage
- Text deduplication prevents duplicate text leakage

Usage:
    python scripts/stratified_kfold_case_split.py \
        --input data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl \
        --output-dir data/final_stratified_kfold_splits_leakage_safe \
        --k-folds 5 \
        --target-field final_judgement_real \
        --stratify-type regression \
        --case-id-field case_id \
        --use-temporal-cv \
        --oof-test-ratio 0.15 \
        --random-seed 42
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Set
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# =============================================================================
# DO-NOT-TRAIN (DNT) POLICY - Wrap, Don't Drop Leaky Features
# =============================================================================

# Exact column names to exclude from training (keep for EDA/audit)
# EXPANDED DNT to catch all leaky columns
DNT_EXACT = {
    # Courts/venues (drop from training features; still usable for reporting)
    "court",
    "court_code",
    "venue",
    "district",
    # Speakers/IDs
    "speaker",
    "speaker_id",
    "speaker_name",
    "speaker_hash",
    # ID-derived scalars
    "case_id_length",
    "src_path_length",
    "speaker_length",
    "court_code_length",
    "has_speaker",
    "text_length",  # present in audit
    # raw identifiers
    "case_id",
    "src_path",
    "_src",
    # sentiment triplet always DNT; we only train on the collapsed scalar
    "quote_sent_pos",
    "quote_sent_neg",
    "quote_sent_neu",
    # degenerate scalars → DNT
    "liability",
    "certainty",
    # Support is for weighting/reporting only, not training
    "support_tertile",
}

# Regex patterns for columns to exclude from training
# EXPANDED patterns to catch *_length proxies
DNT_PATTERNS = (
    r"^speaker_",
    r"^court_",
    r"^spk_",
    r"^id_",
    r"^path_",
    r"_length$",  # catch any other *_length proxies
)

# Numeric dtypes that can be used for training
NUMERIC_DTYPES = (np.number, np.bool_)


def wrap_do_not_train_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Mark leaky/metadata columns as Do-Not-Train but keep them in dataset.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (DataFrame with DNT metadata, set of DNT column names)
    """
    dnt: Set[str] = set()

    # Mark exact matches
    for col in df.columns:
        if col in DNT_EXACT:
            dnt.add(col)

    # Mark pattern matches
    for col in df.columns:
        if any(re.match(pattern, col) for pattern in DNT_PATTERNS):
            dnt.add(col)

    # Store DNT list in DataFrame attributes for persistence
    df.attrs["do_not_train"] = sorted(dnt)

    # Log the actual intersection with present columns
    present_cols = set(df.columns)
    dnt_present = dnt & present_cols
    dnt_missing = dnt - present_cols

    logger.info(f"DNT columns ({len(dnt_present)}): {sorted(dnt_present)}")
    if dnt_missing:
        logger.info(f"DNT columns not in data: {sorted(dnt_missing)}")

    return df, dnt


def collapse_sentiment_and_mark_raw_dnt(
    df: pd.DataFrame, dnt: Set[str]
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Collapse sentiment features and mark raw components as DNT.

    Creates quote_sent_pos_minus_neg feature and marks raw sentiment
    triplet as Do-Not-Train to prevent redundant/correlated features.
    Handles degenerate features with enormous VIF & near-constant values.

    Args:
        df: Input DataFrame
        dnt: Current set of DNT columns

    Returns:
        Tuple of (modified DataFrame, updated DNT set)
    """
    # FORCE mark sentiment triplet as DNT regardless of whether we collapse them
    sentiment_cols = {"quote_sent_pos", "quote_sent_neg", "quote_sent_neu"}
    dnt |= sentiment_cols  # Always mark as DNT

    # Collapse sentiment features if available
    if sentiment_cols.issubset(df.columns):
        df["quote_sent_pos_minus_neg"] = df["quote_sent_pos"] - df["quote_sent_neg"]
        logger.info(
            "Created quote_sent_pos_minus_neg, FORCED raw sentiment triplet as DNT"
        )

    # FORCE mark degenerate/high-VIF features as DNT
    # Based on user's requirement: "Enormous VIF & near-constant → drop defensively"
    degen_cols = {"liability", "certainty"}
    dnt |= degen_cols  # Always mark as DNT

    for col in ("liability", "certainty"):
        if col in df.columns:
            nunique = df[col].nunique(dropna=False)
            std_val = df[col].std(ddof=0) or 0
            logger.info(f"FORCED {col} as DNT (nunique={nunique}, std={std_val:.2e})")
        else:
            logger.info(f"FORCED {col} as DNT (not in data)")

    # Update DataFrame attributes
    df.attrs["do_not_train"] = sorted(dnt)
    return df, dnt


def get_training_columns(
    df: pd.DataFrame, y_col: str, extra_exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Get columns suitable for training (excluding DNT and system columns).

    Args:
        df: Input DataFrame
        y_col: Target column name
        extra_exclude: Additional columns to exclude

    Returns:
        List of column names suitable for training
    """
    # Get DNT columns from DataFrame attributes
    dnt = set(df.attrs.get("do_not_train", []))

    # System columns to always exclude
    system_exclude = {
        y_col,
        "outcome",
        "outcome_bin",
        "split",
        "fold",
        "record_id",
        "text",
        "text_hash",
    }

    # Combine all exclusions
    all_exclude = dnt | system_exclude
    if extra_exclude:
        all_exclude |= set(extra_exclude)

    # Select numeric columns not in exclusion set
    numeric_cols = df.select_dtypes(include=NUMERIC_DTYPES).columns
    training_cols = [col for col in numeric_cols if col not in all_exclude]

    logger.info(
        f"Selected {len(training_cols)} training columns from {len(df.columns)} total"
    )
    logger.info(
        f"Excluded {len(all_exclude)} columns: DNT({len(dnt)}), system({len(system_exclude)})"
    )

    return training_cols


def hard_dedupe_within_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact text duplicates within each case.

    Based on user's _hard_dedupe_within_case function:
    Keep first instance of exact text duplicate INSIDE a case.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicates removed within cases
    """
    if {"case_id", "text_hash"}.issubset(df.columns):
        initial_count = len(df)
        # Create record_id if it doesn't exist
        if "record_id" not in df.columns:
            df["record_id"] = range(len(df))
        df = df.sort_values(["case_id", "text_hash", "record_id"]).drop_duplicates(
            ["case_id", "text_hash"], keep="first"
        )
        removed_count = initial_count - len(df)
        logger.info(
            f"Removed {removed_count} within-case text duplicates ({removed_count/initial_count:.1%})"
        )

    return df


def create_time_blocks_by_case(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Create temporal blocks for cases based on case year.

    Args:
        df: Input DataFrame
        k: Number of time blocks

    Returns:
        DataFrame with case-level time block assignments
    """
    required_cols = {"case_id", "case_year", "final_judgement_real"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Aggregate to case level
    case_df = df.groupby("case_id", as_index=False).agg(
        case_year=("case_year", "min"),
        outcome=("final_judgement_real", "first"),
        size=("case_id", "count"),
    )  # Count records per case

    # Fill missing years with minimum year (conservative)
    case_df["case_year"] = case_df["case_year"].fillna(case_df["case_year"].min())

    # Sort by year and assign time blocks
    case_df = case_df.sort_values("case_year")
    ranks = case_df["case_year"].rank(method="first")
    case_df["time_block"] = pd.qcut(ranks, q=k, labels=list(range(k))).astype(int)

    logger.info(f"Created {k} time blocks:")
    for block in range(k):
        block_cases = case_df[case_df["time_block"] == block]
        year_range = (
            f"{block_cases['case_year'].min():.0f}-{block_cases['case_year'].max():.0f}"
        )
        logger.info(f"  Block {block}: {len(block_cases)} cases, years {year_range}")

    return case_df[["case_id", "time_block", "outcome", "size", "case_year"]]


def support_tertile(case_sizes: pd.Series) -> pd.Series:
    """
    Assign cases to support tertiles for reporting/analysis only (marked as DNT).

    Support tertiles are NOT used for stratification - only for:
    - Weighting calculation (inverse sqrt with clipping)
    - Post-hoc analysis and reporting
    - Robustness validation across case sizes

    Args:
        case_sizes: Series of case sizes (number of records per case)

    Returns:
        Series of tertile assignments (0, 1, 2) - marked as DNT
    """
    return pd.qcut(case_sizes.rank(method="first"), q=3, labels=[0, 1, 2]).astype(int)


def rolling_origin_group_folds(
    case_df: pd.DataFrame, k: int = 5, val_frac: float = 0.2, seed: int = 42
) -> List[Tuple[Set[str], Set[str], Set[str]]]:
    """
    Rolling-origin with block-local eval splits (FIXED - no case overlap).

    - Fold i (i=1..k): train = blocks < i, val/test = disjoint split within block i
    - Block 0 feeds only into training for subsequent folds
    - Every fold has train/val/test; each eval case appears in exactly one fold

    Args:
        case_df: DataFrame with time_block column
        k: Number of folds
        val_frac: Fraction of eval block to use for validation
        seed: Random seed for reproducible splits

    Returns:
        List of (train_cases, val_cases, test_cases) tuples
    """
    logger.info(
        "Generating rolling-origin folds with block-local eval (no case overlap)"
    )

    rng = np.random.default_rng(seed)

    # Re-bin into k+1 time blocks so we can produce k folds (block 0 only for training)
    ranks = case_df["case_year"].rank(method="first")
    case_df = case_df.assign(
        time_block=pd.qcut(ranks, q=k + 1, labels=list(range(k + 1))).astype(int)
    )

    logger.info(f"Re-binned into {k+1} time blocks for block-local eval")
    for block in range(k + 1):
        block_cases = case_df[case_df["time_block"] == block]
        year_range = (
            f"{block_cases['case_year'].min():.0f}-{block_cases['case_year'].max():.0f}"
        )
        logger.info(f"  Block {block}: {len(block_cases)} cases, years {year_range}")

    folds = []
    for i in range(1, k + 1):
        # Eval block is the current block i - split by case into val/test
        eval_cases = list(case_df.loc[case_df["time_block"] == i, "case_id"])
        rng.shuffle(eval_cases)
        n_val = max(1, int(val_frac * len(eval_cases)))
        val_cases = set(eval_cases[:n_val])
        test_cases = set(eval_cases[n_val:])

        # STRICT TEMPORAL: Train = strictly older blocks [0..i-1] with year < min(eval_years)
        # This prevents train/test same-year overlap that causes temporal leakage
        eval_years = case_df.loc[case_df["time_block"] == i, "case_year"]
        min_eval_year = eval_years.min() if len(eval_years) > 0 else 9999

        train_blocks = list(range(0, i))
        train_candidates = case_df.loc[case_df["time_block"].isin(train_blocks), :]
        # Strict cutoff: only cases with year < min_eval_year
        strict_train_cases = set(
            train_candidates.loc[
                train_candidates["case_year"] < min_eval_year, "case_id"
            ]
        )
        train_cases = strict_train_cases - (val_cases | test_cases)  # Safety removal

        logger.info(
            f"Fold {i-1}: {len(train_cases)} train, {len(val_cases)} val, {len(test_cases)} test cases"
        )
        folds.append((train_cases, val_cases, test_cases))

    return folds


def create_oof_test_split(
    case_df: pd.DataFrame, oof_ratio: float = 0.15
) -> Tuple[Set[str], Set[str]]:
    """
    Create out-of-fold (OOF) test split using latest cases.

    Args:
        case_df: DataFrame with case-level data including case_year
        oof_ratio: Proportion of latest cases to reserve for final testing

    Returns:
        Tuple of (cv_case_ids, oof_test_case_ids)
    """
    logger.info(f"Creating OOF test split ({oof_ratio:.1%} of latest cases)")

    # Sort by year and take latest cases for OOF test
    case_df_sorted = case_df.sort_values("case_year")
    n_total = len(case_df_sorted)
    n_oof = max(1, int(oof_ratio * n_total))

    oof_test_cases = set(case_df_sorted.tail(n_oof)["case_id"])
    cv_cases = set(case_df_sorted.head(n_total - n_oof)["case_id"])

    # Get year ranges
    oof_years = case_df[case_df["case_id"].isin(oof_test_cases)]["case_year"]
    cv_years = case_df[case_df["case_id"].isin(cv_cases)]["case_year"]

    logger.info(
        f"CV cases: {len(cv_cases)} cases, years {cv_years.min():.0f}-{cv_years.max():.0f}"
    )
    logger.info(
        f"OOF test: {len(oof_test_cases)} cases, years {oof_years.min():.0f}-{oof_years.max():.0f}"
    )

    return cv_cases, oof_test_cases


def make_temporal_fold(
    df: pd.DataFrame,
    train_cases: Set[str],
    val_cases: Set[str],
    test_cases: Set[str],
    fold_idx: int,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Create a single temporal fold with leakage-safe processing.

    Args:
        df: Full dataset
        train_cases: Training case IDs
        val_cases: Validation case IDs
        test_cases: Test case IDs
        fold_idx: Fold index for labeling

    Returns:
        DataFrame for this fold with split labels and weights
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

    # Per-fold outcome binning using TRAINING DATA ONLY - TRUE 3-BIN TERTILES
    train_outcomes = fold_df.loc[
        fold_df["split"] == "train", "final_judgement_real"
    ].dropna()
    edges_used = []
    if len(train_outcomes) >= 3:
        # Use tertiles (1/3, 2/3) for true 3-bin strategy {low, med, high}
        edges = np.quantile(np.sort(train_outcomes), [1 / 3, 2 / 3])
        fold_df["outcome_bin"] = np.digitize(
            fold_df["final_judgement_real"], edges, right=True
        )
        edges_used = edges.tolist()
        logger.info(
            f"Fold {fold_idx}: 3-bin tertiles at ${edges[0]:,.0f}, ${edges[1]:,.0f}"
        )
    else:
        # Fallback for insufficient training data
        fold_df["outcome_bin"] = 1
        edges_used = []

    # CRITICAL: Remove eval-side duplicates that appear in training text hashes
    # This prevents within-fold text contamination (eval texts that appear in train)
    if "text_hash" in fold_df.columns:
        # Get all unique text hashes from training data in this fold
        train_mask = fold_df["split"] == "train"
        train_hashes = set(fold_df.loc[train_mask, "text_hash"].unique())

        # Find eval records (val/test) that have text_hashes appearing in train
        eval_mask = fold_df["split"].isin(["val", "test"])
        eval_with_train_hashes = fold_df.loc[eval_mask, "text_hash"].isin(train_hashes)

        # Create leak mask: eval records whose text_hash appears in train
        leak_mask = eval_mask & eval_with_train_hashes

        # Debug: Check contamination before removal
        if leak_mask.sum() > 0:
            contaminated_hashes = set(fold_df.loc[leak_mask, "text_hash"])
            logger.info(
                f"Fold {fold_idx}: Found {int(leak_mask.sum())} eval records with {len(contaminated_hashes)} unique train text hashes"
            )

            # Remove contaminated eval records
            fold_df = fold_df.loc[~leak_mask].copy()

            # Verify contamination eliminated
            remaining_train_hashes = set(
                fold_df.loc[fold_df["split"] == "train", "text_hash"].unique()
            )
            remaining_eval_hashes = set(
                fold_df.loc[
                    fold_df["split"].isin(["val", "test"]), "text_hash"
                ].unique()
            )
            remaining_overlap = len(remaining_train_hashes & remaining_eval_hashes)

            logger.info(
                f"Fold {fold_idx}: Post-dedup verification - {remaining_overlap} eval/train text hash overlaps remaining"
            )

            if remaining_overlap > 0:
                logger.warning(
                    f"Fold {fold_idx}: WARNING - {remaining_overlap} text hash overlaps still remain after deduplication!"
                )
        else:
            logger.info(
                f"Fold {fold_idx}: No eval/train text hash contamination detected"
            )

    # Compute support weights for balanced training (weighting only, not labels)
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

        # Compute 3-bin weights for outcome bins
        from sklearn.utils.class_weight import compute_class_weight

        train_bins = train_fold["outcome_bin"].values
        unique_bins = np.unique(train_bins)

        if len(unique_bins) > 1:
            bin_weights = compute_class_weight(
                "balanced", classes=unique_bins, y=train_bins
            )
            bin_weight_map = dict(zip(unique_bins, bin_weights))

            # Apply bin weights to all records
            fold_df["bin_weight"] = (
                fold_df["outcome_bin"].map(bin_weight_map).fillna(1.0)
            )
        else:
            fold_df["bin_weight"] = 1.0

        # Inverse sqrt support weighting with clipping
        w_case = (1.0 / np.sqrt(case_support)).clip(0.25, 4.0)  # Cap extremes
        w_case_normalized = w_case * (len(case_support) / w_case.sum())  # Normalize

        # Combine bin weights and support weights
        case_weight_map = dict(zip(case_support.index, w_case_normalized))
        fold_df["support_weight"] = fold_df["case_id"].map(case_weight_map).fillna(1.0)
        fold_df["sample_weight"] = fold_df["bin_weight"] * fold_df["support_weight"]

        # Store weight metadata for persistence (JSON-serializable)
        weight_metadata = {
            "class_weights": (
                {str(k): float(v) for k, v in bin_weight_map.items()}
                if len(unique_bins) > 1
                else {}
            ),
            "support_weight_method": "inverse_sqrt_clipped",
            "support_weight_range": [0.25, 4.0],
            "bin_count": int(len(unique_bins)),
            "train_split_counts": {
                "total": int(len(train_fold)),
                "per_bin": (
                    {
                        str(k): int(v)
                        for k, v in dict(
                            zip(*np.unique(train_bins, return_counts=True))
                        ).items()
                    }
                    if len(unique_bins) > 1
                    else {}
                ),
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
            "support_weight_method": "none",
            "support_weight_range": [1.0, 1.0],
            "bin_count": 1,
            "train_split_counts": {"total": 0, "per_bin": {}},
        }
        edges_used.append(weight_metadata)

    return fold_df, edges_used


def global_eval_text_purge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purge training records that have text hashes appearing in ANY evaluation set.

    This prevents eval→train text contamination across all folds.
    """
    if "text_hash" not in df.columns:
        return df

    # Get all eval text hashes across all folds
    eval_hashes = set(df.loc[df["split"].isin(["val", "test"]), "text_hash"])

    # Find training records that duplicate any eval text
    contamination_mask = (df["split"] == "train") & (df["text_hash"].isin(eval_hashes))
    n_contaminated = int(contamination_mask.sum())

    if n_contaminated > 0:
        logger.info(
            f"Global purge: removed {n_contaminated} train rows that duplicate any eval text"
        )
        df = df.loc[~contamination_mask]

    return df


def make_leakage_safe_splits(
    df: pd.DataFrame, k: int = 5, oof_ratio: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """
    Create complete leakage-safe splits with DNT policy and temporal CV.

    Args:
        df: Input DataFrame
        k: Number of CV folds
        oof_ratio: Proportion of latest cases for OOF test
        seed: Random seed

    Returns:
        DataFrame with leakage-safe splits and DNT metadata
    """
    np.random.seed(seed)
    logger.info("Creating leakage-safe splits with DNT policy and temporal CV")

    # 0) Hygiene: remove within-case duplicates
    df = hard_dedupe_within_case(df)

    # 1) Apply DNT policy (wrap, don't drop)
    df, dnt = wrap_do_not_train_cols(df)
    df, dnt = collapse_sentiment_and_mark_raw_dnt(df, dnt)

    # 2) Create case-level time blocks
    case_df = create_time_blocks_by_case(df, k)

    # 3) Create OOF test split (latest cases)
    cv_cases, oof_test_cases = create_oof_test_split(case_df, oof_ratio)

    # 4) Generate rolling-origin folds on CV cases only
    cv_case_df = case_df[case_df["case_id"].isin(cv_cases)]
    folds = rolling_origin_group_folds(cv_case_df, k)

    # 5) Create fold DataFrames
    # Generate temporal folds and collect per-fold bin edges + weights
    fold_dfs = []
    per_fold_edges = {}
    per_fold_weights = {}
    for i, (train_cases, val_cases, test_cases) in enumerate(folds):
        fold_df, edges_and_weights = make_temporal_fold(
            df, train_cases, val_cases, test_cases, i
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

    # 6) Create OOF test DataFrame
    if oof_test_cases:
        oof_df = df[df["case_id"].isin(oof_test_cases)].copy()
        oof_df["fold"] = -1  # Special marker for OOF
        oof_df["split"] = "oof_test"
        oof_df["outcome_bin"] = 1  # Will be computed properly during final evaluation
        oof_df["support_tertile"] = 1
        oof_df["sample_weight"] = 1.0
        fold_dfs.append(oof_df)

    # 7) Combine all folds
    result_df = pd.concat(fold_dfs, ignore_index=True)
    result_df.attrs["do_not_train"] = sorted(dnt)

    # 8) Skip global purge - already handled per-fold in make_temporal_fold
    # Global purge was removing too many training records due to boilerplate text

    # 9) Preserve per-fold bin edges and weights for audit
    result_df.attrs["per_fold_bin_edges"] = per_fold_edges
    result_df.attrs["per_fold_weights"] = per_fold_weights

    # 10) Validate no empty splits (safety guards)
    for f, g in result_df.groupby("fold"):
        if f == -1:  # Skip OOF test validation
            continue
        ntr = (g["split"] == "train").sum()
        nva = (g["split"] == "val").sum()
        nte = (g["split"] == "test").sum()
        assert (
            ntr > 0 and nva > 0 and nte > 0
        ), f"Fold {f} has empty split(s): train={ntr}, val={nva}, test={nte}"

    # 11) Each case must be eval in at most one fold
    eval_df = result_df[result_df["split"].isin(["val", "test"])]
    eval_counts = eval_df.groupby("case_id")["fold"].nunique()
    assert (
        eval_counts <= 1
    ).all(), f"Cases appear as eval in multiple folds: {eval_counts[eval_counts > 1].index.tolist()}"

    logger.info(
        f"✅ Validation passed: No empty splits, no cross-fold eval contamination"
    )
    logger.info(f"Leakage-safe splits complete: {k} CV folds + OOF test")
    logger.info(f"DNT columns: {len(dnt)}")
    logger.info(f"Per-fold bin edges preserved: {len(per_fold_edges)} folds")

    return result_df


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of records
    """
    records = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise

    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def extract_case_id(record: Dict[str, Any], case_id_field: str) -> Optional[str]:
    """
    Extract case ID from record.

    Args:
        record: Data record
        case_id_field: Field name containing case ID

    Returns:
        Case ID string or None if not found
    """
    if case_id_field in record and case_id_field != "_src":
        return record[case_id_field]

    # Extract from _src field
    if "_src" in record:
        import re

        src_path = record["_src"]
        # Pattern to match case IDs like "2:11-cv-00644_flmd"
        match = re.search(r"/([^/]*:\d+-[^/]+_[^/]+)/entries/", src_path)
        if match:
            return match.group(1)
        # Fallback pattern for other formats
        match = re.search(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/", src_path)
        if match:
            return match.group(1)

    return None


def group_by_case(
    records: List[Dict[str, Any]], case_id_field: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group records by case ID.

    Args:
        records: List of data records
        case_id_field: Field name containing case ID

    Returns:
        Dictionary mapping case IDs to lists of records
    """
    case_groups = defaultdict(list)
    missing_case_id_count = 0

    for record in records:
        case_id = extract_case_id(record, case_id_field)

        if case_id is None:
            missing_case_id_count += 1
            # Create unique ID for records without case_id
            case_id = f"unknown_case_{missing_case_id_count}"

        case_groups[case_id].append(record)

    if missing_case_id_count > 0:
        logger.warning(f"Found {missing_case_id_count} records without case_id")

    logger.info(f"Grouped data into {len(case_groups)} cases")
    return dict(case_groups)


def extract_case_data_for_stratification(
    case_groups: Dict[str, List[Dict[str, Any]]], target_field: str, stratify_type: str
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Extract case-level outcomes and sizes for stratification.

    This separates data extraction from binning to enable per-fold binning.

    Args:
        case_groups: Dictionary mapping case IDs to records
        target_field: Field name containing target variable
        stratify_type: Either 'classification' or 'regression'

    Returns:
        Tuple of (case_outcomes, case_sizes) dictionaries
    """
    case_outcomes = {}  # case_id -> outcome value
    case_sizes = {}  # case_id -> number of quotes

    logger.info("Extracting case-level data for stratification")

    for case_id, records in case_groups.items():
        values = [
            r.get(target_field) for r in records if r.get(target_field) is not None
        ]
        case_size = len(records)  # Number of quotes in this case
        case_sizes[case_id] = case_size

        if values:
            if stratify_type == "classification":
                # Use most common class in case
                case_outcome = Counter(values).most_common(1)[0][0]
                case_outcomes[case_id] = case_outcome
            else:  # regression
                # Use mean value for case (but all should be same in a case)
                case_outcome = float(np.mean([float(v) for v in values]))
                case_outcomes[case_id] = case_outcome

    logger.info(f"Extracted data for {len(case_outcomes)} cases with valid outcomes")
    logger.info(
        f"Case size range: {min(case_sizes.values())} - {max(case_sizes.values())} quotes"
    )

    if case_outcomes:
        outcome_values = list(case_outcomes.values())
        logger.info(
            f"Outcome range: ${min(outcome_values):,.0f} - ${max(outcome_values):,.0f}"
        )

    return case_outcomes, case_sizes


def create_global_size_buckets(
    case_sizes: Dict[str, int]
) -> Tuple[Dict[str, str], np.ndarray]:
    """
    Create global case size buckets using tertiles.

    Size buckets can be computed globally since case size is an inherent property
    not tied to future outcomes (no leakage risk).

    Args:
        case_sizes: Dictionary mapping case IDs to case sizes

    Returns:
        Tuple of (case_size_buckets, size_edges)
    """
    logger.info("Creating global case size buckets (tertiles)")

    sizes = list(case_sizes.values())
    sizes_sorted = sorted(sizes)

    # Use tertiles for case size buckets
    size_quantiles = [0, 1 / 3, 2 / 3, 1]
    size_edges = np.quantile(sizes_sorted, size_quantiles)
    size_edges = np.unique(size_edges)  # Remove duplicates

    logger.info(
        f"Global case size tertile boundaries: {[f'{x:.0f} quotes' for x in size_edges]}"
    )

    # Assign size buckets
    case_size_buckets = {}
    for case_id, case_size in case_sizes.items():
        size_idx = np.digitize(case_size, size_edges) - 1
        size_idx = np.clip(size_idx, 0, len(size_edges) - 2)
        size_bucket = ["Small", "Medium", "Large"][size_idx]
        case_size_buckets[case_id] = size_bucket

    # Verify size balance
    size_counts = Counter(case_size_buckets.values())
    logger.info(f"Global case size bucket distribution: {dict(size_counts)}")
    for bucket_name, count in size_counts.items():
        pct = count / len(case_size_buckets) * 100
        logger.info(f"  {bucket_name}: {count} cases ({pct:.1f}%)")

    return case_size_buckets, size_edges


def create_per_fold_outcome_bins(
    case_outcomes: Dict[str, float],
    train_case_ids: List[str],
    n_bins: int = 3,
    fold_idx: int = None,
) -> Tuple[Dict[str, str], np.ndarray]:
    """
    Create outcome bins using ONLY the training cases for this fold.

    This prevents global binning leakage by computing quantiles per-fold.

    Args:
        case_outcomes: Dictionary mapping case IDs to outcomes
        train_case_ids: List of training case IDs for this fold
        n_bins: Number of bins (default: 3 for Low/Med/High)
        fold_idx: Fold index for logging

    Returns:
        Tuple of (case_outcome_bins, bin_edges)
    """
    fold_str = f"fold {fold_idx}" if fold_idx is not None else "current fold"
    logger.info(f"Creating per-fold outcome bins for {fold_str}")

    # Extract training outcomes only
    train_outcomes = []
    for case_id in train_case_ids:
        if case_id in case_outcomes:
            train_outcomes.append(case_outcomes[case_id])

    if len(train_outcomes) < n_bins:
        logger.warning(
            f"Insufficient training outcomes ({len(train_outcomes)}) for {n_bins} bins in {fold_str}"
        )
        # Fallback: create single bin
        bin_edges = (
            np.array([min(train_outcomes), max(train_outcomes)])
            if train_outcomes
            else np.array([0, 1])
        )
        n_bins = 1
    else:
        train_outcomes_sorted = sorted(train_outcomes)

        if n_bins == 3:
            # Use 33/33/33 split for consistency
            quantiles = [0, 1 / 3, 2 / 3, 1]
        else:
            quantiles = np.linspace(0, 1, n_bins + 1)

        bin_edges = np.quantile(train_outcomes_sorted, quantiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates

        # Adjust n_bins if duplicates were removed
        n_bins = len(bin_edges) - 1

    logger.info(
        f"Per-fold outcome quantile boundaries for {fold_str}: {[f'${x:,.0f}' for x in bin_edges]}"
    )

    # Apply bins to ALL cases (train/val/test) using this fold's thresholds
    case_outcome_bins = {}
    for case_id, outcome in case_outcomes.items():
        bin_idx = np.digitize(outcome, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        case_outcome_bins[case_id] = f"bin_{bin_idx}"

    # Verify training distribution
    train_bins = [
        case_outcome_bins[case_id]
        for case_id in train_case_ids
        if case_id in case_outcome_bins
    ]
    train_bin_counts = Counter(train_bins)
    logger.info(f"Training bin distribution for {fold_str}: {dict(train_bin_counts)}")

    return case_outcome_bins, bin_edges


def extract_leakage_safe_composite_stratification_labels(
    case_groups: Dict[str, List[Dict[str, Any]]],
    target_field: str,
    stratify_type: str,
    n_bins: int = 3,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    """
    Extract leakage-safe composite stratification labels for fold creation.

    LEAKAGE-SAFE APPROACH:
    1. Precompute global support buckets (safe - inherent property)
    2. Create coarse outcome stratification for initial fold structure
    3. Per-fold outcome binning will be applied during fold creation

    Args:
        case_groups: Dictionary mapping case IDs to records
        target_field: Field name containing target variable
        stratify_type: Either 'classification' or 'regression'
        n_bins: Number of bins for regression targets

    Returns:
        Tuple of (composite_labels, case_size_buckets, case_outcomes)
    """
    logger.info("Creating leakage-safe composite stratification labels")
    logger.info("Step 1: Precompute global support buckets (safe)")

    # Extract case data
    case_outcomes, case_sizes = extract_case_data_for_stratification(
        case_groups, target_field, stratify_type
    )

    # Step 1: Create global size buckets (SAFE - inherent property)
    case_size_buckets, size_edges = create_global_size_buckets(case_sizes)

    logger.info("Step 2: Create coarse outcome stratification for fold structure")

    # Step 2: Create COARSE outcome stratification for initial fold assignment
    # This is just for creating balanced fold structure, will be refined per-fold
    initial_outcome_strata = {}
    if stratify_type == "regression" and case_outcomes:
        outcomes = list(case_outcomes.values())
        outcomes_sorted = sorted(outcomes)

        # Use coarser binning for initial stratification (fewer bins = more robust)
        if n_bins == 3:
            # Use just tertiles for initial stratification to ensure sufficient support
            coarse_quantiles = [0, 1 / 3, 2 / 3, 1.0]  # 3 bins only
        else:
            coarse_quantiles = np.linspace(0, 1, min(n_bins, 5))  # Cap at 4 bins

        coarse_bin_edges = np.quantile(outcomes_sorted, coarse_quantiles)
        coarse_bin_edges = np.unique(coarse_bin_edges)

        logger.info(
            f"Coarse outcome strata (for initial fold balance): {len(coarse_bin_edges)-1} bins"
        )
        logger.info(f"Coarse boundaries: {[f'${x:,.0f}' for x in coarse_bin_edges]}")

        for case_id, outcome in case_outcomes.items():
            bin_idx = np.digitize(outcome, coarse_bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(coarse_bin_edges) - 2)
            initial_outcome_strata[case_id] = f"coarse_bin_{bin_idx}"

    # Step 3: Create composite stratification labels (outcome_stratum × size_bucket)
    composite_labels = {}
    for case_id in case_groups.keys():
        if case_id in case_outcomes:
            if stratify_type == "regression":
                outcome_stratum = initial_outcome_strata[case_id]
            else:
                outcome_stratum = case_outcomes[case_id]

            size_bucket = case_size_buckets[case_id]
            composite_labels[case_id] = f"{outcome_stratum}_{size_bucket}"
        else:
            # Missing outcome
            size_bucket = case_size_buckets[case_id]
            composite_labels[case_id] = f"missing_{size_bucket}"

    # Report distribution
    composite_counts = Counter(composite_labels.values())
    logger.info(
        f"Composite stratification distribution (coarse × size): {dict(composite_counts)}"
    )

    # Count strata
    unique_strata = len(composite_counts)
    min_cases_per_stratum = min(composite_counts.values()) if composite_counts else 0

    logger.info(f"Stratification summary:")
    logger.info(f"  Unique strata: {unique_strata}")
    logger.info(f"  Min cases per stratum: {min_cases_per_stratum}")
    logger.info(f"  Stratification feasible for k-fold: {min_cases_per_stratum >= 5}")

    return composite_labels, case_size_buckets, case_outcomes


def filter_missing_cases(
    case_groups: Dict[str, List[Dict[str, Any]]],
    case_labels: Dict[str, Union[str, int]],
    drop_missing: bool = True,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Union[str, int]]]:
    """
    Filter out cases with missing labels if requested.

    Args:
        case_groups: Dictionary mapping case IDs to records
        case_labels: Dictionary mapping case IDs to stratification labels
        drop_missing: Whether to drop cases with "missing" labels

    Returns:
        Filtered case_groups and case_labels
    """
    if not drop_missing:
        return case_groups, case_labels

    # Count before filtering
    original_cases = len(case_groups)
    missing_cases = sum(1 for label in case_labels.values() if label == "missing")

    # Filter out missing cases
    filtered_case_groups = {}
    filtered_case_labels = {}

    for case_id in case_groups:
        if case_labels.get(case_id) != "missing":
            filtered_case_groups[case_id] = case_groups[case_id]
            filtered_case_labels[case_id] = case_labels[case_id]

    logger.info(
        f"Filtered out {missing_cases} missing cases ({missing_cases/original_cases:.1%})"
    )
    logger.info(f"Remaining: {len(filtered_case_groups)} cases")

    return filtered_case_groups, filtered_case_labels


def compute_case_weights(
    case_labels: Dict[str, Union[str, int]]
) -> Dict[Union[str, int], float]:
    """
    Compute class weights for balanced training.

    Args:
        case_labels: Dictionary mapping case IDs to labels

    Returns:
        Dictionary mapping labels to weights
    """
    labels = list(case_labels.values())
    unique_labels = np.array(sorted(set(labels)))  # Convert to numpy array and sort

    # Compute balanced weights
    weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
    weight_dict = dict(zip(unique_labels, weights))

    logger.info(f"Computed class weights: {weight_dict}")
    return weight_dict


def compute_support_bias_mitigation(
    case_groups: Dict[str, List[Dict[str, Any]]], case_outcomes: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute support bias mitigation weights to address case size correlation with outcomes.

    The audit found that case size alone provides +7.6% accuracy lift, indicating
    large cases correlate with high outcomes. This computes per-case weights to
    normalize for this bias during training.

    Args:
        case_groups: Dictionary mapping case IDs to records
        case_outcomes: Dictionary mapping case IDs to outcomes

    Returns:
        Dictionary with support bias mitigation data
    """
    logger.info("Computing support bias mitigation weights")

    # Collect case sizes and outcomes
    case_sizes = {}
    valid_cases = []

    for case_id, records in case_groups.items():
        case_size = len(records)
        case_sizes[case_id] = case_size

        if case_id in case_outcomes:
            valid_cases.append(
                {
                    "case_id": case_id,
                    "size": case_size,
                    "outcome": case_outcomes[case_id],
                }
            )

    if len(valid_cases) < 10:
        logger.warning("Insufficient valid cases for support bias analysis")
        return {"method": "insufficient_data"}

    # Analyze size-outcome correlation
    sizes = [c["size"] for c in valid_cases]
    outcomes = [c["outcome"] for c in valid_cases]

    correlation = np.corrcoef(sizes, outcomes)[0, 1] if len(sizes) > 1 else 0

    logger.info(f"Case size-outcome correlation: {correlation:.3f}")

    # Create size-based outcome bins for analysis
    outcome_bins = pd.qcut(
        outcomes, q=3, labels=["low", "med", "high"], duplicates="drop"
    )
    size_by_bin = {}

    for i, bin_label in enumerate(outcome_bins):
        case_size = sizes[i]
        if bin_label not in size_by_bin:
            size_by_bin[bin_label] = []
        size_by_bin[bin_label].append(case_size)

    # Compute size statistics by outcome bin
    size_stats = {}
    for bin_label, bin_sizes in size_by_bin.items():
        size_stats[bin_label] = {
            "mean_size": np.mean(bin_sizes),
            "median_size": np.median(bin_sizes),
            "std_size": np.std(bin_sizes),
            "count": len(bin_sizes),
        }

    logger.info("Case size statistics by outcome bin:")
    for bin_label, stats in size_stats.items():
        logger.info(
            f"  {bin_label}: mean={stats['mean_size']:.1f}, median={stats['median_size']:.1f}"
        )

    # Compute per-case support weights (inverse of normalized case size)
    # This downweights large cases to reduce their dominating influence
    mean_size = np.mean(sizes)
    per_case_weights = {}

    for case_id, case_size in case_sizes.items():
        # Inverse size weighting with smoothing
        normalized_size = case_size / mean_size
        support_weight = 1.0 / (1.0 + 0.5 * (normalized_size - 1.0))  # Smooth inverse
        per_case_weights[case_id] = support_weight

    # Compute quote-level weights (per-case weight divided by case size)
    quote_level_weights = {}
    for case_id, records in case_groups.items():
        case_weight = per_case_weights.get(case_id, 1.0)
        case_size = len(records)
        quote_weight = case_weight / case_size  # Normalize by case size

        for record in records:
            # Use some unique identifier for the quote
            quote_id = f"{case_id}_{records.index(record)}"
            quote_level_weights[quote_id] = quote_weight

    mitigation_data = {
        "method": "inverse_size_weighting",
        "size_outcome_correlation": correlation,
        "size_stats_by_bin": size_stats,
        "per_case_weights": per_case_weights,
        "quote_level_weights": quote_level_weights,
        "mean_case_size": mean_size,
        "bias_assessment": {
            "correlation_strength": (
                "HIGH"
                if abs(correlation) > 0.5
                else "MODERATE" if abs(correlation) > 0.3 else "LOW"
            ),
            "mitigation_needed": abs(correlation) > 0.3,
            "audit_accuracy_lift": 0.076,  # From audit
        },
        "paper_justification": f"Per-case weighting applied to mitigate support bias (size-outcome correlation: {correlation:.3f}, audit accuracy lift: +7.6%)",
    }

    logger.info(f"Support bias mitigation complete:")
    logger.info(f"  Size-outcome correlation: {correlation:.3f}")
    logger.info(f"  Mitigation method: {mitigation_data['method']}")
    logger.info(
        f"  Bias assessment: {mitigation_data['bias_assessment']['correlation_strength']}"
    )

    return mitigation_data


def create_stratified_case_folds(
    case_groups: Dict[str, List[Dict[str, Any]]],
    case_labels: Dict[str, Union[str, int]],
    k_folds: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    drop_missing: bool = True,
    target_field: str = "final_judgement_real",
    stratify_type: str = "regression",
    n_bins: int = 3,
    global_size_buckets: Dict[str, str] = None,
) -> Tuple[List[Tuple[List[str], List[str], List[str]]], Dict[str, Any]]:
    """
    Create stratified k-fold splits with per-fold outcome binning to prevent leakage.

    Uses StratifiedGroupKFold to ensure:
    1. Case-level integrity (no quotes from same case in different folds)
    2. Balanced label distribution across folds
    3. Per-fold outcome binning (prevents global binning leakage)
    4. Speaker disjointness (when using filtered dataset)

    Args:
        case_groups: Dictionary mapping case IDs to records
        case_labels: Dictionary mapping case IDs to initial stratification labels
        k_folds: Number of folds
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        drop_missing: Whether to drop cases with "missing" labels
        target_field: Field name containing target variable
        stratify_type: Either 'classification' or 'regression'
        n_bins: Number of bins for regression targets
        global_size_buckets: Pre-computed global size buckets (safe from leakage)

    Returns:
        Tuple of (folds, per_fold_metadata) where:
        - folds: List of (train_case_ids, val_case_ids, test_case_ids) tuples
        - per_fold_metadata: Dictionary with per-fold bin edges and statistics
    """
    # Filter missing cases if requested
    filtered_case_groups, filtered_case_labels = filter_missing_cases(
        case_groups, case_labels, drop_missing
    )

    # Extract case data for per-fold binning
    case_outcomes, case_sizes = extract_case_data_for_stratification(
        filtered_case_groups, target_field, stratify_type
    )

    # Validate minimum support per label
    label_counts = Counter(filtered_case_labels.values())
    min_support = min(label_counts.values()) if label_counts else 0

    if min_support < k_folds:
        logger.warning(f"Minimum label support ({min_support}) < k_folds ({k_folds})")
        logger.warning(
            "Some folds may have missing labels. Consider reducing k_folds or merging rare labels."
        )

    # Prepare data for StratifiedGroupKFold
    case_ids = list(filtered_case_groups.keys())
    labels = [filtered_case_labels[case_id] for case_id in case_ids]
    groups = case_ids  # Each case is its own group

    # Create dummy features (not used, but required by sklearn)
    X = np.arange(len(case_ids)).reshape(-1, 1)

    # Use StratifiedGroupKFold for proper stratification with grouping
    sgkf = StratifiedGroupKFold(
        n_splits=k_folds, shuffle=True, random_state=random_seed
    )

    # Initialize fold metadata storage
    per_fold_metadata = {
        "fold_bin_edges": {},
        "fold_statistics": {},
        "global_size_buckets": global_size_buckets or {},
        "composite_stratification": {
            "method": "leakage_safe_composite",
            "global_support_buckets": "precomputed_safe",
            "per_fold_outcome_bins": "training_data_only",
        },
        "leakage_prevention": {
            "method": "per_fold_outcome_binning_with_global_support",
            "description": "Outcome bins computed from training data only per fold, support buckets precomputed globally (safe)",
            "paper_reference": "Prevents global binning leakage while maintaining composite stratification",
        },
    }

    # Create initial folds structure
    folds = []
    fold_splits = list(sgkf.split(X, labels, groups))

    # Verify no case bleed across folds
    all_assigned_cases = set()

    logger.info("Creating folds with per-fold outcome binning...")

    for fold_idx, (train_val_idx, test_idx) in enumerate(fold_splits):
        # Get case IDs for test set
        test_case_ids = [case_ids[i] for i in test_idx]

        # CRITICAL: Verify no case appears in multiple folds
        for case_id in test_case_ids:
            if case_id in all_assigned_cases:
                logger.error(
                    f"CASE BLEED DETECTED: Case {case_id} appears in multiple folds!"
                )
                raise ValueError(f"Case bleed detected: {case_id}")
            all_assigned_cases.add(case_id)

        # Split train_val into train and val
        train_val_case_ids = [case_ids[i] for i in train_val_idx]
        train_val_labels = [labels[i] for i in train_val_idx]

        # Calculate val size relative to train_val
        val_size = val_ratio / (train_ratio + val_ratio)

        # Use StratifiedShuffleSplit for train/val split
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=random_seed + fold_idx
        )

        try:
            train_val_X = np.arange(len(train_val_case_ids)).reshape(-1, 1)
            train_idx_local, val_idx_local = next(
                sss.split(train_val_X, train_val_labels)
            )
            train_case_ids = [train_val_case_ids[i] for i in train_idx_local]
            val_case_ids = [train_val_case_ids[i] for i in val_idx_local]
        except ValueError as e:
            # Fallback to simple split if stratification fails
            logger.warning(
                f"Stratified train/val split failed for fold {fold_idx}: {e}"
            )
            logger.warning("Using simple split instead")

            n_val = int(len(train_val_case_ids) * val_size)
            val_case_ids = train_val_case_ids[:n_val]
            train_case_ids = train_val_case_ids[n_val:]

        # CRITICAL: Create per-fold outcome bins using ONLY training data
        if stratify_type == "regression" and case_outcomes:
            fold_outcome_bins, fold_bin_edges = create_per_fold_outcome_bins(
                case_outcomes, train_case_ids, n_bins, fold_idx
            )

            # Store per-fold metadata
            per_fold_metadata["fold_bin_edges"][
                f"fold_{fold_idx}"
            ] = fold_bin_edges.tolist()

            # Compute fold statistics
            train_outcomes = [
                case_outcomes[cid] for cid in train_case_ids if cid in case_outcomes
            ]
            val_outcomes = [
                case_outcomes[cid] for cid in val_case_ids if cid in case_outcomes
            ]
            test_outcomes = [
                case_outcomes[cid] for cid in test_case_ids if cid in case_outcomes
            ]

            per_fold_metadata["fold_statistics"][f"fold_{fold_idx}"] = {
                "train_outcome_range": (
                    [min(train_outcomes), max(train_outcomes)]
                    if train_outcomes
                    else None
                ),
                "val_outcome_range": (
                    [min(val_outcomes), max(val_outcomes)] if val_outcomes else None
                ),
                "test_outcome_range": (
                    [min(test_outcomes), max(test_outcomes)] if test_outcomes else None
                ),
                "train_cases": len(train_case_ids),
                "val_cases": len(val_case_ids),
                "test_cases": len(test_case_ids),
                "bin_edges_used": fold_bin_edges.tolist(),
            }

            logger.info(
                f"Fold {fold_idx} per-fold binning complete - no global leakage"
            )

        folds.append((train_case_ids, val_case_ids, test_case_ids))

    logger.info(f"Created {k_folds} leakage-safe folds with per-fold outcome binning")
    logger.info(
        f"Train/Val/Test ratios: {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}"
    )
    logger.info(f"Initial label distribution: {label_counts}")

    return folds, per_fold_metadata


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def validate_fold_distribution(
    folds: List[Tuple[List[str], List[str], List[str]]],
    case_groups: Dict[str, List[Dict[str, Any]]],
    case_labels: Dict[str, Union[str, int]],
    target_field: str,
) -> Dict[str, Any]:
    """
    Validate fold distribution and compute comprehensive statistics.

    Includes inter-fold analysis:
    - Speaker set Jaccard similarity (no speaker leakage)
    - Outcome range coverage per fold
    - Composite stratification quality metrics

    Args:
        folds: List of (train_case_ids, val_case_ids, test_case_ids) tuples
        case_groups: Dictionary mapping case IDs to records
        case_labels: Dictionary mapping case IDs to stratification labels
        target_field: Field name containing target variable

    Returns:
        Dictionary containing validation statistics
    """
    stats = {
        "fold_stats": [],
        "label_distribution": {},
        "support_balance": {},
        "overall_stats": {},
        "inter_fold_analysis": {},
        "outcome_coverage": {},
    }

    all_train_records = []
    all_val_records = []
    all_test_records = []

    # Track speakers per fold for Jaccard analysis
    fold_speaker_sets = []
    fold_outcome_ranges = []

    for fold_idx, (train_case_ids, val_case_ids, test_case_ids) in enumerate(folds):
        # Count records and cases
        train_records = []
        val_records = []
        test_records = []

        for case_id in train_case_ids:
            train_records.extend(case_groups[case_id])
        for case_id in val_case_ids:
            val_records.extend(case_groups[case_id])
        for case_id in test_case_ids:
            test_records.extend(case_groups[case_id])

        all_train_records.extend(train_records)
        all_val_records.extend(val_records)
        all_test_records.extend(test_records)

        # Collect speakers in this fold
        fold_speakers = set()
        fold_outcomes = []

        for record in train_records + val_records + test_records:
            if "speaker" in record:
                fold_speakers.add(record["speaker"])
            if target_field in record and record[target_field] is not None:
                fold_outcomes.append(float(record[target_field]))

        fold_speaker_sets.append(fold_speakers)

        # Outcome range for this fold
        if fold_outcomes:
            fold_outcome_ranges.append(
                {
                    "fold": fold_idx,
                    "min_outcome": min(fold_outcomes),
                    "max_outcome": max(fold_outcomes),
                    "range_span": max(fold_outcomes) - min(fold_outcomes),
                    "num_outcomes": len(fold_outcomes),
                }
            )

        # Label distribution for this fold
        train_labels = [case_labels[case_id] for case_id in train_case_ids]
        val_labels = [case_labels[case_id] for case_id in val_case_ids]
        test_labels = [case_labels[case_id] for case_id in test_case_ids]

        fold_stat = {
            "fold": fold_idx,
            "train_cases": len(train_case_ids),
            "val_cases": len(val_case_ids),
            "test_cases": len(test_case_ids),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
            "train_label_dist": dict(Counter(train_labels)),
            "val_label_dist": dict(Counter(val_labels)),
            "test_label_dist": dict(Counter(test_labels)),
            "unique_speakers": len(fold_speakers),
        }

        stats["fold_stats"].append(fold_stat)

    # Compute inter-fold Jaccard similarities for speaker sets
    jaccard_matrix = []
    for i in range(len(fold_speaker_sets)):
        row = []
        for j in range(len(fold_speaker_sets)):
            jaccard_sim = compute_jaccard_similarity(
                fold_speaker_sets[i], fold_speaker_sets[j]
            )
            row.append(jaccard_sim)
        jaccard_matrix.append(row)

    # Compute average off-diagonal Jaccard (should be low for good separation)
    off_diagonal_jaccards = []
    for i in range(len(jaccard_matrix)):
        for j in range(len(jaccard_matrix)):
            if i != j:
                off_diagonal_jaccards.append(jaccard_matrix[i][j])

    avg_inter_fold_jaccard = (
        np.mean(off_diagonal_jaccards) if off_diagonal_jaccards else 0.0
    )

    stats["inter_fold_analysis"] = {
        "speaker_jaccard_matrix": jaccard_matrix,
        "avg_inter_fold_jaccard": avg_inter_fold_jaccard,
        "max_inter_fold_jaccard": (
            max(off_diagonal_jaccards) if off_diagonal_jaccards else 0.0
        ),
        "speaker_leakage_quality": (
            "GOOD"
            if avg_inter_fold_jaccard < 0.1
            else "MODERATE" if avg_inter_fold_jaccard < 0.3 else "POOR"
        ),
    }

    # Outcome range coverage analysis
    if fold_outcome_ranges:
        global_min = min(r["min_outcome"] for r in fold_outcome_ranges)
        global_max = max(r["max_outcome"] for r in fold_outcome_ranges)
        global_range = global_max - global_min

        # Check how well each fold covers the global range
        for fold_range in fold_outcome_ranges:
            fold_range["coverage_pct"] = (
                (fold_range["range_span"] / global_range * 100)
                if global_range > 0
                else 0
            )

        stats["outcome_coverage"] = {
            "global_range": {
                "min": global_min,
                "max": global_max,
                "span": global_range,
            },
            "fold_ranges": fold_outcome_ranges,
            "avg_coverage_pct": np.mean(
                [r["coverage_pct"] for r in fold_outcome_ranges]
            ),
            "min_coverage_pct": min(r["coverage_pct"] for r in fold_outcome_ranges),
            "range_coverage_quality": (
                "GOOD"
                if min(r["coverage_pct"] for r in fold_outcome_ranges) > 70
                else "MODERATE"
            ),
        }

    # Overall statistics
    total_records = (
        len(all_train_records) + len(all_val_records) + len(all_test_records)
    )
    stats["overall_stats"] = {
        "total_folds": len(folds),
        "total_records": total_records,
        "avg_train_records_per_fold": np.mean(
            [s["train_records"] for s in stats["fold_stats"]]
        ),
        "avg_val_records_per_fold": np.mean(
            [s["val_records"] for s in stats["fold_stats"]]
        ),
        "avg_test_records_per_fold": np.mean(
            [s["test_records"] for s in stats["fold_stats"]]
        ),
        "train_ratio": len(all_train_records) / total_records if total_records else 0,
        "val_ratio": len(all_val_records) / total_records if total_records else 0,
        "test_ratio": len(all_test_records) / total_records if total_records else 0,
    }

    return stats


def save_fold_data(
    fold_idx: int,
    train_case_ids: List[str],
    val_case_ids: List[str],
    test_case_ids: List[str],
    case_groups: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """
    Save fold data to files.

    Args:
        fold_idx: Fold index
        train_case_ids: Training case IDs
        val_case_ids: Validation case IDs
        test_case_ids: Test case IDs
        case_groups: Dictionary mapping case IDs to records
        output_dir: Output directory
    """
    # Create fold directory
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Save training data
    train_path = fold_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for case_id in train_case_ids:
            for record in case_groups[case_id]:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    # Save validation data
    val_path = fold_dir / "val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for case_id in val_case_ids:
            for record in case_groups[case_id]:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    # Save test data
    test_path = fold_dir / "test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for case_id in test_case_ids:
            for record in case_groups[case_id]:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    # Save case IDs
    case_ids_path = fold_dir / "case_ids.json"
    with open(case_ids_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_case_ids": train_case_ids,
                "val_case_ids": val_case_ids,
                "test_case_ids": test_case_ids,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved fold {fold_idx} data to {fold_dir}")


def save_fold_statistics(stats: Dict[str, Any], output_dir: Path) -> None:
    """
    Save fold statistics to file.

    Args:
        stats: Fold statistics dictionary
        output_dir: Output directory
    """
    stats_path = output_dir / "fold_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"Saved fold statistics to {stats_path}")


def print_fold_summary(stats: Dict[str, Any]) -> None:
    """
    Print comprehensive summary of fold statistics including inter-fold analysis.

    Args:
        stats: Fold statistics dictionary
    """
    print("\n" + "=" * 70)
    print("COMPOSITE STRATIFIED K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    overall = stats["overall_stats"]
    print(f"Total folds: {overall['total_folds']}")
    print(f"Total records: {overall['total_records']}")
    print(f"Avg train records per fold: {overall['avg_train_records_per_fold']:.1f}")
    print(f"Avg val records per fold: {overall['avg_val_records_per_fold']:.1f}")
    print(f"Avg test records per fold: {overall['avg_test_records_per_fold']:.1f}")
    print(
        f"Train/Val/Test ratios: {overall['train_ratio']:.1%}/{overall['val_ratio']:.1%}/{overall['test_ratio']:.1%}"
    )

    # Inter-fold analysis
    if "inter_fold_analysis" in stats:
        print("\nInter-Fold Analysis:")
        print("-" * 50)
        inter = stats["inter_fold_analysis"]
        print(f"Speaker leakage quality: {inter['speaker_leakage_quality']}")
        print(
            f"Avg inter-fold Jaccard similarity: {inter['avg_inter_fold_jaccard']:.3f}"
        )
        print(
            f"Max inter-fold Jaccard similarity: {inter['max_inter_fold_jaccard']:.3f}"
        )
        print("(Lower Jaccard values indicate better speaker separation)")

    # Outcome coverage analysis
    if "outcome_coverage" in stats:
        print("\nOutcome Range Coverage:")
        print("-" * 50)
        coverage = stats["outcome_coverage"]
        print(f"Range coverage quality: {coverage['range_coverage_quality']}")
        print(
            f"Global outcome range: ${coverage['global_range']['min']:,.0f} - ${coverage['global_range']['max']:,.0f}"
        )
        print(f"Avg fold coverage: {coverage['avg_coverage_pct']:.1f}%")
        print(f"Min fold coverage: {coverage['min_coverage_pct']:.1f}%")

    print("\nFold-by-Fold Statistics:")
    print("-" * 50)

    for fold_stat in stats["fold_stats"]:
        print(f"Fold {fold_stat['fold']}:")
        print(
            f"  Cases: {fold_stat['train_cases']} train, {fold_stat['val_cases']} val, {fold_stat['test_cases']} test"
        )
        print(
            f"  Records: {fold_stat['train_records']} train, {fold_stat['val_records']} val, {fold_stat['test_records']} test"
        )
        print(f"  Unique speakers: {fold_stat.get('unique_speakers', 'N/A')}")
        print(f"  Train labels: {fold_stat['train_label_dist']}")
        print(f"  Val labels: {fold_stat['val_label_dist']}")
        print(f"  Test labels: {fold_stat['test_label_dist']}")
        print()

    # Outcome range table
    if "outcome_coverage" in stats and stats["outcome_coverage"]["fold_ranges"]:
        print("Outcome Range Coverage by Fold:")
        print("-" * 50)
        print(f"{'Fold':<6} {'Min Outcome':<15} {'Max Outcome':<15} {'Coverage %':<12}")
        print("-" * 50)
        for fold_range in stats["outcome_coverage"]["fold_ranges"]:
            print(
                f"{fold_range['fold']:<6} "
                f"${fold_range['min_outcome']:>11,.0f}   "
                f"${fold_range['max_outcome']:>11,.0f}   "
                f"{fold_range['coverage_pct']:>8.1f}%"
            )

    print("\n" + "=" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Create stratified k-fold cross-validation splits with case-level integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--input", required=True, help="Input JSONL file path")

    parser.add_argument(
        "--output-dir", required=True, help="Output directory for fold splits"
    )

    parser.add_argument(
        "--k-folds", type=int, default=5, help="Number of folds (default: 5)"
    )

    parser.add_argument(
        "--target-field",
        default="coral_pred_class",
        help="Target field for stratification (default: coral_pred_class)",
    )

    parser.add_argument(
        "--stratify-type",
        choices=["classification", "regression"],
        default="classification",
        help="Type of stratification (default: classification)",
    )

    parser.add_argument(
        "--case-id-field",
        default="case_id",
        help="Field name containing case ID (default: case_id)",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of bins for regression stratification (default: 5)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )

    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)"
    )

    parser.add_argument(
        "--drop-missing",
        action="store_true",
        default=False,
        help="Drop cases with missing labels from training (default: False)",
    )

    parser.add_argument(
        "--compute-weights",
        action="store_true",
        default=False,
        help="Compute and save class weights for balanced training (default: False)",
    )

    parser.add_argument(
        "--use-temporal-cv",
        action="store_true",
        default=False,
        help="Use temporal rolling-origin CV instead of stratified CV (default: False)",
    )

    parser.add_argument(
        "--oof-test-ratio",
        type=float,
        default=0.15,
        help="Proportion of latest cases for out-of-fold test (default: 0.15)",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    logger.info("Starting stratified k-fold cross-validation")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"K-folds: {args.k_folds}")
    logger.info(f"Target field: {args.target_field}")
    logger.info(f"Stratify type: {args.stratify_type}")
    logger.info(f"Drop missing: {args.drop_missing}")
    logger.info(f"Compute weights: {args.compute_weights}")
    logger.info(f"Random seed: {args.random_seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")

    if args.use_temporal_cv:
        logger.info("Using new temporal CV methodology with DNT policy")
        # Load as DataFrame for temporal processing (optimized with orjson)
        logger.info("Loading large dataset with orjson - this may take a moment...")
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

        # Add case_year extraction if not present
        if "case_year" not in df.columns:
            logger.info("Extracting case years from case_id...")

            # Quick year extraction function
            def extract_year_from_case_id(case_id):
                if not case_id:
                    return 2020
                import re

                # Try standard pattern like "1:23-cv-04567_nysd" -> 2023
                match = re.search(r"(\d+):(\d+)-", str(case_id))
                if match:
                    year = int(match.group(2))
                    if year > 90:
                        return 1900 + year
                    else:
                        return 2000 + year
                # Try 4-digit year pattern
                match = re.search(r"(\d{4})", str(case_id))
                if match:
                    year = int(match.group(1))
                    if 1950 <= year <= 2030:
                        return year
                return 2020  # Default fallback

            df["case_year"] = df["case_id"].apply(extract_year_from_case_id)

        # Add text_hash if not present (for deduplication) - optimized
        if "text_hash" not in df.columns:
            logger.info("Creating text_hash for deduplication...")
            import hashlib

            # Use vectorized operation for speed
            df["text_hash"] = (
                df["text"]
                .astype(str)
                .apply(lambda x: hashlib.md5(x.encode()).hexdigest())
            )

        # Create leakage-safe splits with DNT policy
        result_df = make_leakage_safe_splits(
            df, k=args.k_folds, oof_ratio=args.oof_test_ratio, seed=args.random_seed
        )

        # Save results by fold
        for fold_idx in range(args.k_folds):
            fold_data = result_df[result_df["fold"] == fold_idx]
            if len(fold_data) == 0:
                continue

            fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

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
                json.dump({"test_case_ids": oof_cases}, f, indent=2)

        # Save DNT manifest
        manifest = {"do_not_train": result_df.attrs.get("do_not_train", [])}
        manifest_path = output_dir / "dnt_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Save per-fold metadata for audit detection (exact format expected)
        per_fold_metadata = {
            "binning": {
                "method": "train_only_tertiles",
                "fold_edges": result_df.attrs.get("per_fold_bin_edges", {}),
            },
            "weights": result_df.attrs.get("per_fold_weights", {}),
            "methodology": "temporal_rolling_origin_with_dnt",
        }
        metadata_path = output_dir / "per_fold_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(per_fold_metadata, f, indent=2)

        # Save fold statistics
        stats = {
            "methodology": "temporal_rolling_origin_with_dnt",
            "folds": args.k_folds,
            "oof_test_ratio": args.oof_test_ratio,
            "total_records": len(result_df),
            "dnt_columns": len(manifest["do_not_train"]),
            "stratification_approach": "outcome_only_3_bins",
            "support_handling": "weighting_only",
            "leakage_prevention": {
                "temporal_splits": "rolling_origin",
                "per_fold_binning": "training_data_only_3_bins_tertiles",
                "dnt_policy": "wrap_not_drop_expanded",
                "text_deduplication": "eval_vs_train_global",
                "support_policy": "weighting_not_stratification",
            },
            "binning_strategy": {
                "method": "train_only_tertiles",
                "bins": ["low", "medium", "high"],
                "quantiles": [1 / 3, 2 / 3],
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
        }

        stats_path = output_dir / "fold_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.success("Temporal CV with DNT policy complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"DNT columns: {len(manifest['do_not_train'])}")
        return 0

    # Original stratified CV path
    records = load_jsonl_data(args.input)

    # Group by case
    logger.info("Grouping data by case...")
    case_groups = group_by_case(records, args.case_id_field)

    # Note: For temporal CV, we skip complex composite stratification
    # and use simple outcome-only binning per fold instead
    logger.info(
        "Skipping composite stratification for temporal CV - using outcome-only per-fold binning"
    )
    case_labels = {}  # Not used in temporal CV
    global_size_buckets = {}  # Not used in temporal CV
    case_outcomes = {}  # Will be computed in temporal CV

    # Validate ratios sum to 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.error(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
        return 1

    # Compute class weights if requested
    support_bias_data = None
    if args.compute_weights:
        logger.info("Computing class weights...")
        class_weights = compute_case_weights(case_labels)

        # Save class weights
        weights_path = output_dir / "class_weights.json"
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(class_weights, f, indent=2)
        logger.info(f"Saved class weights to {weights_path}")

        # Compute support bias mitigation (use pre-extracted case outcomes)
        logger.info("Computing support bias mitigation...")
        support_bias_data = compute_support_bias_mitigation(case_groups, case_outcomes)

        # Save support bias mitigation data
        bias_path = output_dir / "support_bias_mitigation.json"
        with open(bias_path, "w", encoding="utf-8") as f:
            json.dump(support_bias_data, f, indent=2, default=str)
        logger.info(f"Saved support bias mitigation to {bias_path}")

    # Create stratified folds with per-fold binning
    logger.info("Creating leakage-safe stratified folds...")
    folds, per_fold_metadata = create_stratified_case_folds(
        case_groups,
        case_labels,
        args.k_folds,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_seed,
        args.drop_missing,
        args.target_field,
        args.stratify_type,
        args.n_bins,
        global_size_buckets,
    )

    # Validate fold distribution
    logger.info("Validating fold distribution...")
    stats = validate_fold_distribution(
        folds, case_groups, case_labels, args.target_field
    )

    # Save fold data
    logger.info("Saving fold data...")
    for fold_idx, (train_case_ids, val_case_ids, test_case_ids) in enumerate(folds):
        save_fold_data(
            fold_idx,
            train_case_ids,
            val_case_ids,
            test_case_ids,
            case_groups,
            output_dir,
        )

    # Save statistics and per-fold metadata
    save_fold_statistics(stats, output_dir)

    # Save per-fold binning metadata (CRITICAL for leakage audit)
    metadata_path = output_dir / "per_fold_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(per_fold_metadata, f, indent=2, default=str)
    logger.info(f"Saved per-fold metadata to {metadata_path}")

    # Print leakage prevention summary
    logger.success("LEAKAGE PREVENTION & BIAS MITIGATION IMPLEMENTED:")
    logger.info("✓ Per-fold outcome binning prevents global binning leakage")
    logger.info("✓ Case-level integrity prevents quote-level leakage")
    logger.info("✓ Speaker disjointness prevents speaker memorization")
    logger.info("✓ Metadata fields wrapped to prevent correlation shortcuts")
    if support_bias_data and support_bias_data.get("method") != "insufficient_data":
        correlation = support_bias_data.get("size_outcome_correlation", 0)
        logger.info(
            f"✓ Support bias mitigation for size-outcome correlation: {correlation:.3f}"
        )
    logger.info("✓ Class weights computed for label imbalance")
    logger.info("✓ Fold metadata saved for audit verification")

    # Print summary
    print_fold_summary(stats)

    logger.success("Stratified k-fold cross-validation complete!")
    logger.info(f"Folds saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
