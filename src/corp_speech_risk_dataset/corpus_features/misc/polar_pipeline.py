"""POLAR (Proportional Odds Logistic Regression) pipeline with full CV protocol.

This module implements the complete paper-quality CV and final run protocol
for POLAR models, including:
- Column governance (interpretable features only)
- Per-fold tertile cutpoints
- Alpha-normalized combined weights
- Cumulative isotonic calibration
- Comprehensive evaluation metrics
- Progress reporting
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    f1_score,
    brier_score_loss,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
    RobustScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from loguru import logger
import joblib
from tqdm import tqdm
from datetime import datetime

from .models import POLR, MLR
from .column_governance import (
    validate_columns,
    derive_interpretable,
    pyify,
)

# Constants
EMIT_PREFIX = "polr"
BUCKET_NAMES = ["low", "medium", "high"]


# Pickle-safe transformation functions
def _binarize_transform(X):
    """Binarize transformation: convert values >0 to 1.0, <=0 to 0.0"""
    return (X > 0).astype(float)


def _log1p_transform(X):
    """Log1p transformation: apply log(1+x)"""
    return np.log1p(X)


def _asinh_rate_transform(X):
    """Asinh-rate transformation: expand tiny rates without exploding tails"""
    return np.arcsinh(3.0 * 1_000.0 * X)  # k=3, per-1k tokens


def _winsor99(X):
    """Cap extreme heavy tails per column at 99th percentile"""
    hi = np.nanpercentile(X, 99.0, axis=0)
    return np.minimum(X, hi)


@dataclass
class POLARConfig:
    """Configuration for POLAR pipeline."""

    # Data paths
    kfold_dir: str
    output_dir: str

    # Model parameters
    model_type: str = "polr"  # "polr" or "mlr"
    hyperparameter_grid: Optional[Dict[str, List[Any]]] = None
    scoring_priority: Optional[List[str]] = None

    # Training parameters
    n_inner_cv: int = 3
    calibration_method: str = "isotonic_cumulative"
    calibration_split: float = 0.15

    # Options - All labels and weights are precomputed in authoritative data
    continuous_target_field: str = "final_judgement_real"  # For reference only
    seed: int = 42
    n_jobs: int = -1

    # Temporal DEV policy parameters (tiny-data friendly)
    dev_tail_frac: float = 0.20
    min_dev_cases: int = 3
    min_dev_quotes: int = 150
    require_all_classes: bool = False  # Accept ≥2 classes by default
    embargo_days: int = 90
    safe_qwk: bool = True
    min_cal_n: int = 500  # Minimum samples for direct isotonic
    iso_bins: int = 30  # Quantile bins for small-sample isotonic
    max_categories: int = 50  # Top-K + __OTHER__ threshold
    allow_speaker: bool = False  # Whether to allow speaker features

    def __post_init__(self):
        if self.hyperparameter_grid is None:
            if self.model_type == "mlr":
                # Multinomial LR hyperparameters
                self.hyperparameter_grid = {
                    "C": [0.01, 1, 100],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                    "tol": [1e-4],
                    "class_weight": [
                        "balanced",
                        None,
                    ],  # Test both balanced and unbalanced
                }
            else:
                # POLR hyperparameters (default)
                self.hyperparameter_grid = {
                    "C": [0.01, 1, 100],  # low, mid, high
                    "solver": ["lbfgs"],  # fastest, robust default
                    "max_iter": [200],  # enough for convergence
                    "tol": [1e-4],
                }
            #     "C": [0.001, 0.01, 0.1, 1, 10, 100],
            #     "solver": ["lbfgs", "newton-cg"],
            #     "max_iter": [200, 500],
            #     "tol": [1e-4]
            # }

        if self.scoring_priority is None:
            self.scoring_priority = ["qwk", "macro_f1", "mae", "brier"]


class ProgressReporter:
    """Helper class for consistent progress reporting."""

    def __init__(self, total_folds: int):
        self.total_folds = total_folds
        self.current_fold = 0

    def start_fold(self, fold: int):
        self.current_fold = fold
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold}/{self.total_folds-1} STARTED")
        logger.info(f"{'='*60}")

    def report(self, stage: str, message: str):
        prefix = f"[Fold {self.current_fold}] {stage}"
        logger.info(f"{prefix}: {message}")

    def finish_fold(self, fold: int, metrics: Dict[str, float]):
        logger.info(f"\nFold {fold} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


def load_fold_data(fold_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test data for a fold."""
    train_df = pd.read_json(fold_dir / "train.jsonl", lines=True)
    val_df = pd.read_json(fold_dir / "val.jsonl", lines=True)
    test_df = pd.read_json(fold_dir / "test.jsonl", lines=True)

    logger.info(
        f"Loaded fold data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def create_temporal_dev_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: POLARConfig,
    case_time_field: str = "case_time",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create temporal-safe DEV split from train/val data.

    Args:
        train_df: Training data
        val_df: Validation data (may be added to DEV if temporally safe)
        test_df: Test data (used to determine temporal boundary)
        case_time_field: Field containing case timestamps
        embargo_days: Days to embargo before DEV
        min_dev_cases: Minimum DEV cases required
        min_dev_quotes: Minimum DEV quotes required
        min_classes: Minimum classes required in DEV

    Returns:
        train_core: Training data minus DEV and embargo
        dev: DEV data for hyperparameter tuning and calibration
        metadata: Split metadata for transparency
    """
    # Ensure case_time exists or create it from available data
    all_dfs = [train_df, val_df, test_df]

    for df in all_dfs:
        if case_time_field not in df.columns:
            # Try to use doc_id or case_id as proxy for time ordering
            if "doc_id" in df.columns:
                df[case_time_field] = pd.to_datetime(df["doc_id"], errors="coerce")
            elif "case_id" in df.columns:
                # Use case_id hash as pseudo-time (deterministic but not real time)
                df[case_time_field] = pd.to_datetime(
                    df["case_id"].apply(hash).abs() % 1000000, unit="s", errors="coerce"
                )
            else:
                # Use row index as fallback
                df[case_time_field] = pd.to_datetime(
                    df.index, unit="s", errors="coerce"
                )

    # Get case-level info
    def get_case_info(df):
        case_info = (
            df.groupby("case_id")
            .agg(
                {
                    case_time_field: "first",
                    "quote_id": "count",  # quotes per case
                    "y": "first",  # class per case
                }
            )
            .rename(columns={"quote_id": "n_quotes", "y": "class"})
        )
        return case_info

    train_cases = get_case_info(train_df)
    val_cases = get_case_info(val_df)
    test_cases = get_case_info(test_df)

    # Step 1: Find temporal boundary from test
    test_start = test_cases[case_time_field].min()
    logger.info(f"Test temporal boundary: {test_start}")

    # Step 2: Build candidate pool (before test_start) - ONLY VAL cases strictly < test_start
    train_pool = train_cases[train_cases[case_time_field] < test_start].index.tolist()
    val_pool = val_cases[val_cases[case_time_field] < test_start].index.tolist()

    logger.info(
        f"Temporal pool - Train cases: {len(train_pool)}, Val cases: {len(val_pool)}"
    )
    logger.info(
        f"Temporal safety: Only VAL cases with case_time < {test_start} are eligible for DEV"
    )

    # Combine and sort by time
    all_pool = train_pool + val_pool
    all_cases = pd.concat([train_cases, val_cases])
    all_pool = sorted(all_pool, key=lambda cid: all_cases.loc[cid, case_time_field])

    # Step 3: Find optimal DEV tail using config parameters
    dev_case_ids = []
    dev_metadata = {}

    # Start with config.dev_tail_frac and increase if needed
    frac_search = [config.dev_tail_frac, 0.25, 0.30, 0.35, 0.40, 0.50]
    if config.dev_tail_frac not in frac_search:
        frac_search = [config.dev_tail_frac] + frac_search
        frac_search = sorted(set(frac_search))

    min_classes = 3 if config.require_all_classes else 2
    fallback_quotes = max(
        100, config.min_dev_quotes // 2
    )  # Fallback to 100 or half the target

    for frac in frac_search:
        k = max(config.min_dev_cases, int(len(all_pool) * frac))
        if k >= len(all_pool):
            k = len(all_pool) - 1

        candidate_dev_ids = all_pool[-k:] if k > 0 else []

        if not candidate_dev_ids:
            continue

        # Check requirements
        dev_quotes = all_cases.loc[candidate_dev_ids, "n_quotes"].sum()
        dev_classes = len(set(all_cases.loc[candidate_dev_ids, "class"].dropna()))

        # Primary requirement: meet minima
        meets_primary = (
            len(candidate_dev_ids) >= config.min_dev_cases
            and dev_quotes >= config.min_dev_quotes
            and dev_classes >= min_classes
        )

        # Fallback requirement: lower quotes threshold
        meets_fallback = (
            len(candidate_dev_ids) >= config.min_dev_cases
            and dev_quotes >= fallback_quotes
            and dev_classes >= 2
        )  # Always accept ≥2 classes in fallback

        if meets_primary or meets_fallback:
            dev_case_ids = candidate_dev_ids
            dev_metadata = {
                "frac_used": frac,
                "n_cases": len(dev_case_ids),
                "n_quotes": int(dev_quotes),
                "n_classes": dev_classes,
                "classes": sorted(
                    all_cases.loc[dev_case_ids, "class"].dropna().unique().tolist()
                ),
                "used_fallback": not meets_primary,
                "quotes_threshold": (
                    fallback_quotes if not meets_primary else config.min_dev_quotes
                ),
            }
            if not meets_primary:
                logger.warning(
                    f"Using fallback quotes threshold: {fallback_quotes} (target was {config.min_dev_quotes})"
                )
            break

    # Fallback: take best available
    if not dev_case_ids and all_pool:
        k = max(1, len(all_pool) // 3)  # At least 1/3 for DEV
        dev_case_ids = all_pool[-k:]
        dev_quotes = all_cases.loc[dev_case_ids, "n_quotes"].sum()
        dev_classes = len(set(all_cases.loc[dev_case_ids, "class"].dropna()))

        dev_metadata = {
            "frac_used": k / len(all_pool),
            "n_cases": len(dev_case_ids),
            "n_quotes": int(dev_quotes),
            "n_classes": dev_classes,
            "classes": sorted(
                all_cases.loc[dev_case_ids, "class"].dropna().unique().tolist()
            ),
            "fallback": True,
        }
        logger.warning(
            f"DEV fallback: {len(dev_case_ids)} cases, {dev_quotes} quotes, {dev_classes} classes"
        )

    if not dev_case_ids:
        raise ValueError("Cannot create DEV split - insufficient data")

    # Step 4: Apply embargo (before DEV start)
    remaining_pool = [cid for cid in all_pool if cid not in dev_case_ids]
    dev_start = all_cases.loc[dev_case_ids, case_time_field].min()
    embargo_cutoff = dev_start - pd.Timedelta(days=config.embargo_days)

    train_core_ids = [
        cid
        for cid in remaining_pool
        if all_cases.loc[cid, case_time_field] <= embargo_cutoff
    ]

    # Step 5: Create final DataFrames
    train_core = train_df[train_df["case_id"].isin(train_core_ids)].copy()

    # DEV comes from both train and val pools
    dev_from_train = train_df[
        train_df["case_id"].isin([cid for cid in dev_case_ids if cid in train_pool])
    ]
    dev_from_val = val_df[
        val_df["case_id"].isin([cid for cid in dev_case_ids if cid in val_pool])
    ]
    dev = pd.concat([dev_from_train, dev_from_val], ignore_index=True)

    # Temporal safety checks for logging
    train_core_max_time = (
        all_cases.loc[train_core_ids, case_time_field].max() if train_core_ids else None
    )
    dev_min_time = all_cases.loc[dev_case_ids, case_time_field].min()
    dev_max_time = all_cases.loc[dev_case_ids, case_time_field].max()
    test_min_time = test_start

    # Verify temporal ordering: max(TRAIN_CORE.time) < min(DEV.time) < min(TEST.time)
    temporal_ordering_ok = True
    if train_core_max_time and train_core_max_time >= dev_min_time:
        temporal_ordering_ok = False
        logger.error(
            f"TEMPORAL VIOLATION: max(TRAIN_CORE.time)={train_core_max_time} >= min(DEV.time)={dev_min_time}"
        )
    if dev_max_time >= test_min_time:
        temporal_ordering_ok = False
        logger.error(
            f"TEMPORAL VIOLATION: max(DEV.time)={dev_max_time} >= min(TEST.time)={test_min_time}"
        )

    if temporal_ordering_ok:
        logger.info("✅ Temporal ordering verified: TRAIN_CORE < DEV < TEST")

    # Metadata for transparency
    metadata = {
        "temporal_boundary": str(test_start),
        "embargo_days": config.embargo_days,
        "embargo_cutoff": str(embargo_cutoff),
        "dev_start": str(dev_start),
        "train_core": {
            "n_cases": len(train_core_ids),
            "n_quotes": len(train_core),
            "date_range": (
                [
                    str(all_cases.loc[train_core_ids, case_time_field].min()),
                    str(all_cases.loc[train_core_ids, case_time_field].max()),
                ]
                if train_core_ids
                else None
            ),
        },
        "dev": dev_metadata,
        "val_cases_included": len([cid for cid in dev_case_ids if cid in val_pool]),
        "temporal_safety": f"All DEV cases < {test_start}",
        "temporal_ordering_verified": temporal_ordering_ok,
        "safety_checks": {
            "max_train_core_time": (
                str(train_core_max_time) if train_core_max_time else None
            ),
            "min_dev_time": str(dev_min_time),
            "max_dev_time": str(dev_max_time),
            "min_test_time": str(test_min_time),
        },
    }

    logger.info(
        f"Temporal split - Train Core: {len(train_core)} quotes, DEV: {len(dev)} quotes"
    )
    logger.info(
        f"DEV includes {metadata['val_cases_included']} cases from original val"
    )

    return train_core, dev, metadata


# REMOVED: All tertile computation functions - using precomputed outcome_bin labels


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)

    return {cls: w for cls, w in zip(classes, weights)}


def compute_tempered_alpha_weights(
    df: pd.DataFrame,
    alpha: float = 0.5,  # Square-root discount
    beta: float = 0.5,  # Tempered inverse frequency
    s_min: float = 0.25,
    s_max: float = 4.0,
    c_min: float = 0.5,
    c_max: float = 2.0,
    use_fold_class_weights: bool = False,
    fold_class_weights: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Compute tempered class weights with √N case discount.

    Implements the following weighting formula:
    w_q = clip(n_c^(-α), s_min, s_max) × clip((1/p_k*)^β, c_min, c_max)

    Where p_k* is computed as:
    p_k* = Σ_{c∈k} n_c · clip(n_c^(-α), s_min, s_max) / Σ_c n_c · clip(n_c^(-α), s_min, s_max)

    Args:
        df: DataFrame with case_id and y (labels)
        alpha: Case discount exponent (0.5 = sqrt)
        beta: Class weight tempering exponent
        s_min/s_max: Clipping bounds for case support
        c_min/c_max: Clipping bounds for class weights
        use_fold_class_weights: Whether to use pre-computed fold class weights
        fold_class_weights: Pre-computed fold class weights (if available)

    Returns:
        weights: Combined weights normalized to mean=1
        normalization_factor: Factor used to normalize weights
        stats: Weight statistics
    """
    # TRAIN fold only
    case_sizes = df.groupby("case_id").size().astype(float)

    # 1) Case support (√N discount, clipped)
    w_case = (case_sizes ** (-alpha)).clip(
        s_min, s_max
    )  # per-quote factor for each case

    # Create per-quote support weights
    df["support_weight"] = df["case_id"].map(w_case).astype(float)

    if use_fold_class_weights and fold_class_weights is not None:
        # Use pre-computed fold class weights
        w_class_dict = fold_class_weights
    else:
        # 2) Compute discounted class priors p_k*
        case_bin = df.drop_duplicates("case_id").set_index("case_id")[
            "y"
        ]  # case→class k
        disc_mass_per_case = case_sizes * w_case  # ≈ n_c^{1-α} (clipped)
        disc_mass_per_class = disc_mass_per_case.groupby(case_bin).sum()
        p_star = disc_mass_per_class / disc_mass_per_class.sum()

        # 3) Tempered class weights
        w_class_dict = {}
        for k in sorted(df["y"].unique()):
            if k in p_star.index:
                w_class_dict[k] = np.clip((1.0 / p_star[k]) ** beta, c_min, c_max)
            else:
                # Handle missing class
                w_class_dict[k] = 1.0

    # 4) Apply class weights
    df["class_weight"] = df["y"].map(w_class_dict).astype(float)

    # 5) Combine weights
    df["sample_weight"] = df["support_weight"] * df["class_weight"]

    # 6) Normalize mean weight to 1 for stability
    normalization_factor = 1.0 / df["sample_weight"].mean()
    weights = df["sample_weight"].values * normalization_factor

    # Verify mean = 1
    assert abs(weights.mean() - 1.0) < 1e-6, f"Weight mean {weights.mean()} != 1.0"

    # Compute statistics
    stats = {
        "alpha": float(alpha),
        "beta": float(beta),
        "s_bounds": [float(s_min), float(s_max)],
        "c_bounds": [float(c_min), float(c_max)],
        "normalization_factor": float(normalization_factor),
        "class_weights": {int(k): float(v) for k, v in w_class_dict.items()},
        "quotes_per_case": {
            "mean": float(case_sizes.mean()),
            "std": float(case_sizes.std()),
            "min": int(case_sizes.min()),
            "max": int(case_sizes.max()),
            "histogram": {
                int(k): int(v)
                for k, v in case_sizes.value_counts().head(20).to_dict().items()
            },
        },
        "support_weights": {
            "mean": float(df["support_weight"].mean()),
            "std": float(df["support_weight"].std()),
            "min": float(df["support_weight"].min()),
            "max": float(df["support_weight"].max()),
        },
        "combined_weights": {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
        },
        "class_priors": (
            {int(k): float(v) for k, v in p_star.items()}
            if not use_fold_class_weights
            else None
        ),
    }

    return weights, normalization_factor, stats


def compute_alpha_normalized_weights(
    df: pd.DataFrame, class_weights: Dict[int, float]
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Compute alpha-normalized combined weights.

    Returns:
        weights: Combined weights with mean=1
        alpha: Normalization factor
        stats: Dictionary of weight statistics
    """
    # Quote-level weights (1/m_case)
    m_case = df.groupby("case_id").size()
    w_quote = 1.0 / df["case_id"].map(m_case).astype(float).values

    # Class weights
    w_class = df["y"].map(class_weights).astype(float).values

    # Base combined weight
    base = w_class * w_quote

    # Alpha normalization
    alpha = len(df) / base.sum()
    w_combined = alpha * base

    # Verify mean = 1
    assert (
        abs(w_combined.mean() - 1.0) < 1e-6
    ), f"Weight mean {w_combined.mean()} != 1.0"

    stats = {
        "alpha": float(alpha),
        "class_weights": {k: float(v) for k, v in class_weights.items()},
        "quotes_per_case": {
            "mean": float(m_case.mean()),
            "std": float(m_case.std()),
            "min": int(m_case.min()),
            "max": int(m_case.max()),
            "histogram": {
                int(k): int(v) for k, v in m_case.value_counts().to_dict().items()
            },
        },
        "combined_weights": {
            "mean": float(w_combined.mean()),
            "std": float(w_combined.std()),
            "min": float(w_combined.min()),
            "max": float(w_combined.max()),
        },
    }

    return w_combined, alpha, stats


def prepare_features(
    df: pd.DataFrame,
    fit_preprocessor: bool = False,
    preprocessor: Optional[ColumnTransformer] = None,
    max_categories: int = 50,
) -> Tuple[pd.DataFrame, Optional[ColumnTransformer]]:
    """Prepare features with strict column governance.

    Uses column governance to ensure only interpretable features are used.
    """
    logger.info(f"Applying column governance to {len(df.columns)} total columns")

    # Apply column governance - filter features but don't error on blocked ones
    try:
        validation = validate_columns(df.columns.tolist(), allow_extra=True)
        feature_cols = validation["interpretable_features"]
    except ValueError as e:
        # If validation fails due to blocked features, extract approved features from the dataframe
        logger.warning(
            f"Column governance found blocked features, filtering them out: {str(e)[:200]}..."
        )

        # Get interpretable features directly from column names
        all_interpretable = [
            col for col in df.columns if col.startswith("interpretable_")
        ]

        # Define the approved features from our feature dictionary
        approved_features = {
            "interpretable_lex_deception_norm",
            "interpretable_lex_deception_present",
            "interpretable_lex_guarantee_norm",
            "interpretable_lex_guarantee_present",
            "interpretable_lex_hedges_norm",
            "interpretable_lex_hedges_present",
            "interpretable_lex_pricing_claims_present",
            "interpretable_lex_superlatives_present",
            "interpretable_ling_high_certainty",
            "interpretable_seq_discourse_additive",
        }

        # Filter to only approved features that exist in dataframe
        feature_cols = [
            col
            for col in all_interpretable
            if col in approved_features and col in df.columns
        ]
        logger.info(f"Using approved features only: {len(feature_cols)} features")

    if len(feature_cols) == 0:
        logger.warning("No approved interpretable features found!")
    else:
        logger.info(
            f"Column governance approved {len(feature_cols)} features for training"
        )

    # Ensure all approved features are numeric
    for col in feature_cols:
        if col in df.columns:
            if str(df[col].dtype) == "object":
                logger.warning(f"Converting object-typed feature {col} to numeric")
                df[col] = pd.to_numeric(df[col], errors="coerce")
                n_nan = df[col].isna().sum()
                if n_nan > 0:
                    logger.info(f"  {col}: {n_nan} values converted to NaN")

    # Create working DataFrame with selected features
    df_feat = df[feature_cols].copy()

    logger.info(f"Using {len(feature_cols)} interpretable features for training")

    # Define feature-specific transformations per feature dictionary
    feature_transforms = {
        # Rates → asinh-rate (expand tiny rates without exploding tails)
        "interpretable_lex_deception_norm": "asinh_rate",
        "interpretable_lex_guarantee_norm": "asinh_rate",
        "interpretable_lex_hedges_norm": "asinh_rate",
        # Counts (were binarized) → log1p + winsor + robust scale
        "interpretable_ling_high_certainty": "log1p_robust",
        "interpretable_seq_discourse_additive": "log1p_robust",
        # Keep presence flags as-is (0/1) - complementary info
        "interpretable_lex_hedges_present": "none",
        "interpretable_lex_superlatives_present": "none",
        "interpretable_lex_deception_present": "none",
        "interpretable_lex_guarantee_present": "none",
        "interpretable_lex_pricing_claims_present": "none",
        # New derived features - ratios are long-tailed
        "interpretable_ratio_guarantee_vs_hedge": "log1p_robust",
        "interpretable_ratio_deception_vs_hedge": "log1p_robust",
        "interpretable_ratio_guarantee_vs_superlative": "log1p_robust",
        # Derived interactions
        "interpretable_interact_guarantee_x_cert": "log1p_robust",
        "interpretable_interact_superlative_x_cert": "log1p_robust",
        "interpretable_interact_hedge_x_guarantee": "log1p_robust",
    }

    # Group features by transformation type
    available_cols = [
        col for col in feature_cols if col in df_feat.columns
    ]  # Use df_feat now
    asinh_rate_cols = [
        col for col in available_cols if feature_transforms.get(col) == "asinh_rate"
    ]
    log1p_robust_cols = [
        col for col in available_cols if feature_transforms.get(col) == "log1p_robust"
    ]
    none_cols = [col for col in available_cols if feature_transforms.get(col) == "none"]
    binarize_cols = [
        col for col in available_cols if feature_transforms.get(col) == "binarize"
    ]  # Keep for backward compatibility
    log1p_cols = [
        col for col in available_cols if feature_transforms.get(col) == "log1p"
    ]  # Keep for backward compatibility

    logger.info(
        f"Feature transformations: {len(asinh_rate_cols)} asinh_rate, {len(log1p_robust_cols)} log1p_robust, {len(none_cols)} none, {len(binarize_cols)} binarize, {len(log1p_cols)} log1p"
    )

    if fit_preprocessor:
        # Create preprocessing pipeline with feature-specific transformations
        transformers = []

        if asinh_rate_cols:
            # Asinh-rate: median impute -> asinh-rate transform -> robust scale
            asinh_rate_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "asinh",
                        FunctionTransformer(
                            _asinh_rate_transform,
                            validate=False,
                            feature_names_out="one-to-one",
                        ),
                    ),
                    ("scaler", RobustScaler(quantile_range=(10, 90))),
                ]
            )
            transformers.append(("asinh_rate", asinh_rate_transformer, asinh_rate_cols))

        if log1p_robust_cols:
            # Log1p-robust: median impute -> winsor -> log1p -> robust scale
            log1p_robust_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "winsor",
                        FunctionTransformer(
                            _winsor99, validate=False, feature_names_out="one-to-one"
                        ),
                    ),
                    (
                        "log1p",
                        FunctionTransformer(
                            _log1p_transform,
                            validate=False,
                            feature_names_out="one-to-one",
                        ),
                    ),
                    ("scaler", RobustScaler(quantile_range=(10, 90))),
                ]
            )
            transformers.append(
                ("log1p_robust", log1p_robust_transformer, log1p_robust_cols)
            )

        if none_cols:
            # None: median impute -> standard scale (for 0/1 flags)
            none_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "scaler",
                        StandardScaler(),
                    ),  # 0/1 → centered, keeps interpretability
                ]
            )
            transformers.append(("none", none_transformer, none_cols))

        # Keep legacy transformers for backward compatibility
        if binarize_cols:
            binarize_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "binarize",
                        FunctionTransformer(
                            _binarize_transform,
                            validate=False,
                            feature_names_out="one-to-one",
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("binarize", binarize_transformer, binarize_cols))

        if log1p_cols:
            log1p_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "log1p",
                        FunctionTransformer(
                            _log1p_transform,
                            validate=False,
                            feature_names_out="one-to-one",
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("log1p", log1p_transformer, log1p_cols))

        if transformers:
            preprocessor = ColumnTransformer(
                transformers=transformers, remainder="drop"
            )

            # Fit on training data using df_feat
            X = preprocessor.fit_transform(df_feat)

            # Get feature names
            feature_names = preprocessor.get_feature_names_out()
            X_df = pd.DataFrame(X, index=df_feat.index, columns=feature_names)

            logger.info(
                f"Applied feature-specific transformations to {X_df.shape[1]} features"
            )

        else:
            # If no features, create empty dataframe
            logger.warning("No approved features found for training!")
            preprocessor = FunctionTransformer(lambda x: x, validate=False)
            X_df = pd.DataFrame(index=df_feat.index)

    else:
        # Transform using existing preprocessor
        df_feat = df[feature_cols].copy()

        if isinstance(preprocessor, FunctionTransformer):
            X_df = pd.DataFrame(index=df_feat.index)
        else:
            X = preprocessor.transform(df_feat)
            feature_names = preprocessor.get_feature_names_out()
            X_df = pd.DataFrame(X, index=df_feat.index, columns=feature_names)

    return X_df, preprocessor


# ------- cumulative isotonic with guards -------
def fit_cumulative_isotonic(
    Q_hat: np.ndarray,
    Q_true: np.ndarray,
    sample_weight=None,
    min_cal_n: int = 500,
    n_bins: int = 30,
):
    """
    Q_hat: (n,2) cumulative preds for P(Y<=0), P(Y<=1)
    Q_true: (n,2) cumulative truths (0/1)
    """

    def _prep(x):
        if len(x) < min_cal_n:
            q = np.linspace(0, 1, n_bins + 1)
            bins = np.quantile(x, q)
            idx = np.clip(np.searchsorted(bins, x, side="right") - 1, 0, n_bins - 1)
            return (idx + 0.5) / n_bins
        return x

    iso1 = IsotonicRegression(out_of_bounds="clip")
    iso2 = IsotonicRegression(out_of_bounds="clip")
    x1 = _prep(Q_hat[:, 0])
    t1 = Q_true[:, 0]
    x2 = _prep(Q_hat[:, 1])
    t2 = Q_true[:, 1]
    iso1.fit(x1, t1, sample_weight=sample_weight)
    iso2.fit(x2, t2, sample_weight=sample_weight)
    return iso1, iso2


def apply_cumulative_isotonic(iso1, iso2, Q_hat: np.ndarray) -> np.ndarray:
    Q1 = iso1.predict(Q_hat[:, 0])
    Q2 = iso2.predict(Q_hat[:, 1])
    p_low = np.clip(Q1, 0, 1)
    p_med = np.clip(Q2 - Q1, 0, 1)
    p_high = np.clip(1.0 - Q2, 0, 1)
    P = np.stack([p_low, p_med, p_high], axis=1)
    Z = P.sum(axis=1, keepdims=True)
    Z[Z == 0] = 1.0
    return P / Z


def fit_cumulative_isotonic_calibration(
    y_true: np.ndarray,
    cum_probs: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    min_cal_n: int = 500,
    iso_bins: int = 30,
) -> List[IsotonicRegression]:
    """Fit isotonic calibration for cumulative probabilities with small-sample guards.

    Args:
        y_true: True labels (0, 1, 2)
        cum_probs: Cumulative probabilities P(Y<=k) for k in {0, 1}
        sample_weight: Sample weights for calibration
        min_cal_n: Minimum samples for direct isotonic (else use binning)
        iso_bins: Number of quantile bins for small-sample isotonic

    Returns:
        List of fitted isotonic regressors
    """
    calibrators = []
    n_samples = len(y_true)

    logger.info(f"Calibration: {n_samples} samples, min_cal_n={min_cal_n}")

    # For each cumulative threshold
    for k in range(2):  # k=0: P(Y<=0), k=1: P(Y<=1)
        # True cumulative indicator
        y_cum_true = (y_true <= k).astype(float)

        # Check for degenerate cases (near-zero positives/negatives)
        n_pos = y_cum_true.sum()
        n_neg = len(y_cum_true) - n_pos

        if n_pos < 5 or n_neg < 5:
            logger.warning(
                f"Threshold {k}: degenerate case (pos={n_pos}, neg={n_neg}), using identity calibration"
            )
            # Identity calibration (no adjustment)
            iso = IsotonicRegression(out_of_bounds="clip")
            # Fit on dummy data to create valid object
            iso.fit([0, 1], [0, 1])
            calibrators.append(iso)
            continue

        # Small sample binning guard
        if n_samples < min_cal_n:
            logger.info(
                f"Small sample ({n_samples} < {min_cal_n}), using {iso_bins} quantile bins for isotonic"
            )
            # Bin the inputs to reduce overfitting
            try:
                bins = np.quantile(cum_probs[:, k], np.linspace(0, 1, iso_bins + 1))
                bins = np.unique(bins)  # Remove duplicates

                if len(bins) < 3:
                    # Too few unique values, use identity
                    logger.warning(
                        f"Too few unique values for binning, using identity calibration"
                    )
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit([0, 1], [0, 1])
                else:
                    # Digitize and average within bins
                    bin_indices = np.digitize(cum_probs[:, k], bins, right=True) - 1
                    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

                    # Average within bins
                    bin_probs = []
                    bin_targets = []
                    bin_weights = []

                    for b in range(len(bins) - 1):
                        mask = bin_indices == b
                        if mask.sum() > 0:
                            bin_probs.append(cum_probs[mask, k].mean())
                            bin_targets.append(y_cum_true[mask].mean())
                            if sample_weight is not None:
                                bin_weights.append(sample_weight[mask].sum())
                            else:
                                bin_weights.append(mask.sum())

                    if len(bin_probs) >= 2:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        if sample_weight is not None:
                            iso.fit(bin_probs, bin_targets, sample_weight=bin_weights)
                        else:
                            iso.fit(bin_probs, bin_targets)
                    else:
                        # Fallback to identity
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit([0, 1], [0, 1])

            except Exception as e:
                logger.warning(f"Binning failed: {e}, using identity calibration")
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit([0, 1], [0, 1])
        else:
            # Direct isotonic regression for large samples
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(cum_probs[:, k], y_cum_true, sample_weight=sample_weight)

        calibrators.append(iso)

    return calibrators


def apply_cumulative_isotonic_calibration(
    cum_probs: np.ndarray, calibrators: List[IsotonicRegression]
) -> np.ndarray:
    """Apply isotonic calibration and convert back to class probabilities.

    Args:
        cum_probs: Uncalibrated cumulative probabilities
        calibrators: Fitted isotonic regressors

    Returns:
        Calibrated class probabilities
    """
    n_samples = cum_probs.shape[0]

    # Calibrate cumulative probabilities
    cal_cum_probs = np.zeros((n_samples, 2))
    for k in range(2):
        cal_cum_probs[:, k] = calibrators[k].predict(cum_probs[:, k])

    # Ensure monotonicity: P(Y<=0) <= P(Y<=1)
    cal_cum_probs[:, 1] = np.maximum(cal_cum_probs[:, 1], cal_cum_probs[:, 0])

    # Convert to class probabilities
    p_low = cal_cum_probs[:, 0]
    p_med = np.clip(cal_cum_probs[:, 1] - cal_cum_probs[:, 0], 0, 1)
    p_high = np.clip(1 - cal_cum_probs[:, 1], 0, 1)

    # Stack and renormalize
    probs = np.stack([p_low, p_med, p_high], axis=1)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs


# ------- weights with alpha normalization -------
def compute_alpha_weights(
    df: pd.DataFrame, class_weights: Dict[int, float]
) -> Tuple[np.ndarray, float]:
    m_case = df["case_id"].value_counts()
    w_class = df["y"].map(class_weights).astype(float).values
    w_quote = 1.0 / df["case_id"].map(m_case).astype(float).values
    base = w_class * w_quote
    alpha = len(df) / base.sum()
    w = alpha * base
    return w, float(alpha)


# ------- safe QWK (handles missing classes) -------
def safe_qwk(y_true, y_pred) -> float:
    present = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    if len(present) < 2:
        return 0.0
    idx = {c: i for i, c in enumerate(present)}
    y_true_m = np.array([idx[int(y)] for y in y_true])
    y_pred_m = np.array([idx[int(y)] for y in y_pred])
    k = len(present)
    w = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
    from sklearn.metrics import confusion_matrix

    O = confusion_matrix(y_true_m, y_pred_m, labels=range(k))
    if O.sum() == 0:
        return 0.0
    row_marg = O.sum(axis=1, keepdims=True)
    col_marg = O.sum(axis=0, keepdims=True)
    E = row_marg @ col_marg / O.sum()
    denom = (w * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (w * O).sum() / denom


def calculate_qwk(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> float:
    """Calculate Quadratic Weighted Kappa with safety for missing classes."""
    return safe_qwk(y_true, y_pred)  # Use safe implementation


def calculate_safe_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> float:
    """Calculate macro F1 only on present classes."""
    present_classes = sorted(set(y_true))
    if len(present_classes) < 2:
        logger.warning("Only 1 class present, returning 0 for macro F1")
        return 0.0

    return f1_score(
        y_true,
        y_pred,
        average="macro",
        sample_weight=sample_weight,
        labels=present_classes,
        zero_division=0,
    )


def calculate_multiclass_brier(
    y_true: np.ndarray, y_prob: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """Calculate multiclass Brier score."""
    n_samples = len(y_true)
    n_classes = y_prob.shape[1]

    # Create one-hot encoding of true labels (ensure integer indices)
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true.astype(int)] = 1

    # Calculate squared differences
    sq_diff = (y_prob - y_true_onehot) ** 2
    brier_scores = sq_diff.sum(axis=1)

    if weights is not None:
        return np.average(brier_scores, weights=weights)
    else:
        return brier_scores.mean()


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    # Per-class ECE
    ece_scores = []

    for class_idx in range(y_prob.shape[1]):
        # Get probabilities for this class
        probs = y_prob[:, class_idx]

        # Binary labels: is this the true class?
        binary_true = (y_true == class_idx).astype(float)

        # Compute ECE for this class
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_ece = 0.0

        for i in range(n_bins):
            # Find samples in this bin
            bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if i == n_bins - 1:  # Include right boundary for last bin
                bin_mask = (probs >= bin_boundaries[i]) & (
                    probs <= bin_boundaries[i + 1]
                )

            if bin_mask.sum() > 0:
                # Average predicted probability in bin
                bin_pred_prob = probs[bin_mask].mean()
                # Actual frequency in bin
                bin_true_prob = binary_true[bin_mask].mean()
                # Weighted absolute difference
                bin_weight = bin_mask.sum() / len(y_true)
                bin_ece += bin_weight * abs(bin_pred_prob - bin_true_prob)

        ece_scores.append(bin_ece)

    # Return mean ECE across classes
    return np.mean(ece_scores)


def emit_polar_predictions(
    record: Dict[str, Any],
    pred_probs: np.ndarray,
    cum_scores: np.ndarray,
    weights: Dict[str, float],
    fold: int,
    split: str,
    hyperparams: Dict[str, Any],
    cutpoints: Optional[Dict[str, float]] = None,
    threshold: float = 0.5,
    model_type: str = "polr",
) -> Dict[str, Any]:
    """Emit POLAR predictions in standard format."""
    k = int(pred_probs.argmax())

    output = {
        # Original record
        **record,
        # POLAR predictions
        f"{EMIT_PREFIX}_pred_bucket": BUCKET_NAMES[k],
        f"{EMIT_PREFIX}_pred_class": k,
        f"{EMIT_PREFIX}_confidence": float(pred_probs[k]),
        f"{EMIT_PREFIX}_class_probs": {
            BUCKET_NAMES[0]: float(pred_probs[0]) if len(pred_probs) > 0 else 0.0,
            BUCKET_NAMES[1]: float(pred_probs[1]) if len(pred_probs) > 1 else 0.0,
            BUCKET_NAMES[2]: float(pred_probs[2]) if len(pred_probs) > 2 else 0.0,
        },
        f"{EMIT_PREFIX}_scores": [
            float(cum_scores[0]),
            float(cum_scores[1]) if len(cum_scores) > 1 else 0.0,
        ],
        f"{EMIT_PREFIX}_prob_low": float(pred_probs[0]) if len(pred_probs) > 0 else 0.0,
        f"{EMIT_PREFIX}_prob_medium": (
            float(pred_probs[1]) if len(pred_probs) > 1 else 0.0
        ),
        f"{EMIT_PREFIX}_prob_high": (
            float(pred_probs[2]) if len(pred_probs) > 2 else 0.0
        ),
        f"{EMIT_PREFIX}_model_threshold": float(threshold),
        f"{EMIT_PREFIX}_model_buckets": BUCKET_NAMES,
        # Metadata
        "weights": weights,
        "fold": fold,
        "split": split,
        "model": model_type,
        "hyperparams": hyperparams,
        "calibration": {"method": "isotonic_cumulative", "version": "v1.0"},
    }

    if cutpoints:
        output["cutpoints"] = cutpoints

    return output


def hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    weights_train: np.ndarray,
    weights_dev: np.ndarray,
    param_grid: Dict[str, List[Any]],
    scoring_priority: List[str],
    reporter: ProgressReporter,
    fold: int,
    dev_metadata: Dict[str, Any],
    model_type: str = "polr",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Perform hyperparameter search for POLR using temporal DEV split.

    Uses robust metrics that handle missing classes gracefully.

    Returns:
        best_params: Best hyperparameters
        all_results: All parameter combinations and scores
    """
    param_combinations = list(ParameterGrid(param_grid))
    reporter.report(
        "HYPERPARAM", f"Testing {len(param_combinations)} parameter combinations"
    )

    # Adapt scoring based on DEV characteristics
    n_classes = dev_metadata.get("n_classes", 3)
    n_dev_samples = len(y_dev)

    adaptive_priority = scoring_priority.copy()
    if n_classes == 1:
        # Single class: use MAE only
        adaptive_priority = ["mae"]
        reporter.report("HYPERPARAM", "Single class detected - using MAE only")
    elif n_classes == 2:
        # Two classes: MAE primary, F1 secondary
        adaptive_priority = ["mae", "macro_f1", "brier"]
        reporter.report("HYPERPARAM", "Two classes detected - MAE primary")

    results = []

    for params in tqdm(param_combinations, desc=f"Fold {fold} hyperparam search"):
        # Train model with dynamic model selection
        if model_type == "mlr":
            model = MLR(**params)
        else:
            model = POLR(**params)

        try:
            model.fit(X_train, y_train, sample_weight=weights_train)

            # Get predictions
            y_pred = model.predict(X_dev)
            y_prob = model.predict_proba(X_dev)

            # Calculate robust metrics
            metrics = {
                "qwk": calculate_qwk(y_dev, y_pred, weights_dev),
                "macro_f1": calculate_safe_macro_f1(y_dev, y_pred, weights_dev),
                "mae": mean_absolute_error(y_dev, y_pred, sample_weight=weights_dev),
                "brier": calculate_multiclass_brier(y_dev, y_prob, weights_dev),
            }

            results.append({"params": params, "metrics": metrics, "converged": True})

        except Exception as e:
            logger.warning(f"Failed to fit with params {params}: {e}")
            results.append(
                {
                    "params": params,
                    "metrics": {metric: -np.inf for metric in adaptive_priority},
                    "converged": False,
                }
            )

    # Find best params based on adaptive scoring priority
    def score_result(result):
        if not result["converged"]:
            return tuple([-np.inf] * len(adaptive_priority))
        scores = []
        for metric in adaptive_priority:
            val = result["metrics"].get(metric, -np.inf)
            # Negate mae and brier so higher is better
            if metric in ["mae", "brier"]:
                val = -val if val != -np.inf else val
            scores.append(val)
        return tuple(scores)

    best_result = max(results, key=score_result)
    best_params = best_result["params"]

    reporter.report("HYPERPARAM", f"Best params: {best_params}")
    reporter.report("HYPERPARAM", f"Best metrics: {best_result['metrics']}")
    reporter.report("HYPERPARAM", f"DEV characteristics: {dev_metadata}")

    return best_params, {
        "all_results": results,
        "best": best_result,
        "dev_metadata": dev_metadata,
    }


def train_polar_cv(config: POLARConfig) -> Dict[str, Any]:
    """Run complete POLAR training with cross-validation protocol.

    Implements the full paper-quality protocol including:
    - Column governance
    - Per-fold tertile cutpoints (if applicable)
    - Alpha-normalized combined weights
    - Hyperparameter search
    - Cumulative isotonic calibration
    - Comprehensive evaluation

    Returns:
        Dictionary with all results, models, and evaluation metrics
    """
    # Setup
    kfold_dir = Path(config.kfold_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results storage
    results = {
        "config": asdict(config),
        "folds": {},
        "oof_predictions": [],
        "final_model": None,
        "timestamp": datetime.now().isoformat(),
    }

    # Count folds - only use folds 0, 1, 2 for hyperparameter search
    fold_dirs = sorted(
        [d for d in kfold_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]
    )
    cv_fold_dirs = [d for d in fold_dirs if d.name in ["fold_0", "fold_1", "fold_2"]]
    n_cv_folds = len(cv_fold_dirs)

    reporter = ProgressReporter(n_cv_folds)

    logger.info(
        f"Starting POLR CV training with {n_cv_folds} folds (0, 1, 2) for hyperparameter search"
    )
    logger.info(f"Output directory: {output_dir}")

    # Process each CV fold (0, 1, 2 only)
    for fold_dir in cv_fold_dirs:
        # Extract fold index from directory name
        fold_idx = int(fold_dir.name.split("_")[1])
        reporter.start_fold(fold_idx)

        # Load data
        train_df, val_df, test_df = load_fold_data(fold_dir)

        # Load fold metadata for pre-computed boundaries and weights
        fold_metadata_path = Path(config.kfold_dir) / "per_fold_metadata.json"
        fold_metadata = None
        fold_specific_edges = None
        fold_class_weights = None

        if fold_metadata_path.exists():
            with open(fold_metadata_path, "r") as f:
                fold_metadata = json.load(f)

                # Get fold-specific tertile edges
                if (
                    "binning" in fold_metadata
                    and "fold_edges" in fold_metadata["binning"]
                ):
                    fold_edges = fold_metadata["binning"]["fold_edges"]
                    if f"fold_{fold_idx}" in fold_edges:
                        fold_specific_edges = fold_edges[f"fold_{fold_idx}"]
                        cutpoints = {
                            "q1": fold_specific_edges[0],
                            "q2": fold_specific_edges[1],
                        }
                        logger.info(
                            f"Using pre-computed fold-specific tertile boundaries: {cutpoints}"
                        )

                # Get fold-specific class weights
                if (
                    "weights" in fold_metadata
                    and f"fold_{fold_idx}" in fold_metadata["weights"]
                ):
                    fold_weight_info = fold_metadata["weights"][f"fold_{fold_idx}"]
                    if "class_weights" in fold_weight_info:
                        # Convert string keys to int
                        fold_class_weights = {
                            int(k): v
                            for k, v in fold_weight_info["class_weights"].items()
                        }
                        logger.info(
                            f"Using pre-computed fold class weights: {fold_class_weights}"
                        )

        # Use precomputed ground truth labels from authoritative data
        if "outcome_bin" in train_df.columns:
            train_df["y"] = train_df["outcome_bin"].astype(int)
            val_df["y"] = val_df["outcome_bin"].astype(int)
            test_df["y"] = test_df["outcome_bin"].astype(int)
            logger.info("Using precomputed outcome_bin labels from authoritative data")
        else:
            raise ValueError(
                "No precomputed outcome_bin field found in authoritative data"
            )

        # Get precomputed cutpoints from metadata for reference only
        cutpoints = None
        if fold_specific_edges is not None:
            cutpoints = {"q1": fold_specific_edges[0], "q2": fold_specific_edges[1]}

        # Use existing splits directly - no temporal DEV splitting needed
        reporter.report("SPLITS", "Using existing train/val splits directly")
        train_core = train_df
        dev_df = val_df

        # Simple metadata for the existing DEV split
        dev_cases = (
            len(dev_df["case_id"].unique())
            if "case_id" in dev_df.columns
            else len(dev_df)
        )
        dev_quotes = len(dev_df)
        dev_classes = len(dev_df["y"].unique())
        dev_class_list = sorted(dev_df["y"].unique().tolist())

        dev_metadata = {
            "n_cases": dev_cases,
            "n_quotes": dev_quotes,
            "n_classes": dev_classes,
            "classes": dev_class_list,
            "source": "existing_val_split",
        }

        logger.info(
            f"DEV set: {dev_cases} cases, {dev_quotes} quotes, {dev_classes} classes"
        )

        # Use precomputed sample weights from authoritative data
        reporter.report(
            "WEIGHTS", "Using precomputed sample weights from authoritative data"
        )

        if "sample_weight" not in train_core.columns:
            raise ValueError(
                "No precomputed sample_weight field found in authoritative data"
            )

        weights_train = train_core["sample_weight"].values
        weights_dev = dev_df["sample_weight"].values

        # Create weight stats for logging (using precomputed values)
        weight_stats_train = {
            "class_weights": fold_class_weights or {0: 1.0, 1: 1.0, 2: 1.0},
            "normalization_factor": 1.0,  # Already normalized in precomputed weights
            "combined_weights": {
                "mean": float(weights_train.mean()),
                "std": float(weights_train.std()),
                "min": float(weights_train.min()),
                "max": float(weights_train.max()),
            },
        }

        weight_stats_dev = {
            "combined_weights": {
                "mean": float(weights_dev.mean()),
                "std": float(weights_dev.std()),
                "min": float(weights_dev.min()),
                "max": float(weights_dev.max()),
            }
        }

        reporter.report(
            "WEIGHTS", f"Precomputed weights - Train mean: {weights_train.mean():.4f}"
        )
        reporter.report(
            "WEIGHTS", f"Using fold class weights: {fold_class_weights or 'balanced'}"
        )

        # Prepare features (fit preprocessor on TRAIN_CORE only)
        reporter.report("FEATURES", "Preparing features with column governance")
        X_train, preprocessor = prepare_features(
            train_core, fit_preprocessor=True, max_categories=config.max_categories
        )
        X_dev, _ = prepare_features(
            dev_df, preprocessor=preprocessor, max_categories=config.max_categories
        )

        # Convert to numpy arrays for model fitting
        X_train_np = X_train.values if hasattr(X_train, "values") else X_train
        X_dev_np = X_dev.values if hasattr(X_dev, "values") else X_dev

        reporter.report("FEATURES", f"Feature dimensions: {X_train_np.shape[1]}")

        # Skip saving fold preprocessors (only save final preprocessor)
        preprocessor_path = output_dir / f"fold_{fold_idx}_preprocessor.joblib"

        # Hyperparameter search on DEV
        best_params, hp_results = hyperparameter_search(
            X_train_np,
            train_core["y"].values,
            X_dev_np,
            dev_df["y"].values,
            weights_train,
            weights_dev,
            config.hyperparameter_grid,
            config.scoring_priority,
            reporter,
            fold_idx,
            dev_metadata,
            model_type=config.model_type,
        )

        # Train final model for this fold with best params
        model_name = "MLR" if config.model_type == "mlr" else "POLR"
        reporter.report("TRAINING", f"Training {model_name} with best hyperparameters")
        if config.model_type == "mlr":
            model = MLR(**best_params)
        else:
            model = POLR(**best_params)
        model.fit(X_train_np, train_core["y"].values, sample_weight=weights_train)

        # Get cumulative probabilities for calibration on DEV
        cum_probs_dev = model.get_cumulative_probs(X_dev_np)

        # Fit calibration on DEV with guards (skip if DEV is too small or has insufficient classes)
        if len(dev_df) < 10 or len(dev_df["y"].unique()) < 2:
            logger.warning(
                f"Skipping calibration: DEV too small ({len(dev_df)} samples) or insufficient classes ({len(dev_df['y'].unique())})"
            )
            calibrators = None
            cal_probs_dev = cum_probs_dev  # Use uncalibrated probabilities
        else:
            reporter.report(
                "CALIBRATION", "Fitting cumulative isotonic calibration on DEV"
            )
            calibrators = fit_cumulative_isotonic_calibration(
                dev_df["y"].values,
                cum_probs_dev,
                weights_dev,
                min_cal_n=config.min_cal_n,
                iso_bins=config.iso_bins,
            )
            # Apply calibration and get final predictions on DEV
            cal_probs_dev = apply_cumulative_isotonic_calibration(
                cum_probs_dev, calibrators
            )

        y_pred_dev = cal_probs_dev.argmax(axis=1)

        # Calculate DEV metrics with robust functions
        dev_metrics = {
            "qwk": calculate_qwk(dev_df["y"].values, y_pred_dev, weights_dev),
            "macro_f1": calculate_safe_macro_f1(
                dev_df["y"].values, y_pred_dev, weights_dev
            ),
            "mae": mean_absolute_error(
                dev_df["y"].values, y_pred_dev, sample_weight=weights_dev
            ),
            "ece": calculate_ece(dev_df["y"].values, cal_probs_dev),
            "brier": calculate_multiclass_brier(
                dev_df["y"].values, cal_probs_dev, weights_dev
            ),
        }

        reporter.finish_fold(fold_idx, dev_metrics)

        # Save fold results
        fold_results = {
            "cutpoints": cutpoints,
            "class_weights": fold_class_weights,
            "weight_stats": {"train_core": weight_stats_train, "dev": weight_stats_dev},
            "dev_metadata": dev_metadata,
            "best_params": best_params,
            "hp_results": hp_results,
            "dev_metrics": dev_metrics,
            "model_path": str(output_dir / f"fold_{fold_idx}_model.joblib"),
            "calibrators_path": str(output_dir / f"fold_{fold_idx}_calibrators.joblib"),
            "preprocessor_path": str(preprocessor_path),
        }

        # Skip saving intermediate fold models (only save final model)

        results["folds"][fold_idx] = fold_results

        # Generate OOF predictions for DEV set
        for idx, (_, row) in enumerate(dev_df.iterrows()):
            # Calculate quote weight correctly
            case_quotes = train_core[train_core["case_id"] == row["case_id"]]
            quote_weight = 1.0 / len(case_quotes) if len(case_quotes) > 0 else 1.0

            pred_record = emit_polar_predictions(
                row.to_dict(),
                cal_probs_dev[idx],
                cum_probs_dev[idx],
                {
                    "class": float(
                        fold_class_weights[row["y"]] if fold_class_weights else 1.0
                    ),
                    "quote": float(quote_weight),
                    "combined": float(weights_dev[idx]),
                },
                fold_idx,
                "validation",
                best_params,
                cutpoints,
                model_type=config.model_type,
            )
            results["oof_predictions"].append(pred_record)

    # Aggregate OOF metrics
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 60)

    def _nanmean(xs):
        a = np.array(xs, dtype=float)
        a = a[~np.isnan(a)]
        return float(a.mean()) if len(a) else float("nan")

    def _nanstd(xs):
        a = np.array(xs, dtype=float)
        a = a[~np.isnan(a)]
        return float(a.std()) if len(a) else float("nan")

    all_metrics = {}
    for metric in config.scoring_priority:
        # Only aggregate metrics from folds 0, 1, 2
        metric_values = []
        for fold_idx in [0, 1, 2]:
            if fold_idx in results["folds"]:
                metric_values.append(results["folds"][fold_idx]["dev_metrics"][metric])

        all_metrics[metric] = {
            "mean": _nanmean(metric_values),
            "std": _nanstd(metric_values),
            "values": metric_values,
        }
        logger.info(
            f"{metric}: {all_metrics[metric]['mean']:.4f} ± {all_metrics[metric]['std']:.4f}"
        )

    results["cv_metrics"] = all_metrics

    # Save results with JSON safety (skipping CV OOF predictions as requested)
    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(pyify(results), f, indent=2)
    logger.info(f"Saved CV results to {results_path}")

    # Skip paper assets during CV - they will be generated from final model OOF results
    logger.info("\n" + "=" * 60)
    logger.info("CV COMPLETE - PAPER ASSETS WILL BE GENERATED FROM FINAL MODEL")
    logger.info("=" * 60)

    return results


def train_final_polar_model(
    config: POLARConfig, cv_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Train final POLAR model on all data (train+val+test) excluding OOF cases.

    Uses the best hyperparameters from CV, trains on ALL fold data excluding
    OOF cases, then evaluates on the clean OOF test set.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING FINAL POLAR MODEL")
    logger.info("=" * 60)

    output_dir = Path(config.output_dir)
    kfold_dir = Path(config.kfold_dir)

    # CRITICAL FIX: Don't average C from degenerate folds - re-select on proper DEV
    # First get a reasonable starting point from non-degenerate folds
    valid_folds = []
    for i in range(len(cv_results["folds"])):
        fold_info = cv_results["folds"][i]
        dev_meta = fold_info.get("dev_metadata", {})
        # Only consider folds with at least 2 classes and reasonable size
        if dev_meta.get("n_classes", 0) >= 2 and dev_meta.get("n_quotes", 0) >= 50:
            valid_folds.append(i)

    if valid_folds:
        # Use params from valid folds only
        all_best_params = [cv_results["folds"][i]["best_params"] for i in valid_folds]
    else:
        # Fallback to all folds if none are valid
        logger.warning("No valid folds found for hyperparameter selection, using all")
    all_best_params = [
        cv_results["folds"][i]["best_params"] for i in range(len(cv_results["folds"]))
    ]

    # Get baseline params (for non-C parameters)
    final_params = {}
    for key in all_best_params[0].keys():
        values = [p[key] for p in all_best_params]
        if key == "C":
            # Don't average C - we'll re-select it below
            final_params[key] = 1.0  # Default, will be overridden
        elif isinstance(values[0], (int, float)):
            final_params[key] = np.mean(values)
            if key in ["max_iter"]:
                final_params[key] = int(final_params[key])
        else:
            from collections import Counter

            final_params[key] = Counter(values).most_common(1)[0][0]

    logger.info(f"Baseline hyperparameters (C will be re-selected): {final_params}")

    # Load fold 3's train + dev data for final training
    fold3_dir = kfold_dir / "fold_3"
    if not fold3_dir.exists():
        raise ValueError(f"Fold 3 directory not found at {fold3_dir}")

    all_parts = []
    # Load fold 3 train
    train_path = fold3_dir / "train.jsonl"
    if train_path.exists():
        train_df = pd.read_json(train_path, lines=True)
        all_parts.append(train_df)
        logger.info(f"Loaded fold_3/train.jsonl: {len(train_df)} rows")

    # Load fold 3 dev (used for calibration)
    dev_path = fold3_dir / "dev.jsonl"
    if dev_path.exists():
        dev_df = pd.read_json(dev_path, lines=True)
        all_parts.append(dev_df)
        logger.info(f"Loaded fold_3/dev.jsonl: {len(dev_df)} rows")
    else:
        # Fallback to val.jsonl if dev.jsonl doesn't exist
        val_path = fold3_dir / "val.jsonl"
        if val_path.exists():
            val_df = pd.read_json(val_path, lines=True)
            all_parts.append(val_df)
            logger.info(f"Loaded fold_3/val.jsonl: {len(val_df)} rows")

    final_train_df = pd.concat(all_parts, ignore_index=True)
    logger.info(f"Final training set size from fold_3: {len(final_train_df)}")

    # Load and exclude OOF cases
    oof_dir = kfold_dir / "oof_test"
    oof_case_ids = set()

    # Try to get OOF case IDs from case_ids.json first
    case_ids_json = oof_dir / "case_ids.json"
    if case_ids_json.exists():
        with open(case_ids_json, "r") as f:
            oof_data = json.load(f)
            oof_case_ids = set(oof_data.get("test_case_ids", []))

    # If no case_ids.json or empty, get from OOF test file
    oof_test_path = oof_dir / "test.jsonl"
    if not oof_case_ids and oof_test_path.exists():
        oof_df_temp = pd.read_json(oof_test_path, lines=True)
        if "case_id" in oof_df_temp.columns:
            oof_case_ids = set(oof_df_temp["case_id"].astype(str).unique())

    # Exclude OOF cases from training data
    if "case_id" in final_train_df.columns and oof_case_ids:
        final_train_df["case_id"] = final_train_df["case_id"].astype(str)
        mask = ~final_train_df["case_id"].isin(oof_case_ids)
        excluded_count = len(final_train_df) - mask.sum()
        final_train_df = final_train_df[mask].copy()
        logger.info(
            f"Excluded {excluded_count} rows overlapping with {len(oof_case_ids)} OOF case IDs"
        )

    logger.info(f"Final training set size after OOF exclusion: {len(final_train_df)}")

    # Load fold 3 metadata for boundaries
    fold_metadata_path = Path(config.kfold_dir) / "per_fold_metadata.json"
    fold3_edges = None
    fold3_class_weights = None

    if fold_metadata_path.exists():
        with open(fold_metadata_path, "r") as f:
            fold_metadata = json.load(f)

            # Get fold 3 tertile edges
            if "binning" in fold_metadata and "fold_edges" in fold_metadata["binning"]:
                fold_edges = fold_metadata["binning"]["fold_edges"]
                if "fold_3" in fold_edges:
                    fold3_edges = fold_edges["fold_3"]
                    cutpoints = {"q1": fold3_edges[0], "q2": fold3_edges[1]}
                    logger.info(
                        f"Using pre-computed fold_3 tertile boundaries: {cutpoints}"
                    )

            # Get fold 3 class weights
            if "weights" in fold_metadata and "fold_3" in fold_metadata["weights"]:
                fold3_weight_info = fold_metadata["weights"]["fold_3"]
                if "class_weights" in fold3_weight_info:
                    # Convert string keys to int
                    fold3_class_weights = {
                        int(k): v for k, v in fold3_weight_info["class_weights"].items()
                    }
                    logger.info(
                        f"Using pre-computed fold_3 class weights: {fold3_class_weights}"
                    )

    # Use precomputed ground truth labels from authoritative data
    if "outcome_bin" in final_train_df.columns:
        final_train_df["y"] = final_train_df["outcome_bin"].astype(int)
        logger.info("Using precomputed outcome_bin labels from authoritative data")
    else:
        raise ValueError("No precomputed outcome_bin field found in authoritative data")

    # Get precomputed cutpoints from metadata for reference only
    cutpoints = None
    if fold3_edges is not None:
        cutpoints = {"q1": fold3_edges[0], "q2": fold3_edges[1]}
        logger.info(f"Reference cutpoints from fold_3 metadata: {cutpoints}")

    # Use precomputed sample weights from authoritative data
    if "sample_weight" not in final_train_df.columns:
        raise ValueError(
            "No precomputed sample_weight field found in authoritative data"
        )

    weights_all = final_train_df["sample_weight"].values
    norm_factor = 1.0  # Already normalized in precomputed weights
    weight_stats = {
        "class_weights": fold3_class_weights or {0: 1.0, 1: 1.0, 2: 1.0},
        "normalization_factor": norm_factor,
        "combined_weights": {
            "mean": float(weights_all.mean()),
            "std": float(weights_all.std()),
            "min": float(weights_all.min()),
            "max": float(weights_all.max()),
        },
    }

    logger.info(f"Using precomputed sample weights - mean: {weights_all.mean():.4f}")
    logger.info(f"Final class weights from metadata: {weight_stats['class_weights']}")

    # CRITICAL FIX: Re-select C on a proper 3-class DEV set
    from sklearn.model_selection import StratifiedGroupKFold

    logger.info("\n" + "=" * 60)
    logger.info("RE-SELECTING C ON PROPER 3-CLASS DEV SET")
    logger.info("=" * 60)

    # Candidate C values - start with reasonable range
    cand_C = [0.01, 0.1, 1.0, 10.0, 100.0]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.seed)

    # Find a split with all 3 classes in DEV
    best_c_selection = None
    for split_idx, (train_idx, dev_idx) in enumerate(
        sgkf.split(final_train_df, final_train_df["y"], final_train_df["case_id"])
    ):
        dev_y = final_train_df.iloc[dev_idx]["y"].values
        n_dev_classes = len(np.unique(dev_y))
        n_dev_cases = len(final_train_df.iloc[dev_idx]["case_id"].unique())

        logger.info(
            f"Split {split_idx}: DEV has {n_dev_classes} classes, {len(dev_idx)} quotes, {n_dev_cases} cases"
        )

        if n_dev_classes < 3:
            continue

        # We found a good DEV set with all 3 classes
        logger.info(f"Using split {split_idx} for C selection (3 classes present)")

        train_sub = final_train_df.iloc[train_idx]
        dev_sub = final_train_df.iloc[dev_idx]

        # Compute weights for this split using same class weights
        w_train_sub, _, _ = compute_tempered_alpha_weights(
            train_sub.copy(),
            alpha=0.5,
            beta=0.5,
            use_fold_class_weights=True,
            fold_class_weights=weight_stats["class_weights"],
        )
        w_dev_sub, _, _ = compute_tempered_alpha_weights(
            dev_sub.copy(),
            alpha=0.5,
            beta=0.5,
            use_fold_class_weights=True,
            fold_class_weights=weight_stats["class_weights"],
        )

        # Prepare features
        X_tr, pre = prepare_features(
            train_sub, fit_preprocessor=True, max_categories=config.max_categories
        )
        X_dv, _ = prepare_features(
            dev_sub, preprocessor=pre, max_categories=config.max_categories
        )

        X_tr_np = X_tr.values if hasattr(X_tr, "values") else X_tr
        X_dv_np = X_dv.values if hasattr(X_dv, "values") else X_dv

        y_tr = train_sub["y"].values
        y_dv = dev_sub["y"].values

        # Test each C value
        results = []
        for C in cand_C:
            logger.info(f"Testing C={C}")
            try:
                if config.model_type == "mlr":
                    m = MLR(
                        C=C,
                        solver=final_params["solver"],
                        max_iter=final_params["max_iter"],
                        tol=final_params["tol"],
                        random_state=config.seed,
                        class_weight=final_params.get("class_weight"),
                    )
                else:
                    m = POLR(
                        C=C,
                        solver=final_params["solver"],
                        max_iter=final_params["max_iter"],
                        tol=final_params["tol"],
                        random_state=config.seed,
                    )
                m.fit(X_tr_np, y_tr, sample_weight=w_train_sub)

                # Get uncalibrated predictions for C selection
                p_dv = m.predict_proba(X_dv_np)
                yhat = p_dv.argmax(axis=1)

                # Prioritize ordinal discrimination (QWK) and macro F1
                qwk = calculate_qwk(y_dv, yhat, w_dev_sub)
                f1 = calculate_safe_macro_f1(y_dv, yhat, w_dev_sub)
                mae = mean_absolute_error(y_dv, yhat, sample_weight=w_dev_sub)

                results.append(
                    {
                        "C": C,
                        "qwk": qwk,
                        "f1": f1,
                        "mae": mae,
                        "model": m,
                        "preprocessor": pre,
                    }
                )

                logger.info(f"  C={C}: QWK={qwk:.4f}, F1={f1:.4f}, MAE={mae:.4f}")

            except Exception as e:
                logger.warning(f"Failed to fit with C={C}: {e}")

        if results:
            # Select best C based on QWK (primary) and F1 (secondary)
            best = max(results, key=lambda r: (r["qwk"], r["f1"]))
            best_c_selection = best
            final_params["C"] = float(best["C"])
            logger.info(
                f"\nSelected C={best['C']} (QWK={best['qwk']:.4f}, F1={best['f1']:.4f})"
            )
            break

    if best_c_selection is None:
        # Fallback if we couldn't find a 3-class DEV
        logger.warning("Could not find 3-class DEV set, using C=1.0")
        final_params["C"] = 1.0

    logger.info(f"\nFinal hyperparameters with selected C: {final_params}")

    # Now do the calibration split
    if config.calibration_split > 0:
        n_splits = max(2, int(round(1.0 / config.calibration_split)))
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=config.seed + 1
        )  # Different seed
        train_idx, cal_idx = next(
            sgkf.split(final_train_df, final_train_df["y"], final_train_df["case_id"])
        )

        train_subset = final_train_df.iloc[train_idx]
        cal_subset = final_train_df.iloc[cal_idx]
        weights_train = weights_all[train_idx]
        weights_cal = weights_all[cal_idx]

        logger.info(
            f"Training subset: {len(train_subset)}, Calibration subset: {len(cal_subset)}"
        )
    else:
        train_subset = final_train_df
        cal_subset = None
        weights_train = weights_all
        weights_cal = None
        logger.info("No calibration split - using all data for training")

    # Prepare features
    X_train, preprocessor = prepare_features(
        train_subset, fit_preprocessor=True, max_categories=config.max_categories
    )
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train

    # Train final model
    model_name = "MLR" if config.model_type == "mlr" else "POLR"
    logger.info(f"Training final {model_name} model...")
    if config.model_type == "mlr":
        final_model = MLR(**final_params)
    else:
        final_model = POLR(**final_params)
    final_model.fit(X_train_np, train_subset["y"].values, sample_weight=weights_train)

    # Calibration (if enabled)
    calibrators = None
    if config.calibration_split > 0 and cal_subset is not None and len(cal_subset) > 0:
        logger.info("Fitting calibration on held-out data...")
        X_cal, _ = prepare_features(
            cal_subset, preprocessor=preprocessor, max_categories=config.max_categories
        )
        X_cal_np = X_cal.values if hasattr(X_cal, "values") else X_cal

    cum_probs_cal = final_model.get_cumulative_probs(X_cal_np)
    calibrators = fit_cumulative_isotonic_calibration(
        cal_subset["y"].values,
        cum_probs_cal,
        weights_cal,
        min_cal_n=config.min_cal_n,
        iso_bins=config.iso_bins,
    )

    # Save model artifacts
    final_model_path = output_dir / "final_polar_model.joblib"
    preprocessor_path = output_dir / "final_preprocessor.joblib"
    calibrators_path = output_dir / "final_calibrators.joblib"

    joblib.dump(final_model, final_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(calibrators, calibrators_path)

    # ============================================================
    # EVALUATE ON OOF TEST SET
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING ON OOF TEST SET")
    logger.info("=" * 60)

    if not oof_test_path.exists():
        logger.warning(f"OOF test file not found at {oof_test_path}")
        oof_metrics = {}
        oof_predictions = []
    else:
        # Load OOF test data
        oof_df = pd.read_json(oof_test_path, lines=True)
        logger.info(f"OOF test set size: {len(oof_df)}")

        # Use precomputed ground truth labels from authoritative OOF data
        if "outcome_bin" in oof_df.columns:
            oof_df["y"] = oof_df["outcome_bin"].astype(int)
            logger.info(
                "Using precomputed outcome_bin labels from authoritative OOF data"
            )

            # Log OOF label distribution for verification
            label_dist = oof_df["y"].value_counts().sort_index()
            logger.info(f"OOF label distribution: {label_dist.to_dict()}")

            if cutpoints is not None and "final_judgement_real" in oof_df.columns:
                # Optional: log continuous outcome distribution for reference
                outcome_dist = oof_df["final_judgement_real"].describe()
                logger.info(f"OOF continuous outcome distribution for reference:")
                logger.info(
                    f"  min={outcome_dist['min']:.0f}, median={outcome_dist['50%']:.0f}, max={outcome_dist['max']:.0f}"
                )
                logger.info(
                    f"  Reference cutpoints: q1={cutpoints['q1']:.0f}, q2={cutpoints['q2']:.0f}"
                )
        else:
            raise ValueError(
                "OOF file missing precomputed outcome_bin field from authoritative data"
            )

        # Prepare OOF features
        X_oof, _ = prepare_features(
            oof_df, preprocessor=preprocessor, max_categories=config.max_categories
        )
        X_oof_np = X_oof.values if hasattr(X_oof, "values") else X_oof

        # Get predictions
        cum_probs_oof = final_model.get_cumulative_probs(X_oof_np)

        # Apply calibration if available
        if calibrators is not None:
            prob_oof = apply_cumulative_isotonic_calibration(cum_probs_oof, calibrators)
            logger.info("Applied isotonic calibration to OOF predictions")
        else:
            # Convert cumulative to class probabilities manually
            p_low = np.clip(cum_probs_oof[:, 0], 0, 1)
            p_med = np.clip(cum_probs_oof[:, 1] - cum_probs_oof[:, 0], 0, 1)
            p_high = np.clip(1.0 - cum_probs_oof[:, 1], 0, 1)
            prob_oof = np.stack([p_low, p_med, p_high], axis=1)
            prob_oof = prob_oof / prob_oof.sum(axis=1, keepdims=True)  # Renormalize
            logger.info("Using uncalibrated predictions for OOF evaluation")

        # Calculate metrics
        y_true_oof = oof_df["y"].values
        y_pred_oof = prob_oof.argmax(axis=1)

        oof_metrics = {
            "qwk": float(calculate_qwk(y_true_oof, y_pred_oof)),
            "macro_f1": float(calculate_safe_macro_f1(y_true_oof, y_pred_oof)),
            "mae": float(mean_absolute_error(y_true_oof, y_pred_oof)),
            "ece": float(calculate_ece(y_true_oof, prob_oof)),
            "brier": float(calculate_multiclass_brier(y_true_oof, prob_oof)),
            "n_samples": int(len(oof_df)),
            "classes_present": [int(x) for x in sorted(np.unique(y_true_oof))],
            "calibrated": calibrators is not None,
        }

        logger.info("OOF Metrics:")
        for metric, value in oof_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        # CRITICAL: Add sanity checks
        logger.info("\n" + "=" * 60)
        logger.info("SANITY CHECKS")
        logger.info("=" * 60)

        # Class balance check
        def log_class_counts(df, name):
            vc = df["y"].value_counts().sort_index()
            counts = {int(k): int(v) for k, v in vc.items()}
            pcts = {int(k): f"{100*v/len(df):.1f}%" for k, v in vc.items()}
            logger.info(f"{name} class distribution: {counts} (percentages: {pcts})")
            return counts

        train_counts = log_class_counts(final_train_df, "TRAIN (after OOF exclusion)")
        oof_counts = log_class_counts(oof_df, "OOF")

        # Prediction collapse check
        pred_counts = pd.Series(y_pred_oof).value_counts().sort_index()
        pred_dict = {int(k): int(v) for k, v in pred_counts.items()}
        pred_pcts = {
            int(k): f"{100*v/len(y_pred_oof):.1f}%" for k, v in pred_counts.items()
        }
        logger.info(
            f"OOF predictions distribution: {pred_dict} (percentages: {pred_pcts})"
        )

        # Check for prediction collapse
        if len(pred_counts) < 3:
            logger.warning(
                f"PREDICTION COLLAPSE: Model only predicting {len(pred_counts)} classes!"
            )

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_true_oof, y_pred_oof, labels=[0, 1, 2])
        logger.info("\nConfusion Matrix:")
        logger.info("True\\Pred   0    1    2")
        for i, row in enumerate(cm):
            logger.info(f"    {i}    {row[0]:4d} {row[1]:4d} {row[2]:4d}")

        # Per-class metrics
        logger.info("\nPer-class metrics:")
        logger.info(
            classification_report(y_true_oof, y_pred_oof, labels=[0, 1, 2], digits=3)
        )

        # Generate prediction records
        oof_predictions = []
        m_case = (
            oof_df["case_id"].value_counts()
            if "case_id" in oof_df.columns
            else pd.Series(1.0, index=oof_df.index)
        )

        for i, (_, row) in enumerate(oof_df.iterrows()):
            case_id = row.get("case_id", None)
            quote_weight = (
                float(1.0 / m_case.get(case_id, 1.0)) if case_id is not None else 1.0
            )

            pred_record = emit_polar_predictions(
                row.to_dict(),
                prob_oof[i],
                cum_probs_oof[i],
                {
                    "class": float(
                        weight_stats["class_weights"].get(int(row["y"]), 1.0)
                    ),
                    "quote": quote_weight,
                    "combined": 1.0,  # Not using actual combined weights for OOF
                },
                fold=-1,
                split="oof",
                hyperparams=final_params,
                cutpoints=cutpoints,
                model_type=config.model_type,
            )
            oof_predictions.append(pred_record)

        # Save OOF results
        oof_pred_path = output_dir / "final_oof_predictions.jsonl"
        oof_metrics_path = output_dir / "final_oof_metrics.json"

        pd.DataFrame(oof_predictions).to_json(
            oof_pred_path, orient="records", lines=True
        )
        with open(oof_metrics_path, "w") as f:
            json.dump(oof_metrics, f, indent=2)

        logger.info(f"\nOOF predictions saved to: {oof_pred_path}")
        logger.info(f"OOF metrics saved to: {oof_metrics_path}")

        # Generate paper assets from final OOF results
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING PAPER ASSETS FROM FINAL OOF RESULTS")
        logger.info("=" * 60)

        try:
            import subprocess
            import sys

            # Create paper assets directory
            paper_assets_dir = output_dir / "paper_assets"
            paper_assets_dir.mkdir(exist_ok=True)

            # Generate LaTeX tables using both CV and final OOF results
            cv_results_path = output_dir / "cv_results.json"
            tables_cmd = [
                sys.executable,
                "scripts/make_paper_tables.py",
                "--oof",
                str(oof_pred_path),
                "--cv",
                str(cv_results_path),
                "--out",
                str(paper_assets_dir),
            ]

            logger.info("Generating LaTeX tables from CV and final OOF results...")
            result = subprocess.run(
                tables_cmd, capture_output=True, text=True, cwd=Path.cwd()
            )
            if result.returncode == 0:
                logger.info("✓ LaTeX tables generated successfully from OOF results")
            else:
                logger.warning(f"LaTeX table generation failed: {result.stderr}")

            # Generate paper figures using both CV and final OOF results
            figures_cmd = [
                sys.executable,
                "scripts/make_paper_figures.py",
                "--oof",
                str(oof_pred_path),
                "--cv",
                str(cv_results_path),
                "--out",
                str(paper_assets_dir),
            ]

            logger.info("Generating paper figures from final OOF results...")
            result = subprocess.run(
                figures_cmd, capture_output=True, text=True, cwd=Path.cwd()
            )
            if result.returncode == 0:
                logger.info("✓ Paper figures generated successfully from OOF results")
                logger.info(f"Final paper assets saved to: {paper_assets_dir}")
            else:
                logger.warning(f"Figure generation failed: {result.stderr}")

        except Exception as e:
            logger.warning(f"Could not generate paper assets from OOF results: {e}")

    # Save comprehensive metadata
    metadata = {
        "model_path": str(final_model_path),
        "preprocessor_path": str(preprocessor_path),
        "calibrators_path": str(calibrators_path),
        "hyperparameters": final_params,
        "cutpoints": (
            [float(cutpoints["q1"]), float(cutpoints["q2"])]
            if cutpoints is not None
            else None
        ),
        "class_weights": {
            str(k): float(v) for k, v in (fold3_class_weights or {}).items()
        },
        "weight_stats": {
            str(k): float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in weight_stats.items()
        },
        "training_size": len(train_subset),
        "calibration_size": len(cal_subset) if cal_subset is not None else 0,
        "oof_evaluation": oof_metrics,
        "oof_cases_excluded": len(oof_case_ids),
        "included_folds": sorted(
            [
                d.name
                for d in kfold_dir.iterdir()
                if d.is_dir() and d.name.startswith("fold_")
            ]
        ),
        "timestamp": datetime.now().isoformat(),
    }

    # Ensure all metadata is JSON serializable
    metadata = pyify(metadata)

    metadata_path = output_dir / "final_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nFinal model saved to: {final_model_path}")
    logger.info(f"Final metadata saved to: {metadata_path}")

    return metadata
