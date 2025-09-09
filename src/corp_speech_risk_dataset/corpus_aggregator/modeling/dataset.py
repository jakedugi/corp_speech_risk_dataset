"""Dataset construction for case-level outcome modeling.

This module aggregates quote-level risk predictions into per-case features under
different positional thresholds, and produces train/test splits with labels
bucketed into three ordinal classes using 33/33/33 quantiles of
`final_judgement_real` over non-null cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import polars as pl

from .utils import select_relevant_columns


@dataclass(frozen=True)
class ThresholdSpec:
    """Specification for a positional threshold used to filter quotes.

    Types:
    - complete: keep all quotes (no filtering)
    - docket_fraction: keep quotes in the first fraction of dockets
    - token_fraction: keep quotes in the first fraction of global tokens
    - token_budget: keep quotes whose token span lies under a global token budget
    """

    name: str
    kind: str  # "complete" | "docket_fraction" | "token_fraction" | "token_budget"
    value: float  # fraction (0..1] for fraction kinds; integer budget for budget kind


DEFAULT_THRESHOLDS: List[ThresholdSpec] = [
    ThresholdSpec(name="complete_case", kind="complete", value=1.0),
    ThresholdSpec(name="docket_half", kind="docket_fraction", value=0.5),
    ThresholdSpec(name="docket_third", kind="docket_fraction", value=1.0 / 3.0),
    ThresholdSpec(name="token_half", kind="token_fraction", value=0.5),
    ThresholdSpec(name="token_third", kind="token_fraction", value=1.0 / 3.0),
    ThresholdSpec(name="token_2500", kind="token_budget", value=2500.0),
]


@dataclass(frozen=True)
class FeatureConfig:
    """Toggles for optional feature families in aggregation."""

    include_mean_probabilities: bool = True
    include_counts: bool = True
    include_confidence: bool = False
    include_pred_class: bool = False
    include_scores: bool = False
    include_position_stats: bool = False
    include_model_threshold: bool = False
    use_pred_class_only: bool = False  # Use coral_pred_class as primary input


DEFAULT_FEATURE_CONFIG = FeatureConfig()


def _bucket_outcomes(df_cases: pl.DataFrame) -> pl.DataFrame:
    """Add a three-bucket label column `outcome_bucket` to case-level frame.

    Buckets are computed via empirical 33rd/66th percentiles on
    `final_judgement_real` excluding nulls.
    """
    df_non_null = df_cases.filter(pl.col("final_judgement_real").is_not_null())
    if df_non_null.is_empty():
        return df_cases.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("outcome_bucket")
        )
    vals = df_non_null.select("final_judgement_real").to_series().drop_nans().to_list()
    p33, p66 = np.percentile(vals, [33, 66])

    def _bucket_expr(col: pl.Expr) -> pl.Expr:
        return (
            pl.when(col.is_null())
            .then(None)
            .when(col <= p33)
            .then(pl.lit("low"))
            .when(col <= p66)
            .then(pl.lit("medium"))
            .otherwise(pl.lit("high"))
        )

    return df_cases.with_columns(
        _bucket_expr(pl.col("final_judgement_real")).alias("outcome_bucket")
    )


def _aggregate_case_features_for_threshold(
    df: pl.DataFrame, spec: ThresholdSpec, features: FeatureConfig
) -> pl.DataFrame:
    """Aggregate quote-level predictions into per-case features under a threshold.

    Input df columns required: case_id, docket_number, global_token_start,
    num_tokens, coral_prob_low, coral_prob_medium, coral_prob_high.
    Optional columns are utilized when available: coral_pred_bucket,
    coral_pred_class, coral_confidence.

    Returns a per-case DataFrame with columns:
        case_id, f_count_low, f_count_medium, f_count_high

    Notes:
    - Extensible: add more aggregated features here (means of probabilities, etc.).
    """
    if df.is_empty():
        return pl.DataFrame([])

    # Precompute per-case max docket and approximate total tokens
    df_props = (
        df.group_by("case_id")
        .agg(
            pl.col("docket_number").max().alias("max_docket"),
            (pl.col("global_token_start") + pl.col("num_tokens"))
            .max()
            .alias("approx_total_tokens"),
        )
        .with_columns(
            pl.col("max_docket").fill_null(0),
            pl.col("approx_total_tokens").fill_null(0),
        )
    )

    df_join = df.join(df_props, on="case_id", how="left")

    if spec.kind == "complete":
        # Keep all quotes for complete case analysis
        df_join = df_join.with_columns(pl.lit(True).alias("keep"))
    elif spec.kind == "docket_fraction":
        df_join = df_join.with_columns(
            (
                pl.col("docket_number")
                <= (
                    pl.max_horizontal(
                        pl.lit(1), (pl.col("max_docket") * spec.value).floor()
                    )
                )
            ).alias("keep")
        )
    elif spec.kind == "token_fraction":
        df_join = df_join.with_columns(
            (
                (pl.col("global_token_start") + pl.col("num_tokens"))
                <= (pl.col("approx_total_tokens") * spec.value).floor()
            ).alias("keep")
        )
    elif spec.kind == "token_budget":
        df_join = df_join.with_columns(
            ((pl.col("global_token_start") + pl.col("num_tokens")) <= spec.value).alias(
                "keep"
            )
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown threshold kind: {spec.kind}")

    df_filt = df_join.filter(pl.col("keep"))

    # Determine risk label based on configuration
    if features.use_pred_class_only:
        # Use coral_pred_class directly, mapping 0/1/2 to low/medium/high
        df_labeled = df_filt.with_columns(
            pl.when(pl.col("coral_pred_class") == 0)
            .then(pl.lit("low"))
            .when(pl.col("coral_pred_class") == 1)
            .then(pl.lit("medium"))
            .when(pl.col("coral_pred_class") == 2)
            .then(pl.lit("high"))
            .otherwise(None)  # Handle missing/invalid values
            .alias("risk_label")
        )
        # Filter out quotes without valid pred_class
        df_labeled = df_labeled.filter(pl.col("risk_label").is_not_null())
    else:
        # Original logic: use pred_bucket or argmax of probabilities
        df_labeled = df_filt.with_columns(
            pl.when(pl.col("coral_pred_bucket").is_not_null())
            .then(pl.col("coral_pred_bucket"))
            .when(
                (pl.col("coral_prob_low") >= pl.col("coral_prob_medium"))
                & (pl.col("coral_prob_low") >= pl.col("coral_prob_high"))
            )
            .then(pl.lit("low"))
            .when(
                (pl.col("coral_prob_medium") >= pl.col("coral_prob_low"))
                & (pl.col("coral_prob_medium") >= pl.col("coral_prob_high"))
            )
            .then(pl.lit("medium"))
            .otherwise(pl.lit("high"))
            .alias("risk_label")
        )

    # Build aggregation list based on requested features
    agg_exprs: List[pl.Expr] = []
    if features.include_counts:
        agg_exprs.extend(
            [
                (pl.col("risk_label") == "low").sum().alias("f_count_low"),
                (pl.col("risk_label") == "medium").sum().alias("f_count_medium"),
                (pl.col("risk_label") == "high").sum().alias("f_count_high"),
                pl.len().alias("f_total_quotes"),
            ]
        )
    if features.include_mean_probabilities:
        agg_exprs.extend(
            [
                pl.col("coral_prob_low").mean().alias("f_mean_prob_low"),
                pl.col("coral_prob_medium").mean().alias("f_mean_prob_medium"),
                pl.col("coral_prob_high").mean().alias("f_mean_prob_high"),
                # Expected ordinal risk score per quote (0*low + 1*medium + 2*high), averaged
                ((pl.col("coral_prob_medium") + 2.0 * pl.col("coral_prob_high")))
                .mean()
                .alias("f_mean_risk_score"),
                # Peak/maximum variants (severity proxies)
                ((pl.col("coral_prob_medium") + 2.0 * pl.col("coral_prob_high")))
                .max()
                .alias("f_max_risk_score"),
                pl.col("coral_prob_high").max().alias("f_max_prob_high"),
            ]
        )
    if features.include_confidence and ("coral_confidence" in df_labeled.columns):
        agg_exprs.append(pl.col("coral_confidence").mean().alias("f_mean_confidence"))
    if features.include_pred_class and ("coral_pred_class" in df_labeled.columns):
        agg_exprs.append(pl.col("coral_pred_class").mean().alias("f_mean_pred_class"))
    if features.include_scores and ("coral_scores" in df_labeled.columns):
        # Expect list-like of length >= 2; use list namespace for variable-length lists
        agg_exprs.extend(
            [
                pl.col("coral_scores")
                .list.get(0)
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("f_mean_score_0"),
                pl.col("coral_scores")
                .list.get(1)
                .cast(pl.Float64, strict=False)
                .mean()
                .alias("f_mean_score_1"),
            ]
        )
    if features.include_model_threshold and (
        "coral_model_threshold" in df_labeled.columns
    ):
        agg_exprs.append(
            pl.col("coral_model_threshold").mean().alias("f_mean_model_threshold")
        )
    if features.include_position_stats:
        agg_exprs.extend(
            [
                pl.col("docket_number").median().alias("f_median_docket_number"),
                pl.col("global_token_start")
                .median()
                .alias("f_median_global_token_start"),
            ]
        )

    df_case_feats = df_labeled.group_by("case_id").agg(agg_exprs)
    # Derive densities from counts
    if features.include_counts:
        df_case_feats = df_case_feats.with_columns(
            ((pl.col("f_count_high") / pl.col("f_total_quotes")).fill_null(0.0)).alias(
                "f_density_high"
            ),
            (
                (pl.col("f_count_medium") / pl.col("f_total_quotes")).fill_null(0.0)
            ).alias("f_density_medium"),
            ((pl.col("f_count_low") / pl.col("f_total_quotes")).fill_null(0.0)).alias(
                "f_density_low"
            ),
        )

    return df_case_feats


def build_datasets(
    rows: List[Dict[str, Any]],
    thresholds: Optional[List[ThresholdSpec]] = None,
    features: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> Dict[str, "DatasetBundle"]:
    """Construct per-threshold datasets: (X, y) frames keyed by threshold name.

    - X columns: counts and mean probabilities per case
    - y column: outcome_bucket (low/medium/high)
    - Cases with null outcomes are dropped.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    df = select_relevant_columns(rows)
    if df.is_empty():
        return {}

    # Build case outcome frame
    df_cases = (
        df.select(["case_id", "final_judgement_real"])
        .group_by("case_id")
        .agg(pl.col("final_judgement_real").first().alias("final_judgement_real"))
    )
    df_cases = _bucket_outcomes(df_cases)
    df_cases = df_cases.filter(pl.col("outcome_bucket").is_not_null())

    # For each threshold, aggregate features and join labels
    datasets: Dict[str, DatasetBundle] = {}
    for spec in thresholds:
        feats = _aggregate_case_features_for_threshold(df, spec, features)
        if feats.is_empty():
            continue
        joined = feats.join(df_cases, on="case_id", how="inner")
        if joined.is_empty():
            continue
        # Dynamically select all feature columns (anything except labels and id)
        X_cols = [
            c
            for c in joined.columns
            if c not in {"outcome_bucket", "final_judgement_real", "case_id"}
        ]
        X = joined.select(["case_id", *X_cols])
        y = joined.select(
            ["case_id", "outcome_bucket"]
        )  # keep case_id for traceability
        y_reg = joined.select(["case_id", "final_judgement_real"])  # numeric outcome
        covered_case_ids = (
            y.select("case_id").to_series().cast(pl.Utf8, strict=False).to_list()
        )
        total_case_ids = (
            df_cases.select("case_id").to_series().cast(pl.Utf8, strict=False).to_list()
        )
        datasets[spec.name] = DatasetBundle(
            X=X,
            y=y,
            y_reg=y_reg,
            covered_case_ids=covered_case_ids,
            total_case_ids=total_case_ids,
        )

    return datasets


@dataclass(frozen=True)
class DatasetBundle:
    """Container for per-threshold dataset and coverage metadata."""

    X: pl.DataFrame
    y: pl.DataFrame
    y_reg: pl.DataFrame
    covered_case_ids: List[str]
    total_case_ids: List[str]
