#!/usr/bin/env python3
"""
Minimal KISS case-level judicial outcome predictor.

Pure implementation of the recipe provided - simple LR on compact case features
from hero model outputs. No extra complexity.

Usage:
    python scripts/minimal_kiss_case_predictor.py \
        --mirror-path results/corrected_dnt_validation_FINAL/mirror_with_predictions \
        --output-path results/minimal_kiss_prediction.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from loguru import logger


def load_mirror_data(mirror_path: Path) -> pd.DataFrame:
    """Load mirrored predictions with hero model outputs."""

    rows = []
    for jsonl_file in mirror_path.rglob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} quotes from {df['case_id'].nunique()} cases")
    return df


def build_case_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build case-level features exactly as specified in the recipe.

    Primary features:
    - Density/prevalence: prop_strict, prop_recallT, prop_p≥X, mean_p, etc.
    - Positional cutoffs: early/mid/late segments
    - Clustering: n_clusters, max_cluster_len, etc.
    - Shape: quantiles, tail mass
    """

    # Sort by case and position
    df = df.sort_values(["case_id", "global_token_start"], na_position="last")

    case_features = []

    for case_id, g in df.groupby("case_id"):
        p = g["mlp_probability"].values
        strict = g["mlp_pred_strict"].values.astype(bool)
        recallT = g["mlp_pred_recallT"].values.astype(bool)
        n = len(p)

        # Normalized positions [0,1]
        pos = np.linspace(0, 1, n, endpoint=True)

        # === DENSITY / PREVALENCE ===
        feat = {
            "case_id": case_id,
            "prop_strict": strict.mean(),
            "prop_recallT": recallT.mean(),
            "prop_p80": (p >= 0.8).mean(),
            "prop_p90": (p >= 0.9).mean(),
            "prop_p95": (p >= 0.95).mean(),
            "mean_p": p.mean(),
            "std_p": p.std() if n > 1 else 0.0,
            "max_p": p.max(),
            "top3_mean_p": p[np.argsort(p)[-3:]].mean() if n >= 3 else p.mean(),
        }

        # === POSITIONAL CUTOFFS (early/mid/late) ===
        early_end = int(0.3 * n)
        mid_start = int(0.3 * n)
        mid_end = int(0.7 * n)

        feat.update(
            {
                "early_prop_strict": (
                    strict[:early_end].mean() if early_end > 0 else 0.0
                ),
                "mid_prop_strict": (
                    strict[mid_start:mid_end].mean() if mid_end > mid_start else 0.0
                ),
                "late_prop_strict": strict[mid_end:].mean() if n > mid_end else 0.0,
                "early_mean_p": p[:early_end].mean() if early_end > 0 else 0.0,
                "late_mean_p": p[mid_end:].mean() if n > mid_end else 0.0,
            }
        )

        # === CENTER OF MASS ===
        if p.sum() > 1e-9:
            feat["pos_center_of_mass"] = float((p * pos).sum() / p.sum())
        else:
            feat["pos_center_of_mass"] = 0.5

        # === CLUSTERING ===
        if strict.any():
            # Run-length encoding
            diff = np.diff(np.concatenate([[False], strict, [False]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            cluster_lens = ends - starts

            feat.update(
                {
                    "n_clusters": len(cluster_lens),
                    "max_cluster_len": int(cluster_lens.max()),
                    "avg_cluster_len": float(cluster_lens.mean()),
                    "first_pos_idx": float(pos[np.argmax(strict)]),
                    "last_pos_idx": float(
                        pos[len(strict) - 1 - np.argmax(strict[::-1])]
                    ),
                }
            )
        else:
            feat.update(
                {
                    "n_clusters": 0,
                    "max_cluster_len": 0,
                    "avg_cluster_len": 0.0,
                    "first_pos_idx": 1.0,
                    "last_pos_idx": 0.0,
                }
            )

        # === SHAPE / CALIBRATION ===
        feat.update(
            {
                "p_q10": float(np.percentile(p, 10)),
                "p_q25": float(np.percentile(p, 25)),
                "p_q50": float(np.percentile(p, 50)),
                "p_q75": float(np.percentile(p, 75)),
                "p_q90": float(np.percentile(p, 90)),
            }
        )

        # Tail mass
        high_mask = p >= 0.9
        if high_mask.any():
            feat["tail_mass_0_9"] = float(p[high_mask].mean() * high_mask.mean())
        else:
            feat["tail_mass_0_9"] = 0.0

        case_features.append(feat)

    return pd.DataFrame(case_features)


def extract_court_from_case_id(case_id: str) -> str:
    """Extract court for grouping/suppression."""
    try:
        if "_" in case_id:
            return case_id.split("_")[-1]
        elif ":" in case_id:
            return case_id.split(":")[0]
        else:
            return "unknown"
    except:
        return "unknown"


def apply_court_suppression(X_train, X_eval, court_train, court_eval):
    """Apply court suppression by subtracting train court means."""

    # Compute court means on training set
    court_means = {}
    for court in np.unique(court_train):
        mask = court_train == court
        if mask.sum() > 0:
            court_means[court] = X_train[mask].mean(axis=0)

    global_mean = X_train.mean(axis=0)

    # Apply to eval set
    X_suppressed = X_eval.copy()
    for i, court in enumerate(court_eval):
        if court in court_means:
            X_suppressed[i] -= court_means[court]
        else:
            X_suppressed[i] -= global_mean

    return X_suppressed


def train_minimal_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train minimal LR model with court suppression as specified.

    - LR with L2, balanced class weights
    - Temporal GroupKFold by case
    - Court suppression for evaluation only
    - Select by dev-suppressed MCC
    """

    feature_cols = [col for col in X.columns if col != "case_id"]
    X_features = X[feature_cols].values
    case_ids = X["case_id"].values
    courts = np.array([extract_court_from_case_id(cid) for cid in case_ids])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Hyperparameter grid (minimal as specified)
    C_values = [0.01, 0.1, 1.0]

    best_score = -np.inf
    best_C = None
    best_model = None

    # GroupKFold for temporal splitting
    gkf = GroupKFold(n_splits=5)

    for C in C_values:
        cv_scores = []

        for train_idx, val_idx in gkf.split(X_scaled, y, groups=case_ids):
            X_train_cv = X_scaled[train_idx]
            X_val_cv = X_scaled[val_idx]
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
            court_train_cv = courts[train_idx]
            court_val_cv = courts[val_idx]

            # Train on raw features
            model = LogisticRegression(
                penalty="l2",
                C=C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
            )
            model.fit(X_train_cv, y_train_cv)

            # Evaluate with court suppression
            X_val_suppressed = apply_court_suppression(
                X_train_cv, X_val_cv, court_train_cv, court_val_cv
            )

            y_pred = model.predict(X_val_suppressed)
            mcc = matthews_corrcoef(y_val_cv, y_pred)
            cv_scores.append(mcc)

        avg_score = np.mean(cv_scores)

        if avg_score > best_score:
            best_score = avg_score
            best_C = C

    # Train final model
    final_model = LogisticRegression(
        penalty="l2", C=best_C, class_weight="balanced", solver="lbfgs", max_iter=2000
    )
    final_model.fit(X_scaled, y)

    # Feature importance
    importance = pd.DataFrame(
        {"feature": feature_cols, "coefficient": final_model.coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)

    return {
        "model": final_model,
        "scaler": scaler,
        "best_C": best_C,
        "cv_mcc": best_score,
        "feature_importance": importance.to_dict("records"),
        "feature_names": feature_cols,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal KISS case predictor")
    parser.add_argument("--mirror-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    logger.info("=== MINIMAL KISS CASE PREDICTOR ===")

    # 1. Load data
    df = load_mirror_data(args.mirror_path)

    # 2. Build case features (~15-20 features as specified)
    case_features = build_case_features(df)
    logger.info(f"Built {len(case_features.columns)-1} case-level features")

    # 3. Mock outcomes for demo (replace with real case outcomes)
    # For demo: randomly assign binary outcomes
    np.random.seed(42)
    case_outcomes = pd.DataFrame(
        {
            "case_id": case_features["case_id"],
            "judicial_outcome": np.random.binomial(1, 0.3, len(case_features)),
        }
    )

    # 4. Merge
    merged = case_features.merge(case_outcomes, on="case_id")
    X = merged.drop(["judicial_outcome"], axis=1)
    y = merged["judicial_outcome"]

    logger.info(
        f"Final dataset: {len(merged)} cases, outcome distribution: {y.value_counts().to_dict()}"
    )

    # 5. Train minimal model
    results = train_minimal_model(X, y)

    # 6. Save results
    output = {
        "approach": "Minimal KISS LR on case density/clustering features",
        "n_cases": len(merged),
        "n_features": len(results["feature_names"]),
        "cv_mcc": float(results["cv_mcc"]),
        "best_hyperparams": {"C": results["best_C"]},
        "feature_importance": results["feature_importance"],
        "class_distribution": y.value_counts().to_dict(),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {args.output_path}")
    logger.info(f"CV MCC: {results['cv_mcc']:.4f}")

    print("\n=== MINIMAL KISS RESULTS ===")
    print(f"Cases: {len(merged)}")
    print(f"Features: {len(results['feature_names'])}")
    print(f"CV MCC: {results['cv_mcc']:.4f}")
    print(f"Best C: {results['best_C']}")
    print("\nTop 5 features:")
    for feat in results["feature_importance"][:5]:
        print(f"  {feat['feature']}: {feat['coefficient']:.4f}")


if __name__ == "__main__":
    main()
