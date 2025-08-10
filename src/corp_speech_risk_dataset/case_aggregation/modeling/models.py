"""Lightweight sklearn models and evaluation utilities for case outcomes.

We favor fast, explainable baselines initially:
- Multinomial Logistic Regression
- Linear SVM (via LinearSVC with one-vs-rest)
- RandomForestClassifier (shallow, for robustness checks)

All models operate on per-case aggregated features produced by `dataset.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import polars as pl
from loguru import logger

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score


ModelName = str


def _to_numpy_Xy(
    X: pl.DataFrame, y: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert Polars frames to numpy arrays, returning class labels as strings."""
    # Ensure consistent ordering by case_id
    joined = X.join(y, on="case_id", how="inner")
    feature_cols = [c for c in X.columns if c != "case_id"]
    X_np = joined.select(feature_cols).to_numpy()
    y_np = joined.select("outcome_bucket").to_series().to_list()
    return X_np, np.array(y_np), feature_cols


def _split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_model_registry() -> Dict[ModelName, object]:
    """Return a small registry of candidate baseline models."""
    return {
        # Let sklearn default to multinomial in 1.7+; avoid future warning by omitting explicit multi_class
        "logreg": LogisticRegression(solver="lbfgs", max_iter=2000),
        "linsvc": LinearSVC(),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
    }


def evaluate_models(
    X: pl.DataFrame, y: pl.DataFrame, test_size: float = 0.2
) -> Dict[str, Dict[str, object]]:
    """Train/test split evaluation across the registry, returning metrics per model.

    The returned dict per model includes: accuracy, report (classification_report dict),
    n_train, n_test, features, y_true, y_pred for plotting confusion matrices.
    """
    X_np, y_np, feature_cols = _to_numpy_Xy(X, y)
    if len(np.unique(y_np)) < 2 or len(y_np) < 5:
        logger.warning(
            "Insufficient label diversity or sample size; skipping eval", n=len(y_np)
        )
        return {}
    X_train, X_test, y_train, y_test = _split(X_np, y_np, test_size=test_size)

    results: Dict[str, Dict[str, object]] = {}
    for name, model in build_model_registry().items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            report = classification_report(
                y_test, preds, output_dict=True, zero_division=0
            )
            results[name] = {
                "accuracy": acc,
                "report": report,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "features": feature_cols,
                "y_true": y_test.tolist(),
                "y_pred": preds.tolist(),
            }
        except Exception:
            logger.exception("Model failed during training/evaluation", model=name)
            continue
    return results


def evaluate_regressors(
    X: pl.DataFrame, y_reg: pl.DataFrame, test_size: float = 0.2
) -> Dict[str, Dict[str, object]]:
    """Evaluate simple interpretable regressors to predict numeric outcomes.

    Models: LinearRegression, Ridge. Returns r2 and coefficients for interpretability.
    """
    # Align and prepare
    joined = X.join(y_reg, on="case_id", how="inner")
    feature_cols = [c for c in X.columns if c != "case_id"]
    if not feature_cols or joined.is_empty():
        return {}
    X_np = joined.select(feature_cols).to_numpy()
    y_np = joined.select("final_judgement_real").to_numpy().ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=42
    )
    regressors = {
        "linreg": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
    }
    out: Dict[str, Dict[str, object]] = {}
    for name, model in regressors.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = float(r2_score(y_test, y_pred))
            coefs = getattr(model, "coef_", None)
            out[name] = {
                "r2": r2,
                "features": feature_cols,
                "coefficients": coefs.tolist() if hasattr(coefs, "tolist") else None,
            }
        except Exception:
            logger.exception("Regressor failed during training/evaluation", model=name)
            continue
    return out
