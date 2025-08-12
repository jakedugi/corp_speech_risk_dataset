"""Enhanced models with cross-validation, tuning, and interpretability.

This module provides interpretable baseline models with:
- Cross-validation and hyperparameter tuning
- Feature importance extraction
- Statistical significance testing
- Confidence intervals via bootstrap
- Interpretable ensemble methods
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import polars as pl
from loguru import logger
from scipy import stats

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    r2_score,
    mean_squared_error,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


ModelName = str


def _to_numpy_Xy(
    X: pl.DataFrame, y: pl.DataFrame, target_col: str = "outcome_bucket"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert Polars frames to numpy arrays."""
    joined = X.join(y, on="case_id", how="inner")
    feature_cols = [c for c in X.columns if c != "case_id"]
    X_np = joined.select(feature_cols).to_numpy()
    y_np = joined.select(target_col).to_numpy().ravel()
    return X_np, y_np, feature_cols


def build_classification_models() -> Dict[ModelName, Tuple[object, Dict[str, Any]]]:
    """Return interpretable classification models with hyperparameter grids."""
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        ),
        "logreg_l1": (
            LogisticRegression(
                max_iter=2000, penalty="l1", solver="saga", random_state=42
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "linsvc": (
            LinearSVC(max_iter=5000, random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "loss": ["hinge", "squared_hinge"],
            },
        ),
        "rf": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            {
                "max_depth": [4, 8, 12],
                "min_samples_split": [5, 10, 20],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "dt": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [3, 5, 7],
                "min_samples_split": [5, 10, 20],
                "criterion": ["gini", "entropy"],
            },
        ),
    }


def build_regression_models() -> Dict[ModelName, Tuple[object, Dict[str, Any]]]:
    """Return interpretable regression models with hyperparameter grids."""
    return {
        "linear": (LinearRegression(), {}),  # No hyperparameters to tune
        "ridge": (
            Ridge(random_state=42),
            {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
        "lasso": (
            Lasso(random_state=42, max_iter=2000),
            {
                "alpha": [0.001, 0.01, 0.1, 1.0],
            },
        ),
        "elastic": (
            ElasticNet(random_state=42, max_iter=2000),
            {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.2, 0.5, 0.8],
            },
        ),
    }


def compute_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence intervals for a metric."""
    n_samples = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) > 1:
            score = metric_func(y_true_boot, y_pred_boot)
            scores.append(score)

    if not scores:
        return 0.0, 0.0, 0.0

    scores = np.array(scores)
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean = np.mean(scores)

    return float(mean), float(lower), float(upper)


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Extract feature importance from fitted model."""
    importance_dict = {}

    # Try different methods based on model type
    if hasattr(model, "coef_"):
        # Linear models
        coef = model.coef_
        if coef.ndim > 1:
            # Multi-class: use mean absolute coefficient
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
        for name, imp in zip(feature_names, importance):
            importance_dict[name] = float(imp)

    elif hasattr(model, "feature_importances_"):
        # Tree-based models
        for name, imp in zip(feature_names, model.feature_importances_):
            importance_dict[name] = float(imp)

    else:
        # Fallback: permutation importance
        try:
            perm_imp = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            for name, imp in zip(feature_names, perm_imp.importances_mean):
                importance_dict[name] = float(imp)
        except Exception:
            logger.warning("Could not extract feature importance")

    return importance_dict


def statistical_test_models(
    scores_a: List[float],
    scores_b: List[float],
    test_type: str = "paired_t",
) -> Dict[str, float]:
    """Perform statistical significance testing between two models."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    if test_type == "paired_t":
        # Paired t-test for cross-validation scores
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    elif test_type == "wilcoxon":
        # Non-parametric alternative
        t_stat, p_value = stats.wilcoxon(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return {
        "test_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
        "se_diff": float(
            np.std(np.array(scores_a) - np.array(scores_b)) / np.sqrt(len(scores_a))
        ),
    }


def evaluate_models_cv(
    X: pl.DataFrame,
    y: pl.DataFrame,
    cv_folds: int = 5,
    enable_tuning: bool = True,
    test_size: float = 0.2,
    feature_selection: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate classification models with cross-validation and tuning."""
    X_np, y_np, feature_cols = _to_numpy_Xy(X, y)

    # Check data validity
    unique_classes = np.unique(y_np)
    if len(unique_classes) < 2 or len(y_np) < 20:
        logger.warning(
            "Insufficient data for modeling", n=len(y_np), classes=len(unique_classes)
        )
        return {}

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=42, stratify=y_np
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, (model, param_grid) in build_classification_models().items():
        try:
            # Build pipeline with optional feature selection
            steps = []
            if feature_selection:
                steps.append(
                    ("feature_selection", SelectKBest(f_classif, k=feature_selection))
                )
            steps.append(("classifier", model))

            pipeline = Pipeline(steps)

            # Adjust param grid for pipeline
            pipeline_params = {f"classifier__{k}": v for k, v in param_grid.items()}

            if enable_tuning and param_grid:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    pipeline,
                    pipeline_params,
                    cv=cv_splitter,
                    scoring="accuracy",
                    n_jobs=-1,
                    error_score="raise",
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                cv_scores = grid_search.cv_results_["mean_test_score"][
                    grid_search.best_index_
                ]
                cv_std = grid_search.cv_results_["std_test_score"][
                    grid_search.best_index_
                ]
                best_params = grid_search.best_params_
            else:
                # Simple cross-validation without tuning
                best_model = pipeline
                cv_scores_array = cross_val_score(
                    best_model,
                    X_train_scaled,
                    y_train,
                    cv=cv_splitter,
                    scoring="accuracy",
                )
                best_model.fit(X_train_scaled, y_train)
                cv_scores = cv_scores_array.mean()
                cv_std = cv_scores_array.std()
                best_params = {}

            # Test set predictions
            y_pred = best_model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)

            # Bootstrap confidence intervals
            acc_mean, acc_lower, acc_upper = compute_confidence_intervals(
                y_test, y_pred, accuracy_score
            )

            # Classification report
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )

            # Feature importance
            if (
                hasattr(best_model, "named_steps")
                and "classifier" in best_model.named_steps
            ):
                clf = best_model.named_steps["classifier"]
            else:
                clf = best_model

            feature_importance = extract_feature_importance(
                clf, feature_cols, X_test_scaled, y_test
            )

            # Multi-class AUC if applicable
            try:
                if len(unique_classes) > 2:
                    # Get prediction probabilities
                    if hasattr(best_model, "predict_proba"):
                        y_proba = best_model.predict_proba(X_test_scaled)
                        auc = roc_auc_score(
                            y_test, y_proba, multi_class="ovr", average="macro"
                        )
                    else:
                        auc = None
                else:
                    auc = None
            except Exception:
                auc = None

            results[name] = {
                "cv_accuracy_mean": float(cv_scores),
                "cv_accuracy_std": float(cv_std),
                "test_accuracy": float(test_accuracy),
                "accuracy_ci": (acc_mean, acc_lower, acc_upper),
                "best_params": best_params,
                "classification_report": report,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "feature_importance": feature_importance,
                "auc_macro": float(auc) if auc else None,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
            }

        except Exception as e:
            logger.exception(f"Model {name} failed", error=str(e))
            continue

    return results


def evaluate_regressors_cv(
    X: pl.DataFrame,
    y_reg: pl.DataFrame,
    cv_folds: int = 5,
    enable_tuning: bool = True,
    test_size: float = 0.2,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate regression models with cross-validation."""
    X_np, y_np, feature_cols = _to_numpy_Xy(X, y_reg, target_col="final_judgement_real")

    if len(y_np) < 20:
        logger.warning("Insufficient data for regression", n=len(y_np))
        return {}

    # Remove NaN values
    valid_mask = ~np.isnan(y_np)
    X_np = X_np[valid_mask]
    y_np = y_np[valid_mask]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, (model, param_grid) in build_regression_models().items():
        try:
            if enable_tuning and param_grid:
                # Grid search
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv_splitter,
                    scoring="r2",
                    n_jobs=-1,
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                cv_r2 = grid_search.cv_results_["mean_test_score"][
                    grid_search.best_index_
                ]
                cv_std = grid_search.cv_results_["std_test_score"][
                    grid_search.best_index_
                ]
                best_params = grid_search.best_params_
            else:
                # Simple cross-validation
                best_model = model
                cv_scores = cross_val_score(
                    best_model, X_train_scaled, y_train, cv=cv_splitter, scoring="r2"
                )
                best_model.fit(X_train_scaled, y_train)
                cv_r2 = cv_scores.mean()
                cv_std = cv_scores.std()
                best_params = {}

            # Test predictions
            y_pred = best_model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)

            # Bootstrap confidence intervals for R²
            r2_mean, r2_lower, r2_upper = compute_confidence_intervals(
                y_test, y_pred, r2_score, n_bootstrap=500
            )

            # Extract coefficients for interpretability
            coefficients = {}
            if hasattr(best_model, "coef_"):
                for feat, coef in zip(feature_cols, best_model.coef_):
                    coefficients[feat] = float(coef)

            # Feature importance
            feature_importance = extract_feature_importance(
                best_model, feature_cols, X_test_scaled, y_test
            )

            # Prediction intervals (approximate via residual std)
            residuals = y_test - y_pred
            residual_std = np.std(residuals)

            results[name] = {
                "cv_r2_mean": float(cv_r2),
                "cv_r2_std": float(cv_std),
                "test_r2": float(test_r2),
                "r2_ci": (r2_mean, r2_lower, r2_upper),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                "best_params": best_params,
                "coefficients": coefficients,
                "feature_importance": feature_importance,
                "residual_std": float(residual_std),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }

        except Exception as e:
            logger.exception(f"Regressor {name} failed", error=str(e))
            continue

    return results


def build_interpretable_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build interpretable ensemble using voting with weights based on CV performance."""
    # Select top 3 models based on CV accuracy
    model_scores = [
        (name, info["cv_accuracy_mean"])
        for name, info in base_results.items()
        if "cv_accuracy_mean" in info
    ]
    model_scores.sort(key=lambda x: x[1], reverse=True)
    top_models = model_scores[:3]

    if len(top_models) < 2:
        return {}

    # Build ensemble
    estimators = []
    weights = []

    for name, score in top_models:
        model_class, _ = build_classification_models()[name]
        estimators.append((name, model_class))
        weights.append(score)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
    )

    try:
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Bootstrap CI
        acc_mean, acc_lower, acc_upper = compute_confidence_intervals(
            y_test, y_pred, accuracy_score
        )

        return {
            "ensemble_accuracy": float(accuracy),
            "accuracy_ci": (acc_mean, acc_lower, acc_upper),
            "base_models": [name for name, _ in top_models],
            "weights": weights,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }
    except Exception as e:
        logger.exception("Ensemble failed", error=str(e))
        return {}


def analyze_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Analyze model fairness across outcome classes."""
    # Class-wise performance
    unique_classes = np.unique(y_true)
    class_metrics = {}

    for cls in unique_classes:
        mask = y_true == cls
        if mask.sum() > 0:
            recall = np.mean(y_pred[mask] == cls)
            precision_mask = y_pred == cls
            if precision_mask.sum() > 0:
                precision = np.mean(y_true[precision_mask] == cls)
            else:
                precision = 0.0

            class_metrics[str(cls)] = {
                "recall": float(recall),
                "precision": float(precision),
                "support": int(mask.sum()),
            }

    # Compute disparate impact ratio (worst vs best recall)
    recalls = [m["recall"] for m in class_metrics.values()]
    if len(recalls) >= 2:
        disparate_impact = min(recalls) / max(recalls) if max(recalls) > 0 else 0
    else:
        disparate_impact = 1.0

    return {
        "class_metrics": class_metrics,
        "disparate_impact_ratio": float(disparate_impact),
        "balanced_accuracy": float(
            np.mean([m["recall"] for m in class_metrics.values()])
        ),
    }
