"""
Metrics and evaluation functions for corpus-temporal-cv module.

Provides comprehensive evaluation metrics for temporal CV experiments.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)
from scipy import stats
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    if len(np.unique(y_true)) < 2:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def find_mcc_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float]:
    """Find threshold that maximizes MCC."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    thresholds = np.linspace(0.0, 1.0, 201)
    best_mcc = -1
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:  # Avoid all same class
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

    return best_threshold, best_mcc


def calculate_operating_point_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Calculate precision, recall, specificity at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle degenerate case (all one class)
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "mcc": mcc,
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        },
        "threshold": threshold,
    }


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = None
) -> Dict[str, Any]:
    """Comprehensive evaluation of predictions."""
    results = {}

    # Basic checks
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in true labels")
        return {"error": "insufficient_classes"}

    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_mcc_optimal_threshold(y_true, y_prob)

    # Classification metrics
    results["threshold"] = threshold
    results["operating_point"] = calculate_operating_point_metrics(
        y_true, y_prob, threshold
    )

    # Probabilistic metrics
    try:
        results["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        results["auc"] = 0.5

    try:
        results["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        results["pr_auc"] = 0.0

    results["brier_score"] = brier_score_loss(y_true, y_prob)
    results["ece"] = calculate_ece(y_true, y_prob)

    # Additional metrics
    y_pred = (y_prob >= threshold).astype(int)
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    results["f1"] = f1_score(y_true, y_pred, zero_division=0)

    return results


def compare_raw_vs_suppressed(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_suppressed: np.ndarray,
    threshold: float = None,
) -> Dict[str, Any]:
    """Compare raw vs identity-suppressed predictions."""
    if threshold is None:
        # Use threshold from suppressed (more conservative)
        threshold, _ = find_mcc_optimal_threshold(y_true, y_prob_suppressed)

    raw_metrics = evaluate_predictions(y_true, y_prob_raw, threshold)
    suppressed_metrics = evaluate_predictions(y_true, y_prob_suppressed, threshold)

    # Calculate deltas
    deltas = {}
    for key in ["auc", "pr_auc", "brier_score", "ece", "operating_point"]:
        if key in raw_metrics and key in suppressed_metrics:
            if key == "operating_point":
                deltas[f"delta_mcc"] = (
                    raw_metrics[key]["mcc"] - suppressed_metrics[key]["mcc"]
                )
            else:
                deltas[f"delta_{key}"] = raw_metrics[key] - suppressed_metrics[key]

    return {
        "raw": raw_metrics,
        "suppressed": suppressed_metrics,
        "deltas": deltas,
        "threshold_used": threshold,
    }


def check_leakage_safety(
    train_indices: np.ndarray, val_indices: np.ndarray, case_ids: np.ndarray
) -> Dict[str, Any]:
    """Check for potential leakage between train and validation sets."""
    train_cases = set(case_ids[train_indices])
    val_cases = set(case_ids[val_indices])

    # Check for case overlap
    case_overlap = train_cases.intersection(val_cases)

    # Check for temporal ordering (assuming case_ids have temporal component)
    # This is a simplified check - in practice you'd want more sophisticated temporal validation

    results = {
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "case_overlap_count": len(case_overlap),
        "case_overlap_pct": len(case_overlap) / len(val_cases) if val_cases else 0,
        "no_case_leakage": len(case_overlap) == 0,
    }

    if case_overlap:
        logger.warning(
            f"Found {len(case_overlap)} overlapping cases between train and validation"
        )
        results["overlapping_cases"] = list(case_overlap)

    return results
