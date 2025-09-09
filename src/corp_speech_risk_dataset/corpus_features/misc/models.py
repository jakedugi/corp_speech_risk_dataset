"""Enhanced interpretable models for legal risk classification.

This module provides state-of-the-art interpretable models that maintain
full transparency while achieving competitive performance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from loguru import logger

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.svm import LinearSVC

try:
    from mord import OrdinalRidge

    HAS_MORD = True
except ImportError:
    logger.warning("mord not available, using fallback ordinal implementation")
    HAS_MORD = False

# Import our weighted ordinal regression
from .weighted_ordinal import WeightedOrdinalRegression

try:
    from interpret.glassbox import ExplainableBoostingClassifier

    HAS_INTERPRET = True
except ImportError:
    logger.warning("interpret not available, EBM models disabled")
    HAS_INTERPRET = False


class MultinomialLogisticRegression(BaseEstimator, ClassifierMixin):
    """Multinomial Logistic Regression for multiclass classification.

    Unlike POLR, this doesn't assume proportional odds, allowing each feature
    to have different effects across class boundaries. Fully interpretable
    with coefficient interpretation per class.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        class_weight: Optional[Union[str, Dict]] = None,
        multi_class: str = "auto",  # Use auto to avoid deprecation warning
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.class_weight = class_weight
        self.multi_class = multi_class

    def fit(self, X, y, sample_weight=None):
        """Fit multinomial logistic regression."""
        X, y = check_X_y(X, y)

        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Fit multinomial logistic regression
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            class_weight=self.class_weight,
            multi_class=self.multi_class,
        )

        self.model_.fit(X, y, sample_weight=sample_weight)

        # Store coefficients for interpretation
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_

        return self

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self)
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self)
        return self.model_.predict_proba(X)

    def get_cumulative_probs(self, X):
        """Get cumulative probabilities for compatibility with POLR pipeline.

        For multinomial LR, we convert class probabilities to cumulative form:
        P(Y <= 0) = P(Y = 0)
        P(Y <= 1) = P(Y = 0) + P(Y = 1)
        """
        probs = self.predict_proba(X)

        # Ensure we have 3 classes (pad with zeros if needed)
        if probs.shape[1] < 3:
            padded_probs = np.zeros((probs.shape[0], 3))
            padded_probs[:, : probs.shape[1]] = probs
            probs = padded_probs

        # Convert to cumulative probabilities
        cum_probs = np.zeros((probs.shape[0], 2))
        cum_probs[:, 0] = probs[:, 0]  # P(Y <= 0)
        cum_probs[:, 1] = probs[:, 0] + probs[:, 1]  # P(Y <= 1)

        return cum_probs

    def get_feature_importance(self, feature_names=None):
        """Get feature importance based on coefficient magnitudes."""
        check_is_fitted(self)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.coef_.shape[1])]

        # Calculate importance as mean absolute coefficient across classes
        importance = np.mean(np.abs(self.coef_), axis=0)

        return dict(zip(feature_names, importance))


class ProportionalOddsLogisticRegression(BaseEstimator, ClassifierMixin):
    """Proportional Odds Ordinal Logistic Regression.

    This is the gold standard for interpretable ordinal classification.
    Maintains ordering assumption: P(Y≤j|X) follows cumulative logistic.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the proportional odds model."""
        X, y = check_X_y(X, y, accept_sparse=True)

        if HAS_MORD:
            # Use mord's OrdinalRidge as proxy (similar formulation)
            # OrdinalRidge doesn't support normalize parameter in newer sklearn versions
            self.model_ = OrdinalRidge(
                alpha=1.0 / self.C,
                fit_intercept=True,
                copy_X=True,
                max_iter=self.max_iter,
                tol=1e-4,
                solver="auto",
            )
            self.model_.fit(X, y)
            # OrdinalRidge uses unique_y_ instead of classes_
            self.classes_ = np.sort(self.model_.unique_y_)
            self.coef_ = self.model_.coef_
            # OrdinalRidge doesn't have theta_, we'll compute thresholds from intercepts
            self.theta_ = None  # Not available in mord
            self.n_classes_ = len(self.classes_)
        else:
            # Fallback: One-vs-Rest with ordinal constraints
            self._fit_ovr_ordinal(X, y)

        return self

    def _fit_ovr_ordinal(self, X, y):
        """Fallback ordinal fitting using OneVsRest approach."""
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)

        # Fit binary classifiers for P(Y >= k)
        self.estimators_ = []
        self.thresholds_ = []

        for k in range(1, n_classes):
            y_binary = (y_encoded >= k).astype(int)
            clf = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            clf.fit(X, y_binary)
            self.estimators_.append(clf)
            self.thresholds_.append(clf.intercept_[0])

        # Average coefficients for interpretability
        self.coef_ = np.mean([e.coef_ for e in self.estimators_], axis=0)

    def predict_proba(self, X):
        """Predict class probabilities maintaining ordinal structure."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        if HAS_MORD and hasattr(self, "model_"):
            # OrdinalRidge doesn't have predict_proba, so we implement it
            # Based on the ordinal regression model: P(Y=k) = P(Y<=k) - P(Y<=k-1)

            # Get the linear predictions
            decision = X @ self.model_.coef_.T
            n_samples = X.shape[0]
            n_classes = len(self.classes_)

            # For ordinal regression, we need thresholds
            # Since mord doesn't expose them directly, we'll use a simple approach
            # This assumes classes are 0, 1, 2 (or similar ordered values)

            # Create cumulative probabilities using logistic function
            # We'll use uniform thresholds as a simple approximation
            thresholds = np.linspace(decision.min() - 1, decision.max() + 1, n_classes)

            cumulative_probs = np.zeros((n_samples, n_classes))
            for i in range(n_classes - 1):
                cumulative_probs[:, i] = 1 / (
                    1 + np.exp(-(thresholds[i] - decision.ravel()))
                )
            cumulative_probs[:, -1] = 1.0

            # Convert cumulative to class probabilities
            probs = np.zeros((n_samples, n_classes))
            probs[:, 0] = cumulative_probs[:, 0]
            for i in range(1, n_classes):
                probs[:, i] = cumulative_probs[:, i] - cumulative_probs[:, i - 1]

            # Ensure probabilities are valid
            probs = np.maximum(probs, 0)
            probs = probs / probs.sum(axis=1, keepdims=True)

            return probs
        else:
            # Compute cumulative probabilities
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            cumulative_probs = np.zeros((n_samples, n_classes + 1))
            cumulative_probs[:, 0] = 1.0
            cumulative_probs[:, -1] = 0.0

            for k, clf in enumerate(self.estimators_):
                cumulative_probs[:, k + 1] = 1 - clf.predict_proba(X)[:, 1]

            # Convert to class probabilities
            probs = np.diff(cumulative_probs, axis=1) * -1
            return np.maximum(probs, 0)  # Ensure non-negative

    def predict(self, X):
        """Predict ordinal classes."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_feature_importance(self) -> np.ndarray:
        """Get interpretable feature coefficients."""
        check_is_fitted(self)
        return self.coef_.ravel()


class POLR(BaseEstimator, ClassifierMixin):
    """POLR: Proportional Odds Logistic Regression.

    TRUE proportional odds logistic regression for ordinal classification.
    ALWAYS uses logistic regression (never Ridge regression).
    Produces interpretable odds ratios for legal domain.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit the POLR model - ALWAYS uses logistic regression.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Sample weights (always supported)

        Returns:
            self
        """
        # ALWAYS use logistic regression, never Ridge
        logger.info("Fitting POLR (true logistic regression)")
        self.model_ = WeightedOrdinalRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        # Fit with or without weights - the model handles both
        self.model_.fit(X, y, sample_weight=sample_weight)

        # Copy attributes
        self.classes_ = self.model_.classes_
        self.n_classes_ = self.model_.n_classes_
        self.coef_ = self.model_.coef_
        self.estimators_ = self.model_.estimators_

        return self

    def predict_proba(self, X):
        """Predict class probabilities using logistic model."""
        check_is_fitted(self)
        return self.model_.predict_proba(X)

    def predict(self, X):
        """Predict ordinal classes."""
        check_is_fitted(self)
        return self.model_.predict(X)

    def get_cumulative_probs(self, X) -> np.ndarray:
        """Get cumulative probabilities P(Y <= k) for calibration."""
        check_is_fitted(self)
        return self.model_.get_cumulative_probs(X)

    def get_odds_ratios(self) -> np.ndarray:
        """Get interpretable odds ratios from logistic coefficients.

        Returns:
            exp(coefficients) = odds ratios
        """
        check_is_fitted(self)
        return np.exp(self.coef_)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature coefficients (log odds ratios)."""
        check_is_fitted(self)
        return self.coef_.ravel()


# =============================================================
# Binary Logistic Regression variants (interpretable)
# =============================================================


class LogisticRegressionL2(BaseEstimator, ClassifierMixin):
    """Binary Logistic Regression with L2 penalty.

    Fully interpretable via coefficients. Supports sample_weight.
    """

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        clf = LogisticRegression(
            penalty="l2",
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        clf.fit(X, y, sample_weight=sample_weight)
        self.model_ = clf
        self.coef_ = clf.coef_
        self.intercept_ = clf.intercept_
        self.classes_ = clf.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.model_.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        check_is_fitted(self)
        return self.coef_.ravel()


class LogisticRegressionL1(BaseEstimator, ClassifierMixin):
    """Binary Logistic Regression with L1 penalty (sparse, interpretable)."""

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "liblinear",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        clf = LogisticRegression(
            penalty="l1",
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        clf.fit(X, y, sample_weight=sample_weight)
        self.model_ = clf
        self.coef_ = clf.coef_
        self.intercept_ = clf.intercept_
        self.classes_ = clf.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.model_.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        check_is_fitted(self)
        return self.coef_.ravel()


class LogisticRegressionElasticNet(BaseEstimator, ClassifierMixin):
    """Binary Logistic Regression with ElasticNet penalty (L1/L2 mix)."""

    def __init__(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
        solver: str = "saga",
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        clf = LogisticRegression(
            penalty="elasticnet",
            C=self.C,
            l1_ratio=self.l1_ratio,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        clf.fit(X, y, sample_weight=sample_weight)
        self.model_ = clf
        self.coef_ = clf.coef_
        self.intercept_ = clf.intercept_
        self.classes_ = clf.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.model_.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        check_is_fitted(self)
        return self.coef_.ravel()


class LinearSVMClassifier(BaseEstimator, ClassifierMixin):
    """Linear SVM (LinearSVC) wrapper with sample_weight support.

    Notes:
    - LinearSVC does not natively provide predict_proba; downstream calibration
      should use decision_function outputs.
    - This wrapper exposes decision_function and standard predict methods.
    """

    def __init__(
        self,
        C: float = 1.0,
        loss: str = "squared_hinge",
        tol: float = 1e-4,
        max_iter: int = 5000,
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.loss = loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.model_ = LinearSVC(
            C=self.C,
            loss=self.loss,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.unique(y)
        # LinearSVC exposes coef_ and intercept_ after fit
        self.coef_ = getattr(self.model_, "coef_", None)
        self.intercept_ = getattr(self.model_, "intercept_", None)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict(X)

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.decision_function(X)

    def get_feature_importance(self, feature_names=None):
        check_is_fitted(self)
        if self.coef_ is None:
            return np.zeros((X.shape[1],), dtype=float)  # type: ignore[name-defined]
        return self.coef_.ravel()


# Keep POLAR for backward compatibility but mark as deprecated
class POLAR(POLR):
    """DEPRECATED: Use POLR instead. This is kept for backward compatibility."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "POLAR is deprecated. Use POLR for true proportional odds logistic regression."
        )
        super().__init__(*args, **kwargs)


class CalibratedInterpretableClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for calibrated probability predictions.

    Ensures well-calibrated probabilities for any interpretable model.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        method: str = "isotonic",
        cv: int = 3,
    ):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        """Fit base model with calibration."""
        # Handle scikit-learn version compatibility
        try:
            # Newer scikit-learn versions use 'estimator'
            self.calibrated_clf_ = CalibratedClassifierCV(
                estimator=self.base_estimator,
                method=self.method,
                cv=self.cv,
            )
        except TypeError:
            # Older versions use 'base_estimator'
            self.calibrated_clf_ = CalibratedClassifierCV(
                base_estimator=self.base_estimator,
                method=self.method,
                cv=self.cv,
            )
        self.calibrated_clf_.fit(X, y)
        self.classes_ = self.calibrated_clf_.classes_
        return self

    def predict(self, X):
        """Predict with calibrated model."""
        check_is_fitted(self)
        return self.calibrated_clf_.predict(X)

    def predict_proba(self, X):
        """Get calibrated probabilities."""
        check_is_fitted(self)
        return self.calibrated_clf_.predict_proba(X)

    def get_base_estimator(self):
        """Access the uncalibrated base estimator for interpretation."""
        check_is_fitted(self)
        # Get the first calibrated classifier's base estimator
        return self.calibrated_clf_.calibrated_classifiers_[0].base_estimator


def create_ebm_classifier(
    max_bins: int = 256,
    max_interaction_bins: int = 32,
    interactions: int = 0,
    outer_bags: int = 8,
    inner_bags: int = 0,
    learning_rate: float = 0.01,
    max_rounds: int = 5000,
    early_stopping_rounds: int = 50,
    random_state: Optional[int] = None,
) -> Optional[ExplainableBoostingClassifier]:
    """Create an Explainable Boosting Machine if available."""
    if not HAS_INTERPRET:
        logger.warning("interpret package not available, returning None")
        return None

    return ExplainableBoostingClassifier(
        max_bins=max_bins,
        max_interaction_bins=max_interaction_bins,
        interactions=interactions,
        outer_bags=outer_bags,
        inner_bags=inner_bags,
        learning_rate=learning_rate,
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )


class TransparentEnsemble(BaseEstimator, ClassifierMixin):
    """Transparent ensemble using interpretable voting or stacking.

    All decisions are fully auditable and explainable.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        """Fit all base estimators."""
        from sklearn.ensemble import VotingClassifier

        self.ensemble_ = VotingClassifier(
            estimators=self.estimators,
            voting=self.voting,
            weights=self.weights,
        )
        self.ensemble_.fit(X, y)
        self.classes_ = self.ensemble_.classes_
        return self

    def predict(self, X):
        """Ensemble prediction."""
        check_is_fitted(self)
        return self.ensemble_.predict(X)

    def predict_proba(self, X):
        """Ensemble probabilities."""
        check_is_fitted(self)
        return self.ensemble_.predict_proba(X)

    def get_estimator_predictions(self, X) -> Dict[str, np.ndarray]:
        """Get individual estimator predictions for transparency."""
        check_is_fitted(self)
        predictions = {}
        for name, estimator in self.ensemble_.estimators_:
            predictions[name] = estimator.predict_proba(X)
        return predictions


# Aliases for backwards compatibility
# Note: POLR class (line 283) already supports sample_weight, ProportionalOddsLogisticRegression doesn't
# Using the real POLR class that supports sample weights
MLR = MultinomialLogisticRegression
LR_L2 = LogisticRegressionL2
LR_L1 = LogisticRegressionL1
LR_ElasticNet = LogisticRegressionElasticNet
SVM_Linear = LinearSVMClassifier
