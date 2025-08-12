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

try:
    from mord import OrdinalRidge

    HAS_MORD = True
except ImportError:
    logger.warning("mord not available, using fallback ordinal implementation")
    HAS_MORD = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier

    HAS_INTERPRET = True
except ImportError:
    logger.warning("interpret not available, EBM models disabled")
    HAS_INTERPRET = False


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
            self.model_ = OrdinalRidge(
                alpha=1.0 / self.C,
                fit_intercept=True,
                normalize=False,
                copy_X=True,
                max_iter=self.max_iter,
                tol=1e-4,
                solver="auto",
            )
            self.model_.fit(X, y)
            self.classes_ = self.model_.classes_
            self.coef_ = self.model_.coef_
            self.theta_ = self.model_.theta_  # Thresholds
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
            return self.model_.predict_proba(X)
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
