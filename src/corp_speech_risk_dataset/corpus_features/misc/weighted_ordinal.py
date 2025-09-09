"""Weighted ordinal regression implementation for POLAR."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from typing import Optional


class WeightedOrdinalRegression(BaseEstimator, ClassifierMixin):
    """Ordinal regression with sample weight support using cumulative link model."""

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
        """Fit ordinal regression with sample weights using cumulative logit approach."""
        X, y = check_X_y(X, y, accept_sparse=True)

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        # Fit K-1 binary classifiers for cumulative probabilities P(Y <= k)
        self.estimators_ = []
        self.coef_ = []

        for k in range(self.n_classes_ - 1):
            # Create binary target: 1 if y <= k, 0 otherwise
            y_binary = (y_encoded <= k).astype(int)

            # Fit weighted logistic regression
            clf = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                class_weight=None,  # We use sample_weight instead
            )
            clf.fit(X, y_binary, sample_weight=sample_weight)

            self.estimators_.append(clf)
            self.coef_.append(clf.coef_[0])

        # Average coefficients for interpretability
        self.coef_ = np.mean(self.coef_, axis=0)

        return self

    def predict_proba(self, X):
        """Predict class probabilities maintaining ordinal structure."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        n_samples = X.shape[0]

        # Get cumulative probabilities P(Y <= k)
        cumulative_probs = np.zeros((n_samples, self.n_classes_))

        for k, clf in enumerate(self.estimators_):
            cumulative_probs[:, k] = clf.predict_proba(X)[:, 1]

        # Add boundary condition P(Y <= K-1) = 1
        cumulative_probs[:, -1] = 1.0

        # Convert to class probabilities
        # P(Y = 0) = P(Y <= 0)
        # P(Y = k) = P(Y <= k) - P(Y <= k-1) for k > 0
        class_probs = np.zeros((n_samples, self.n_classes_))
        class_probs[:, 0] = cumulative_probs[:, 0]

        for k in range(1, self.n_classes_):
            class_probs[:, k] = cumulative_probs[:, k] - cumulative_probs[:, k - 1]

        # Ensure non-negative and normalized
        class_probs = np.maximum(class_probs, 0)
        class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)

        return class_probs

    def predict(self, X):
        """Predict ordinal classes."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_cumulative_probs(self, X):
        """Get cumulative probabilities P(Y <= k) for calibration."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        n_samples = X.shape[0]
        cumulative_probs = np.zeros((n_samples, self.n_classes_ - 1))

        for k, clf in enumerate(self.estimators_):
            cumulative_probs[:, k] = clf.predict_proba(X)[:, 1]

        return cumulative_probs
