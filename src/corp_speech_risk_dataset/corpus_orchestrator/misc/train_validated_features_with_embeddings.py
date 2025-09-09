#!/usr/bin/env python3
"""
Extended Comprehensive Model Training with Legal-BERT Embeddings

This script extends the original comprehensive model training to include 6 new variants:
- 3 models using ONLY legal_bert_emb (E)
- 3 models using legal_bert_emb + 3 validated scalars (E+3)

Original 3 validated features:
- new4_neutral_to_disclaimer_transition_rate
- lex_disclaimers_present
- feat_interact_hedge_x_guarantee

New variants:
1. L2-LR with embeddings only
2. L1-LR with embeddings only
3. Elastic-Net LR with embeddings only
4. L2-LR with embeddings + 3 scalars
5. L1-LR with embeddings + 3 scalars
6. Elastic-Net LR with embeddings + 3 scalars
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Sklearn imports
from sklearn.model_selection import StratifiedKFold, GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy import stats
from sklearn.base import clone
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import mord for POLR
try:
    import mord

    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False
    print("Warning: mord not available, POLR will be skipped")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Feature configurations
VALIDATED_SCALARS = [
    "new4_neutral_to_disclaimer_transition_rate",
    "lex_disclaimers_present",
    "feat_interact_hedge_x_guarantee",
]

EMBEDDING_FEATURE = "legal_bert_emb"  # 768-dimensional Legal-BERT embeddings


class ExtendedModelEvaluator:
    """Extended model training and evaluation with Legal-BERT embeddings."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        fold: int = 4,
        use_grid_search: bool = False,
        scout_pass: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fold = fold
        self.use_grid_search = use_grid_search
        self.scout_pass = scout_pass  # Use reduced grids for initial screening
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.feature_importance = {}
        self.scout_results = {}  # Store scout pass results for promotion
        self.train_df = None  # Store for access in define_models

        logger.info(f"Initialized extended evaluator for fold {fold}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Grid search: {use_grid_search}")
        logger.info(f"Scout pass: {scout_pass}")

        # Add helper functions for additional metrics
        self.optimal_thresholds = {}  # Store MCC-optimal thresholds per model
        self.court_means = {}  # Store court-level means for suppression
        self.state_means = {}  # Store state-level means for suppression fallback
        self.global_mean = None  # Store global mean for final fallback

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/dev/test data for the specified fold."""
        logger.info("Loading data...")

        # Load datasets with orjson optimization
        train_path = self.data_dir / f"fold_{self.fold}" / "train.jsonl"
        dev_path = self.data_dir / f"fold_{self.fold}" / "dev.jsonl"

        # Try oof_test first, fallback to fold test
        test_path = self.data_dir / "oof_test" / "test.jsonl"
        if not test_path.exists():
            test_path = self.data_dir / f"fold_{self.fold}" / "test.jsonl"
            if not test_path.exists():
                # Create dummy test set from dev
                test_path = dev_path
                logger.warning("No test set found, using dev set for testing")

        # Load with pandas (handles JSONL)
        train_df = pd.read_json(train_path, lines=True)
        dev_df = pd.read_json(dev_path, lines=True)
        test_df = pd.read_json(test_path, lines=True)

        logger.info(
            f"Loaded: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}"
        )

        # Combine train and dev for full training set
        full_train_df = pd.concat([train_df, dev_df], ignore_index=True)

        # Check binary labels
        for name, df in [("train", full_train_df), ("test", test_df)]:
            unique_labels = df["outcome_bin"].unique()
            logger.info(f"{name} unique labels: {sorted(unique_labels)}")

            dist = df["outcome_bin"].value_counts(normalize=True).sort_index()
            logger.info(f"{name} label distribution: {dict(dist)}")

        # Verify required features exist
        required_features = VALIDATED_SCALARS + [EMBEDDING_FEATURE]
        missing_features = []

        for feature in required_features:
            if feature not in full_train_df.columns:
                missing_features.append(feature)

        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise ValueError(f"Required features not found: {missing_features}")

        # Check embedding dimensions
        sample_emb = full_train_df[EMBEDDING_FEATURE].iloc[0]
        if isinstance(sample_emb, list):
            emb_dim = len(sample_emb)
            logger.info(f"Legal-BERT embedding dimension: {emb_dim}")
            if emb_dim != 768:
                logger.warning(f"Expected 768-dim embeddings, got {emb_dim}")
        else:
            logger.error(
                f"Embedding feature {EMBEDDING_FEATURE} is not a list: {type(sample_emb)}"
            )
            raise ValueError(f"Invalid embedding format for {EMBEDDING_FEATURE}")

        return full_train_df, test_df

    def prepare_features(
        self, df: pd.DataFrame, feature_config: str, scaler=None, fit=False
    ) -> Tuple[np.ndarray, StandardScaler]:
        """Prepare features based on configuration.

        Args:
            df: Input dataframe
            feature_config: 'E' for embeddings only, 'E+3' for embeddings + 3 scalars
            scaler: Pre-fitted scaler (for test data)
            fit: Whether to fit new scaler (for train data)
        """
        logger.info(f"Preparing features for config: {feature_config}")

        features = []
        feature_names = []

        # Always include embeddings
        embeddings = np.array(df[EMBEDDING_FEATURE].tolist())
        features.append(embeddings)
        feature_names.extend([f"emb_{i}" for i in range(embeddings.shape[1])])

        # Add scalars if E+3 configuration
        if feature_config == "E+3":
            for scalar_feature in VALIDATED_SCALARS:
                scalar_values = df[scalar_feature].values.reshape(-1, 1)
                features.append(scalar_values)
                feature_names.append(scalar_feature)

        # Combine all features
        X = np.hstack(features)

        # Scale features
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info(
                f"Fitted scaler for {feature_config}: {X.shape} -> {X_scaled.shape}"
            )
        else:
            if scaler is None:
                raise ValueError("Scaler required when fit=False")
            X_scaled = scaler.transform(X)

        logger.info(f"Feature matrix shape: {X_scaled.shape}")
        logger.info(f"Feature names: {len(feature_names)} features")

        return X_scaled, scaler, feature_names

    def prepare_identity_suppressed_features(
        self, df: pd.DataFrame, feature_config: str, scaler=None, fit=False
    ) -> Tuple[np.ndarray, StandardScaler, List[str]]:
        """Prepare identity-suppressed features (centered by case/era)."""
        logger.info(
            f"Preparing identity-suppressed features for config: {feature_config}"
        )

        features = []
        feature_names = []

        # Always include embeddings
        embeddings = np.array(df[EMBEDDING_FEATURE].tolist())

        # Center embeddings by case_id (identity suppression)
        if "case_id" in df.columns:
            case_means = (
                df.groupby("case_id")[EMBEDDING_FEATURE]
                .apply(lambda x: np.mean(np.array(x.tolist()), axis=0))
                .to_dict()
            )

            centered_embeddings = []
            for idx, row in df.iterrows():
                case_id = row["case_id"]
                original_emb = np.array(row[EMBEDDING_FEATURE])
                case_mean = case_means[case_id]
                centered_emb = original_emb - case_mean
                centered_embeddings.append(centered_emb)

            embeddings = np.array(centered_embeddings)
            logger.info(
                "Applied case-level centering to embeddings (identity suppression)"
            )

        features.append(embeddings)
        feature_names.extend([f"emb_centered_{i}" for i in range(embeddings.shape[1])])

        # Add scalars if E+3 configuration (also centered by case)
        if feature_config == "E+3":
            for scalar_feature in VALIDATED_SCALARS:
                if scalar_feature == "lex_disclaimers_present":
                    # Keep binary feature as 0/1, then optionally standardize
                    scalar_values = df[scalar_feature].values.reshape(-1, 1)
                else:
                    # Center by case for continuous features
                    if "case_id" in df.columns:
                        case_scalar_means = (
                            df.groupby("case_id")[scalar_feature].mean().to_dict()
                        )
                        centered_scalars = []
                        for idx, row in df.iterrows():
                            case_id = row["case_id"]
                            original_val = row[scalar_feature]
                            case_mean = case_scalar_means[case_id]
                            centered_val = original_val - case_mean
                            centered_scalars.append(centered_val)
                        scalar_values = np.array(centered_scalars).reshape(-1, 1)
                    else:
                        scalar_values = df[scalar_feature].values.reshape(-1, 1)

                features.append(scalar_values)
                feature_names.append(f"{scalar_feature}_centered")

        # Combine all features
        X = np.hstack(features)

        # Scale features
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info(
                f"Fitted scaler for identity-suppressed {feature_config}: {X.shape} -> {X_scaled.shape}"
            )
        else:
            if scaler is None:
                raise ValueError("Scaler required when fit=False")
            X_scaled = scaler.transform(X)

        logger.info(f"Identity-suppressed feature matrix shape: {X_scaled.shape}")
        return X_scaled, scaler, feature_names

    def calculate_ece(self, y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error."""
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

    def find_mcc_optimal_threshold(self, y_true, y_prob):
        """Find threshold that maximizes MCC."""
        thresholds = np.linspace(0.1, 0.9, 50)
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

    def calculate_operating_point_metrics(self, y_true, y_prob, threshold):
        """Calculate precision, recall, specificity at given threshold."""
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # Specificity = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

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

    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for all 13 model variants (scout vs full pass)."""

        if self.scout_pass:
            # Scout pass: reduced grids for initial screening
            # C ∈ {0.1, 1, 10} and l1_ratio ∈ {0.1, 0.9}
            scout_grids = {
                # A) 3 Validated Features only (7 variants)
                "lr_l2_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l1_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_elasticnet_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "svm_linear_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "loss": ["squared_hinge"],
                    "penalty": ["l2"],
                    "dual": [True],
                    "max_iter": [5000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "mlr_enhanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.9],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                    "estimator__class_weight": ["balanced"],
                },
                "mlr_balanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.9],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                    "estimator__class_weight": ["balanced"],
                },
                # B) Embedding variants (6 variants)
                "lr_l2_E": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l1_E": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E": {
                    "C": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l2_E+3": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l1_E+3": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E+3": {
                    "C": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
            }
            return scout_grids

        elif self.use_grid_search:
            # Full pass: complete grids as specified
            full_grids = {
                # A) 3 Validated Features only (7 variants)
                "lr_l2_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l1_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_elasticnet_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "svm_linear_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "loss": ["squared_hinge"],
                    "penalty": ["l2"],
                    "dual": [True],
                    "max_iter": [5000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "mlr_enhanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.5],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                    "estimator__class_weight": ["balanced"],
                },
                "mlr_balanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.5],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                    "estimator__class_weight": ["balanced"],
                },
                # B) Embedding variants (6 variants)
                "lr_l2_E": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l1_E": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l2_E+3": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_l1_E+3": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E+3": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "class_weight": ["balanced"],
                    "tol": [1e-4],
                },
            }
            return full_grids

        else:
            # Default: minimal grids for quick testing
            return {
                "lr_l2_E": {
                    "C": [1.0],
                    "solver": ["lbfgs"],
                    "class_weight": ["balanced"],
                },
                "lr_l1_E": {
                    "C": [1.0],
                    "solver": ["saga"],
                    "class_weight": ["balanced"],
                },
                "lr_elasticnet_E": {
                    "C": [1.0],
                    "l1_ratio": [0.5],
                    "solver": ["saga"],
                    "class_weight": ["balanced"],
                },
                "lr_l2_E+3": {
                    "C": [1.0],
                    "solver": ["lbfgs"],
                    "class_weight": ["balanced"],
                },
                "lr_l1_E+3": {
                    "C": [1.0],
                    "solver": ["saga"],
                    "class_weight": ["balanced"],
                },
                "lr_elasticnet_E+3": {
                    "C": [1.0],
                    "l1_ratio": [0.5],
                    "solver": ["saga"],
                    "class_weight": ["balanced"],
                },
            }

    def define_models(self) -> Dict[str, Any]:
        """Define all 13 model variants (3VF + embedding variants)."""
        models = {}
        grids = self.get_hyperparameter_grids()

        # Base models for all 13 variants
        base_models = {
            # A) 3 Validated Features only (7 variants)
            "lr_l2_3VF": LogisticRegression(penalty="l2", random_state=42),
            "lr_l1_3VF": LogisticRegression(penalty="l1", random_state=42),
            "lr_elasticnet_3VF": LogisticRegression(
                penalty="elasticnet", random_state=42
            ),
            "svm_linear_3VF": LinearSVC(random_state=42),
            "mlr_enhanced_3VF": OneVsRestClassifier(
                LogisticRegression(penalty="elasticnet", random_state=42)
            ),
            "mlr_balanced_3VF": OneVsRestClassifier(
                LogisticRegression(penalty="elasticnet", random_state=42)
            ),
            # B) Embedding variants (6 variants)
            "lr_l2_E": LogisticRegression(penalty="l2", random_state=42),
            "lr_l1_E": LogisticRegression(penalty="l1", random_state=42),
            "lr_elasticnet_E": LogisticRegression(
                penalty="elasticnet", random_state=42
            ),
            "lr_l2_E+3": LogisticRegression(penalty="l2", random_state=42),
            "lr_l1_E+3": LogisticRegression(penalty="l1", random_state=42),
            "lr_elasticnet_E+3": LogisticRegression(
                penalty="elasticnet", random_state=42
            ),
        }

        # Add POLR if available (optional 7th variant in 3VF block)
        if MORD_AVAILABLE:
            base_models["polr_3VF"] = mord.LogisticAT()
            if "polr_3VF" not in grids:
                grids["polr_3VF"] = {"alpha": [0.01, 0.1, 1.0, 10.0]}

        # Create GridSearchCV models if grid search enabled
        for model_name, base_model in base_models.items():
            if (self.use_grid_search or self.scout_pass) and model_name in grids:
                # Use GroupKFold for proper case-level CV
                cv_strategy = (
                    GroupKFold(n_splits=3)
                    if "case_id" in self.train_df.columns
                    else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                )

                # Create custom scorer that uses identity-suppressed MCC for selection
                def suppressed_mcc_scorer(estimator, X, y):
                    # This will be called during GridSearchCV
                    # For now, use raw AUC - we'll override with suppressed evaluation in main loop
                    if hasattr(estimator, "predict_proba"):
                        y_proba = estimator.predict_proba(X)[:, 1]
                        return (
                            roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5
                        )
                    else:
                        return 0.5

                models[model_name] = GridSearchCV(
                    base_model,
                    grids[model_name],
                    cv=cv_strategy,
                    scoring=suppressed_mcc_scorer,  # Will be overridden by manual evaluation
                    n_jobs=-1,
                    verbose=1,
                )
                logger.info(f"Created GridSearchCV model: {model_name}")
            else:
                # Use default parameters for quick testing
                models[model_name] = base_model
                logger.info(f"Created default model: {model_name}")

        logger.info(f"Defined {len(models)} models: {list(models.keys())}")
        return models

    def evaluate_model_cv(
        self, model, X: np.ndarray, y: np.ndarray, case_ids: np.ndarray, model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model using stratified cross-validation."""
        logger.info(f"Evaluating {model_name} with CV...")

        # Use StratifiedKFold
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_scores = []
        oof_preds = np.full(len(y), np.nan)
        oof_proba = np.full(len(y), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_tr, y_tr)

            # Get predictions
            if hasattr(model_clone, "predict_proba"):
                proba = model_clone.predict_proba(X_val)
                if proba.shape[1] == 2:
                    val_proba = proba[:, 1]
                else:
                    val_proba = proba[:, 0]
            elif hasattr(model_clone, "decision_function"):
                val_proba = expit(model_clone.decision_function(X_val))
            else:
                val_proba = model_clone.predict(X_val).astype(float)

            # Store OOF predictions
            oof_preds[val_idx] = model_clone.predict(X_val)
            oof_proba[val_idx] = val_proba

            # Calculate fold AUC
            if len(np.unique(y_val)) > 1:
                fold_auc = roc_auc_score(y_val, val_proba)
                cv_scores.append(fold_auc)
                logger.info(f"  Fold {fold_idx+1}: AUC = {fold_auc:.4f}")

        # Calculate overall CV metrics
        mask = ~np.isnan(oof_proba)

        # Primary metrics (domain-standard evaluation suite)
        oof_auc = (
            roc_auc_score(y[mask], oof_proba[mask])
            if len(np.unique(y[mask])) > 1
            else 0.5
        )
        oof_pr_auc = average_precision_score(y[mask], oof_proba[mask])

        # Find MCC-optimal threshold and calculate operating point metrics
        optimal_threshold, max_mcc = self.find_mcc_optimal_threshold(
            y[mask], oof_proba[mask]
        )
        self.optimal_thresholds[model_name] = optimal_threshold

        operating_metrics = self.calculate_operating_point_metrics(
            y[mask], oof_proba[mask], optimal_threshold
        )

        # Calibration metrics
        brier_score = brier_score_loss(y[mask], oof_proba[mask])
        ece = self.calculate_ece(y[mask], oof_proba[mask])

        # Store OOF predictions
        self.oof_predictions[model_name] = {
            "predictions": oof_preds,
            "probabilities": oof_proba,
            "true_labels": y,
        }

        return {
            "cv_auc_mean": np.mean(cv_scores),
            "cv_auc_std": np.std(cv_scores),
            "cv_auc_scores": cv_scores,
            # Primary domain-standard metrics
            "oof_mcc": operating_metrics["mcc"],  # PRIMARY METRIC
            "oof_auc": oof_auc,  # Secondary
            "oof_pr_auc": oof_pr_auc,  # Complements AUROC under imbalance
            "brier_score": brier_score,  # Calibration quality
            "ece": ece,  # Calibration quality
            # Operating point metrics (at MCC-optimal threshold)
            "optimal_threshold": optimal_threshold,
            "precision": operating_metrics["precision"],
            "recall": operating_metrics["recall"],
            "specificity": operating_metrics["specificity"],
            "confusion_matrix": operating_metrics["confusion_matrix"],
        }

    def evaluate_model_test(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """Evaluate model on test set."""
        logger.info(f"Evaluating {model_name} on test set...")

        # Fit on full training data with balanced class weights
        if hasattr(model, "set_params"):
            try:
                model.set_params(class_weight="balanced")
            except:
                pass  # Some models don't support class_weight

        model.fit(X_train, y_train)

        # Apply calibration (Platt scaling)
        try:
            calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            calibrated_model.fit(X_train, y_train)
            model = calibrated_model
            logger.info(f"Applied Platt calibration for {model_name}")
        except Exception as e:
            logger.warning(f"Calibration failed for {model_name}: {e}")
            pass

        # Test predictions
        test_preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            test_proba = model.predict_proba(X_test)
            if test_proba.shape[1] == 2:
                test_proba = test_proba[:, 1]
            else:
                test_proba = test_proba[:, 0]
        elif hasattr(model, "decision_function"):
            test_proba = expit(model.decision_function(X_test))
        else:
            test_proba = test_preds.astype(float)

        # Store test predictions
        self.test_predictions[model_name] = {
            "predictions": test_preds,
            "probabilities": test_proba,
            "true_labels": y_test,
        }

        # Use MCC-optimal threshold from CV for test evaluation
        optimal_threshold = self.optimal_thresholds.get(model_name, 0.5)
        test_operating_metrics = self.calculate_operating_point_metrics(
            y_test, test_proba, optimal_threshold
        )

        # Calculate primary domain-standard metrics
        test_auc = (
            roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5
        )
        test_pr_auc = average_precision_score(y_test, test_proba)
        test_brier = brier_score_loss(y_test, test_proba)
        test_ece = self.calculate_ece(y_test, test_proba)

        # Extract best parameters if GridSearchCV
        best_params = {}
        if hasattr(model, "best_params_"):
            best_params = model.best_params_
            logger.info(f"Best params for {model_name}: {best_params}")

        return {
            # Primary domain-standard metrics
            "test_mcc": test_operating_metrics["mcc"],  # PRIMARY
            "test_auc": test_auc,  # Secondary
            "test_pr_auc": test_pr_auc,  # Complements AUROC
            "test_brier_score": test_brier,  # Calibration
            "test_ece": test_ece,  # Calibration
            # Operating point metrics (at MCC-optimal threshold)
            "test_precision": test_operating_metrics["precision"],
            "test_recall": test_operating_metrics["recall"],
            "test_specificity": test_operating_metrics["specificity"],
            "test_confusion_matrix": test_operating_metrics["confusion_matrix"],
            "test_threshold_used": optimal_threshold,
            "best_params": best_params,
        }

    def evaluate_model_test_with_suppression(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        court_ids_train: np.ndarray,
        court_ids_test: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """Evaluate model on test set with both raw and identity-suppressed views."""
        logger.info(f"Evaluating {model_name} on test set (raw + suppressed)...")

        # Fit on full training data (RAW features only)
        if hasattr(model, "set_params"):
            try:
                model.set_params(class_weight="balanced")
            except:
                pass  # Some models don't support class_weight

        model.fit(X_train, y_train)

        # Apply calibration (Platt scaling) on raw training data
        try:
            calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            calibrated_model.fit(X_train, y_train)
            model = calibrated_model
            logger.info(f"Applied Platt calibration for {model_name}")
        except Exception as e:
            logger.warning(f"Calibration failed for {model_name}: {e}")
            pass

        # Compute suppression means on training data
        self.compute_suppression_means(X_train, court_ids_train, fit_global=True)

        # RAW test evaluation
        test_preds_raw = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            test_proba_raw = model.predict_proba(X_test)
            if test_proba_raw.shape[1] == 2:
                test_proba_raw = test_proba_raw[:, 1]
            else:
                test_proba_raw = test_proba_raw[:, 0]
        elif hasattr(model, "decision_function"):
            test_proba_raw = expit(model.decision_function(X_test))
        else:
            test_proba_raw = test_preds_raw.astype(float)

        # SUPPRESSED test evaluation (same model, suppressed features)
        X_test_suppressed = self.apply_identity_suppression(X_test, court_ids_test)

        if hasattr(model, "predict_proba"):
            test_proba_suppressed = model.predict_proba(X_test_suppressed)
            if test_proba_suppressed.shape[1] == 2:
                test_proba_suppressed = test_proba_suppressed[:, 1]
            else:
                test_proba_suppressed = test_proba_suppressed[:, 0]
        elif hasattr(model, "decision_function"):
            test_proba_suppressed = expit(model.decision_function(X_test_suppressed))
        else:
            test_proba_suppressed = test_proba_raw  # Fallback

        # Use threshold from CV suppressed evaluation
        optimal_threshold = self.optimal_thresholds.get(model_name, 0.5)

        # Calculate metrics for both views
        # RAW metrics
        test_operating_metrics_raw = self.calculate_operating_point_metrics(
            y_test, test_proba_raw, optimal_threshold
        )
        test_auc_raw = (
            roc_auc_score(y_test, test_proba_raw) if len(np.unique(y_test)) > 1 else 0.5
        )
        test_pr_auc_raw = average_precision_score(y_test, test_proba_raw)
        test_brier_raw = brier_score_loss(y_test, test_proba_raw)
        test_ece_raw = self.calculate_ece(y_test, test_proba_raw)

        # SUPPRESSED metrics
        test_operating_metrics_suppressed = self.calculate_operating_point_metrics(
            y_test, test_proba_suppressed, optimal_threshold
        )
        test_auc_suppressed = (
            roc_auc_score(y_test, test_proba_suppressed)
            if len(np.unique(y_test)) > 1
            else 0.5
        )
        test_pr_auc_suppressed = average_precision_score(y_test, test_proba_suppressed)
        test_brier_suppressed = brier_score_loss(y_test, test_proba_suppressed)
        test_ece_suppressed = self.calculate_ece(y_test, test_proba_suppressed)

        # Delta calculations
        delta_mcc_test = (
            test_operating_metrics_raw["mcc"] - test_operating_metrics_suppressed["mcc"]
        )
        delta_auc_test = test_auc_raw - test_auc_suppressed

        # Store test predictions
        self.test_predictions[model_name] = {
            "predictions": test_preds_raw,
            "probabilities_raw": test_proba_raw,
            "probabilities_suppressed": test_proba_suppressed,
            "true_labels": y_test,
        }

        # Extract best parameters if GridSearchCV
        best_params = {}
        if hasattr(model, "best_params_"):
            best_params = model.best_params_
            logger.info(f"Best params for {model_name}: {best_params}")

        logger.info(
            f"  Test RAW: MCC={test_operating_metrics_raw['mcc']:.4f}, AUC={test_auc_raw:.4f}"
        )
        logger.info(
            f"  Test SUPPRESSED: MCC={test_operating_metrics_suppressed['mcc']:.4f}, AUC={test_auc_suppressed:.4f}"
        )

        return {
            # RAW test metrics
            "test_mcc_raw": test_operating_metrics_raw["mcc"],
            "test_auc_raw": test_auc_raw,
            "test_pr_auc_raw": test_pr_auc_raw,
            "test_brier_score_raw": test_brier_raw,
            "test_ece_raw": test_ece_raw,
            "test_precision_raw": test_operating_metrics_raw["precision"],
            "test_recall_raw": test_operating_metrics_raw["recall"],
            "test_specificity_raw": test_operating_metrics_raw["specificity"],
            "test_confusion_matrix_raw": test_operating_metrics_raw["confusion_matrix"],
            # SUPPRESSED test metrics
            "test_mcc_suppressed": test_operating_metrics_suppressed["mcc"],
            "test_auc_suppressed": test_auc_suppressed,
            "test_pr_auc_suppressed": test_pr_auc_suppressed,
            "test_brier_score_suppressed": test_brier_suppressed,
            "test_ece_suppressed": test_ece_suppressed,
            "test_precision_suppressed": test_operating_metrics_suppressed["precision"],
            "test_recall_suppressed": test_operating_metrics_suppressed["recall"],
            "test_specificity_suppressed": test_operating_metrics_suppressed[
                "specificity"
            ],
            "test_confusion_matrix_suppressed": test_operating_metrics_suppressed[
                "confusion_matrix"
            ],
            # Delta and comparison metrics
            "delta_mcc_test": delta_mcc_test,
            "delta_auc_test": delta_auc_test,
            "test_threshold_used": optimal_threshold,
            "best_params": best_params,
        }

    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on all model variants."""
        logger.info("=" * 80)
        logger.info("STARTING EXTENDED COMPREHENSIVE MODEL EVALUATION")
        logger.info("=" * 80)

        # Load data
        train_df, test_df = self.load_data()
        self.train_df = train_df  # Store for access in define_models

        # Define models
        models = self.define_models()

        # Feature configurations
        feature_configs = ["E", "E+3"]

        # Evaluate each model with each feature configuration
        for model_name, model in models.items():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"EVALUATING: {model_name}")
                logger.info(f"{'='*60}")

                # Determine feature configuration from model name
                if "3VF" in model_name:
                    feature_config = "3VF"  # 3 validated features only
                elif "E+3" in model_name:
                    feature_config = "E+3"  # embeddings + 3 validated features
                else:
                    feature_config = "E"  # embeddings only

                logger.info(f"Feature configuration: {feature_config}")

                # Prepare features (regular and identity-suppressed)
                if feature_config == "3VF":
                    # 3 validated features only
                    X_train = train_df[VALIDATED_SCALARS].values
                    X_test = test_df[VALIDATED_SCALARS].values
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    feature_names = VALIDATED_SCALARS
                else:
                    # Embedding variants
                    X_train, scaler, feature_names = self.prepare_features(
                        train_df, feature_config, fit=True
                    )
                    X_test, _, _ = self.prepare_features(
                        test_df, feature_config, scaler=scaler
                    )

                # Identity-suppressed features for generalization check
                if feature_config == "3VF":
                    # For 3VF, center by case
                    X_train_suppressed = X_train.copy()  # Will implement case centering
                    X_test_suppressed = X_test.copy()
                    scaler_suppressed = scaler
                    feature_names_suppressed = feature_names
                else:
                    X_train_suppressed, scaler_suppressed, feature_names_suppressed = (
                        self.prepare_identity_suppressed_features(
                            train_df, feature_config, fit=True
                        )
                    )
                    X_test_suppressed, _, _ = self.prepare_identity_suppressed_features(
                        test_df, feature_config, scaler=scaler_suppressed
                    )

                y_train = train_df["outcome_bin"].values
                case_ids_train = (
                    train_df["case_id"].values
                    if "case_id" in train_df.columns
                    else np.arange(len(train_df))
                )
                court_ids_train = (
                    train_df["court_id"].values
                    if "court_id" in train_df.columns
                    else np.array(["unknown"] * len(train_df))
                )
                y_test = test_df["outcome_bin"].values
                court_ids_test = (
                    test_df["court_id"].values
                    if "court_id" in test_df.columns
                    else np.array(["unknown"] * len(test_df))
                )

                # Cross-validation evaluation with identity suppression
                cv_results = self.evaluate_model_cv(
                    model, X_train, y_train, case_ids_train, court_ids_train, model_name
                )

                # Test set evaluation with identity suppression
                test_results = self.evaluate_model_test_with_suppression(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    court_ids_train,
                    court_ids_test,
                    model_name,
                )

                # Extract delta metrics from test results
                delta_mcc = test_results.get("delta_mcc_test", 0.0)
                delta_auc = test_results.get("delta_auc_test", 0.0)

                # Selection criteria checks (using CV suppressed metrics for selection)
                cv_delta_mcc = cv_results.get("delta_mcc", 0.0)
                passes_delta_check = abs(cv_delta_mcc) <= 0.02  # Δ ≤ 0.02
                passes_ece_check = (
                    cv_results.get("ece_suppressed", 0.0) <= 0.08
                )  # ECE ≤ 0.08

                logger.info(
                    f"   CV Δ(MCC): {cv_delta_mcc:.4f} ({'PASS' if passes_delta_check else 'FAIL'} ≤ 0.02)"
                )
                logger.info(
                    f"   CV ECE (suppressed): {cv_results.get('ece_suppressed', 0.0):.4f} ({'PASS' if passes_ece_check else 'FAIL'} ≤ 0.08)"
                )
                logger.info(f"   Test Δ(MCC): {delta_mcc:.4f}")
                logger.info(f"   Test Δ(AUC): {delta_auc:.4f}")

                # Store combined results
                self.results[model_name] = {
                    **cv_results,
                    **test_results,
                    "feature_config": feature_config,
                    "n_features": len(feature_names),
                    "model_type": type(model).__name__,
                    # Selection criteria (based on CV suppressed metrics)
                    "cv_delta_mcc": cv_delta_mcc,
                    "test_delta_mcc": delta_mcc,
                    "test_delta_auc": delta_auc,
                    "passes_delta_check": passes_delta_check,
                    "passes_ece_check": passes_ece_check,
                    "meets_selection_criteria": passes_delta_check and passes_ece_check,
                }

                # Extract feature importance if available
                if hasattr(model, "coef_"):
                    self.feature_importance[model_name] = model.coef_.flatten()
                elif hasattr(model, "best_estimator_") and hasattr(
                    model.best_estimator_, "coef_"
                ):
                    self.feature_importance[model_name] = (
                        model.best_estimator_.coef_.flatten()
                    )

                logger.info(f"✅ {model_name} evaluation completed")
                logger.info(
                    f"   CV MCC (SUPPRESSED): {cv_results['oof_mcc_suppressed']:.4f} (PRIMARY SELECTION)"
                )
                logger.info(
                    f"   CV AUC (RAW): {cv_results['cv_auc_raw_mean']:.4f} ± {cv_results['cv_auc_raw_std']:.4f}"
                )
                logger.info(
                    f"   CV AUC (SUPPRESSED): {cv_results['cv_auc_suppressed_mean']:.4f} ± {cv_results['cv_auc_suppressed_std']:.4f}"
                )
                logger.info(f"   Test MCC (RAW): {test_results['test_mcc_raw']:.4f}")
                logger.info(
                    f"   Test MCC (SUPPRESSED): {test_results['test_mcc_suppressed']:.4f}"
                )
                logger.info(
                    f"   Test ECE (SUPPRESSED): {test_results['test_ece_suppressed']:.4f}"
                )
                logger.info(
                    f"   Optimal threshold: {test_results['test_threshold_used']:.3f}"
                )

            except Exception as e:
                logger.error(f"❌ {model_name} evaluation failed: {e}")
                self.results[model_name] = {"error": str(e)}

        # Save results
        self.save_results()

        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE EVALUATION COMPLETED")
        logger.info("=" * 80)

    def save_results(self):
        """Save all results to files."""
        logger.info("Saving results...")

        # Save detailed results
        results_file = self.output_dir / "detailed_results.json"
        with open(results_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, result in self.results.items():
                serializable_results[model_name] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[model_name][key] = value.tolist()
                    elif isinstance(value, np.float64):
                        serializable_results[model_name][key] = float(value)
                    elif isinstance(value, np.int64):
                        serializable_results[model_name][key] = int(value)
                    else:
                        serializable_results[model_name][key] = value

            json.dump(serializable_results, f, indent=2)

        # Create summary table
        summary_rows = []
        for model_name, result in self.results.items():
            if "error" not in result:
                summary_rows.append(
                    {
                        "model": model_name,
                        "feature_config": result.get("feature_config", "unknown"),
                        "n_features": result.get("n_features", 0),
                        # Primary domain-standard metrics (SUPPRESSED view for selection)
                        "cv_mcc_suppressed": result.get(
                            "oof_mcc_suppressed", 0
                        ),  # 1. MCC (PRIMARY SELECTION)
                        "test_mcc_suppressed": result.get("test_mcc_suppressed", 0),
                        "cv_auc_suppressed": result.get(
                            "cv_auc_suppressed_mean", 0
                        ),  # 2. AUROC (suppressed)
                        "test_auc_suppressed": result.get("test_auc_suppressed", 0),
                        "test_pr_auc_suppressed": result.get(
                            "test_pr_auc_suppressed", 0
                        ),  # 3. PR-AUC (suppressed)
                        "test_brier_suppressed": result.get(
                            "test_brier_score_suppressed", 0
                        ),  # 4. Calibration (suppressed)
                        "test_ece_suppressed": result.get("test_ece_suppressed", 0),
                        # RAW metrics (for comparison)
                        "cv_mcc_raw": result.get("oof_mcc_raw", 0),
                        "test_mcc_raw": result.get("test_mcc_raw", 0),
                        "cv_auc_raw": result.get("cv_auc_raw_mean", 0),
                        "test_auc_raw": result.get("test_auc_raw", 0),
                        # Delta metrics
                        "cv_delta_mcc": result.get("cv_delta_mcc", 0),
                        "test_delta_mcc": result.get("test_delta_mcc", 0),
                        "test_delta_auc": result.get("test_delta_auc", 0),
                        # Operating point metrics (suppressed view)
                        "test_precision_suppressed": result.get(
                            "test_precision_suppressed", 0
                        ),
                        "test_recall_suppressed": result.get(
                            "test_recall_suppressed", 0
                        ),
                        "test_specificity_suppressed": result.get(
                            "test_specificity_suppressed", 0
                        ),
                        "optimal_threshold": result.get("test_threshold_used", 0.5),
                        # Selection criteria
                        "passes_criteria": result.get(
                            "meets_selection_criteria", False
                        ),
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = self.output_dir / "model_comparison.csv"
            summary_df.to_csv(summary_file, index=False)

            # Sort by CV suppressed MCC (PRIMARY SELECTION METRIC) for display
            summary_df_sorted = summary_df.sort_values(
                "cv_mcc_suppressed", ascending=False
            )

            logger.info("\n" + "=" * 100)
            logger.info("MODEL PERFORMANCE SUMMARY")
            logger.info("=" * 100)
            logger.info(summary_df_sorted.to_string(index=False, float_format="%.4f"))
            logger.info("=" * 100)

        logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extended model evaluation with Legal-BERT embeddings"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to k-fold data directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument("--fold", type=int, default=4, help="Fold number to use")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable full hyperparameter grid search",
    )
    parser.add_argument(
        "--scout-pass", action="store_true", help="Run scout pass with reduced grids"
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Run all 13 model variants (3VF + embeddings)",
    )

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = ExtendedModelEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold=args.fold,
        use_grid_search=args.grid_search,
        scout_pass=args.scout_pass,
    )

    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
