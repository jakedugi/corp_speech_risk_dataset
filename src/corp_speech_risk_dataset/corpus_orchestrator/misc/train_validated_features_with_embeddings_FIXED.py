#!/usr/bin/env python3
"""
Extended Comprehensive Model Training with Legal-BERT Embeddings - FIXED VERSION

CRITICAL FIX: This version properly inherits the existing temporal case-wise k-fold structure
NO RE-SPLITTING! Uses existing train.jsonl/dev.jsonl/test.jsonl as-is.

Supports all 13 model variants with court-based identity suppression for selection.
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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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

# Fast JSON loading with orjson optimization
try:
    import orjson as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data)

    def _loads_str(data: str) -> Any:
        return _json.loads(data.encode("utf-8"))

except ImportError:
    import json as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data.decode("utf-8"))

    def _loads_str(data: str) -> Any:
        return _json.loads(data)


# MORD for ordinal regression (optional)
MORD_AVAILABLE = False
try:
    import mord

    MORD_AVAILABLE = True
except ImportError:
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Validated features (the 3 that passed fingerprinting)
VALIDATED_SCALARS = [
    "new4_neutral_to_disclaimer_transition_rate",
    "lex_disclaimers_present",
    "feat_interact_hedge_x_guarantee",
]

EMBEDDING_FEATURE = "legal_bert_emb"  # 768-dimensional Legal-BERT embeddings


class TemporalModelEvaluator:
    """Model training and evaluation that inherits existing temporal k-fold structure."""

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
        self.scout_pass = scout_pass
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata (contains class weights and binning info)
        self.metadata = self.load_metadata()

        # Results storage
        self.results = {}
        self.test_predictions = {}
        self.feature_importance = {}
        self.optimal_thresholds = {}

        # Court-based identity suppression storage
        self.court_means = {}
        self.state_means = {}
        self.global_mean = None

        logger.info(f"Initialized temporal evaluator for fold {fold}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Grid search: {use_grid_search}")
        logger.info(f"Scout pass: {scout_pass}")
        logger.info(f"Methodology: {self.metadata.get('methodology', 'unknown')}")

    def load_metadata(self) -> Dict:
        """Load the per-fold metadata with class weights and binning info."""
        metadata_path = self.data_dir / "per_fold_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata: {metadata.get('methodology', 'unknown')}")
            return metadata
        else:
            logger.warning("No metadata found - using default weights")
            return {}

    def get_fold_class_weights(self) -> Dict[str, float]:
        """Get the class weights for the current fold from metadata."""
        if (
            "weights" in self.metadata
            and f"fold_{self.fold}" in self.metadata["weights"]
        ):
            class_weights = self.metadata["weights"][f"fold_{self.fold}"][
                "class_weights"
            ]
            # Convert string keys to int keys for sklearn
            return {int(k): float(v) for k, v in class_weights.items()}
        else:
            logger.warning(
                f"No class weights found for fold {self.fold}, using balanced"
            )
            return "balanced"

    def load_temporal_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the existing temporal train/dev/test splits (NO RE-SPLITTING!)."""
        logger.info("Loading existing temporal splits...")

        # Load training data for this fold
        train_path = self.data_dir / f"fold_{self.fold}" / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        # Load dev data for this fold (used as validation)
        dev_path = self.data_dir / f"fold_{self.fold}" / "dev.jsonl"
        if not dev_path.exists():
            logger.warning(
                f"Dev data not found: {dev_path}, using train as both train/dev"
            )
            dev_path = train_path

        # Load OOF test data (final evaluation)
        test_path = self.data_dir / "oof_test" / "test.jsonl"
        if not test_path.exists():
            logger.warning(f"OOF test not found: {test_path}, using dev as test")
            test_path = dev_path

        # Load with pandas
        train_df = pd.read_json(train_path, lines=True)
        dev_df = pd.read_json(dev_path, lines=True)
        test_df = pd.read_json(test_path, lines=True)

        logger.info(
            f"Loaded temporal splits: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}"
        )

        # Verify the binary outcome distribution
        for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            if "outcome_bin" in df.columns:
                dist = df["outcome_bin"].value_counts().sort_index()
                logger.info(f"{name} outcome distribution: {dist.to_dict()}")

        return train_df, dev_df, test_df

    def extract_state_from_court_id(self, court_ids: np.ndarray) -> np.ndarray:
        """Extract state codes from court_id (letters in court_id)."""
        states = []
        for court_id in court_ids:
            if pd.isna(court_id) or court_id == "":
                states.append("UNK")
            else:
                # Extract letters from court_id (e.g., 'ca9' -> 'ca', 'nyed' -> 'nyed')
                state_code = "".join([c for c in str(court_id).lower() if c.isalpha()])
                states.append(state_code if state_code else "UNK")
        return np.array(states)

    def compute_suppression_means(
        self, X_train: np.ndarray, court_ids: np.ndarray, fit_global: bool = True
    ) -> None:
        """Compute court, state, and global means for identity suppression."""
        logger.info("Computing court-based suppression means...")

        # Extract states from court_ids
        states = self.extract_state_from_court_id(court_ids)

        # Compute court-level means (minimum 10 samples per court)
        self.court_means = {}
        for court in np.unique(court_ids):
            if court != "" and not pd.isna(court):
                court_mask = court_ids == court
                if court_mask.sum() >= 10:  # Minimum threshold
                    self.court_means[court] = X_train[court_mask].mean(axis=0)

        logger.info(f"Computed court means for {len(self.court_means)} courts")

        # Compute state-level means (minimum 25 samples per state)
        self.state_means = {}
        for state in np.unique(states):
            if state != "UNK":
                state_mask = states == state
                if state_mask.sum() >= 25:  # Higher threshold for state fallback
                    self.state_means[state] = X_train[state_mask].mean(axis=0)

        logger.info(f"Computed state means for {len(self.state_means)} states")

        # Global mean as final fallback
        if fit_global:
            self.global_mean = X_train.mean(axis=0)
            logger.info("Computed global mean fallback")

    def apply_identity_suppression(
        self, X: np.ndarray, court_ids: np.ndarray
    ) -> np.ndarray:
        """Apply court-based identity suppression with state/global fallback."""
        X_suppressed = X.copy()
        states = self.extract_state_from_court_id(court_ids)

        suppression_stats = {"court": 0, "state": 0, "global": 0}

        for i, (court_id, state) in enumerate(zip(court_ids, states)):
            # Try court-level suppression first
            if court_id in self.court_means:
                X_suppressed[i] -= self.court_means[court_id]
                suppression_stats["court"] += 1
            # Fallback to state-level suppression
            elif state in self.state_means:
                X_suppressed[i] -= self.state_means[state]
                suppression_stats["state"] += 1
            # Final fallback to global mean
            elif self.global_mean is not None:
                X_suppressed[i] -= self.global_mean
                suppression_stats["global"] += 1

        logger.info(
            f"Suppression applied: {suppression_stats['court']} court, "
            f"{suppression_stats['state']} state, {suppression_stats['global']} global"
        )

        return X_suppressed

    def prepare_features(
        self, df: pd.DataFrame, feature_config: str, scaler=None, fit=False
    ) -> Tuple[np.ndarray, StandardScaler, List[str]]:
        """Prepare features based on configuration."""
        logger.info(f"Preparing features for config: {feature_config}")

        features = []
        feature_names = []

        if feature_config == "3VF":
            # 3 validated features only
            for feature in VALIDATED_SCALARS:
                if feature in df.columns:
                    features.append(df[feature].values.reshape(-1, 1))
                    feature_names.append(feature)
                else:
                    logger.warning(f"Feature {feature} not found in data")

        elif feature_config in ["E", "E+3"]:
            # Always include embeddings
            if EMBEDDING_FEATURE in df.columns:
                embeddings = np.array(df[EMBEDDING_FEATURE].tolist())
                features.append(embeddings)
                feature_names.extend([f"emb_{i}" for i in range(embeddings.shape[1])])
            else:
                raise ValueError(f"Embedding feature {EMBEDDING_FEATURE} not found")

            # Add scalars if E+3 configuration
            if feature_config == "E+3":
                for feature in VALIDATED_SCALARS:
                    if feature in df.columns:
                        features.append(df[feature].values.reshape(-1, 1))
                        feature_names.append(feature)
                    else:
                        logger.warning(f"Feature {feature} not found in data")

        # Combine all features
        if not features:
            raise ValueError(f"No features found for configuration {feature_config}")

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
        """Define hyperparameter grids for all 13 model variants."""

        if self.scout_pass:
            # Scout pass: reduced grids
            return {
                # A) 3 Validated Features only
                "lr_l2_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l1_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_elasticnet_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "svm_linear_3VF": {
                    "C": [0.1, 1.0, 10.0],
                    "loss": ["squared_hinge"],
                    "penalty": ["l2"],
                    "dual": [True],
                    "max_iter": [5000],
                    "tol": [1e-4],
                },
                "mlr_enhanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.9],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                },
                "mlr_balanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.9],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                },
                # B) Embedding variants
                "lr_l2_E": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l1_E": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E": {
                    "C": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l2_E+3": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l1_E+3": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E+3": {
                    "C": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
            }

        elif self.use_grid_search:
            # Full pass: complete grids
            return {
                # A) 3 Validated Features only
                "lr_l2_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l1_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_elasticnet_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "svm_linear_3VF": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "loss": ["squared_hinge"],
                    "penalty": ["l2"],
                    "dual": [True],
                    "max_iter": [5000],
                    "tol": [1e-4],
                },
                "mlr_enhanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.5],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                },
                "mlr_balanced_3VF": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__l1_ratio": [0.1, 0.5],
                    "estimator__solver": ["saga"],
                    "estimator__max_iter": [2000],
                },
                # B) Embedding variants
                "lr_l2_E": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l1_E": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l2_E+3": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_l1_E+3": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
                "lr_elasticnet_E+3": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                    "solver": ["saga"],
                    "max_iter": [2000],
                    "tol": [1e-4],
                },
            }

        else:
            # Default: minimal grids for quick testing
            return {
                "lr_l2_E": {"C": [1.0], "solver": ["lbfgs"]},
                "lr_l1_E": {"C": [1.0], "solver": ["saga"]},
                "lr_elasticnet_E": {"C": [1.0], "l1_ratio": [0.5], "solver": ["saga"]},
                "lr_l2_E+3": {"C": [1.0], "solver": ["lbfgs"]},
                "lr_l1_E+3": {"C": [1.0], "solver": ["saga"]},
                "lr_elasticnet_E+3": {
                    "C": [1.0],
                    "l1_ratio": [0.5],
                    "solver": ["saga"],
                },
            }

    def define_models(self) -> Dict[str, Any]:
        """Define all 13 model variants."""
        models = {}
        grids = self.get_hyperparameter_grids()
        class_weights = self.get_fold_class_weights()

        # Base models for all 13 variants
        base_models = {
            # A) 3 Validated Features only (7 variants)
            "lr_l2_3VF": LogisticRegression(
                penalty="l2", random_state=42, class_weight=class_weights
            ),
            "lr_l1_3VF": LogisticRegression(
                penalty="l1", random_state=42, class_weight=class_weights
            ),
            "lr_elasticnet_3VF": LogisticRegression(
                penalty="elasticnet", random_state=42, class_weight=class_weights
            ),
            "svm_linear_3VF": LinearSVC(random_state=42, class_weight=class_weights),
            "mlr_enhanced_3VF": OneVsRestClassifier(
                LogisticRegression(
                    penalty="elasticnet", random_state=42, class_weight=class_weights
                )
            ),
            "mlr_balanced_3VF": OneVsRestClassifier(
                LogisticRegression(
                    penalty="elasticnet", random_state=42, class_weight=class_weights
                )
            ),
            # B) Embedding variants (6 variants)
            "lr_l2_E": LogisticRegression(
                penalty="l2", random_state=42, class_weight=class_weights
            ),
            "lr_l1_E": LogisticRegression(
                penalty="l1", random_state=42, class_weight=class_weights
            ),
            "lr_elasticnet_E": LogisticRegression(
                penalty="elasticnet", random_state=42, class_weight=class_weights
            ),
            "lr_l2_E+3": LogisticRegression(
                penalty="l2", random_state=42, class_weight=class_weights
            ),
            "lr_l1_E+3": LogisticRegression(
                penalty="l1", random_state=42, class_weight=class_weights
            ),
            "lr_elasticnet_E+3": LogisticRegression(
                penalty="elasticnet", random_state=42, class_weight=class_weights
            ),
        }

        # Add POLR if available
        if MORD_AVAILABLE:
            base_models["polr_3VF"] = mord.LogisticAT()
            if "polr_3VF" not in grids:
                grids["polr_3VF"] = {"alpha": [0.01, 0.1, 1.0, 10.0]}

        # Create GridSearchCV models if grid search enabled
        for model_name, base_model in base_models.items():
            if (self.use_grid_search or self.scout_pass) and model_name in grids:
                models[model_name] = GridSearchCV(
                    base_model,
                    grids[model_name],
                    cv=3,  # Inner CV for hyperparameter selection
                    scoring="roc_auc",  # Will be overridden by suppressed evaluation
                    n_jobs=-1,
                    verbose=1,
                )
                logger.info(f"Created GridSearchCV model: {model_name}")
            else:
                models[model_name] = base_model
                logger.info(f"Created default model: {model_name}")

        logger.info(f"Defined {len(models)} models: {list(models.keys())}")
        return models

    def evaluate_model_on_temporal_split(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        court_ids_train: np.ndarray,
        court_ids_dev: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """Evaluate model using existing temporal train/dev split with identity suppression."""
        logger.info(
            f"Evaluating {model_name} on temporal train/dev split (court-suppressed selection)..."
        )

        # Fit model on RAW training features (never train on suppressed)
        trained_model = clone(model)
        trained_model.fit(X_train, y_train)

        # Compute suppression means on training data only
        self.compute_suppression_means(X_train, court_ids_train, fit_global=True)

        # RAW predictions on dev set
        if hasattr(trained_model, "predict_proba"):
            dev_proba_raw = trained_model.predict_proba(X_dev)
            if dev_proba_raw.shape[1] == 2:
                dev_proba_raw = dev_proba_raw[:, 1]
            else:
                dev_proba_raw = dev_proba_raw[:, 0]
        elif hasattr(trained_model, "decision_function"):
            dev_proba_raw = expit(trained_model.decision_function(X_dev))
        else:
            dev_proba_raw = trained_model.predict(X_dev).astype(float)

        # SUPPRESSED predictions on dev set (for selection)
        X_dev_suppressed = self.apply_identity_suppression(X_dev, court_ids_dev)

        if hasattr(trained_model, "predict_proba"):
            dev_proba_suppressed = trained_model.predict_proba(X_dev_suppressed)
            if dev_proba_suppressed.shape[1] == 2:
                dev_proba_suppressed = dev_proba_suppressed[:, 1]
            else:
                dev_proba_suppressed = dev_proba_suppressed[:, 0]
        elif hasattr(trained_model, "decision_function"):
            dev_proba_suppressed = expit(
                trained_model.decision_function(X_dev_suppressed)
            )
        else:
            dev_proba_suppressed = dev_proba_raw  # Fallback

        # Calculate metrics for both views
        # RAW metrics
        dev_auc_raw = (
            roc_auc_score(y_dev, dev_proba_raw) if len(np.unique(y_dev)) > 1 else 0.5
        )
        dev_pr_auc_raw = average_precision_score(y_dev, dev_proba_raw)
        optimal_threshold_raw, dev_mcc_raw = self.find_mcc_optimal_threshold(
            y_dev, dev_proba_raw
        )
        operating_metrics_raw = self.calculate_operating_point_metrics(
            y_dev, dev_proba_raw, optimal_threshold_raw
        )
        dev_brier_raw = brier_score_loss(y_dev, dev_proba_raw)
        dev_ece_raw = self.calculate_ece(y_dev, dev_proba_raw)

        # SUPPRESSED metrics (PRIMARY for selection)
        dev_auc_suppressed = (
            roc_auc_score(y_dev, dev_proba_suppressed)
            if len(np.unique(y_dev)) > 1
            else 0.5
        )
        dev_pr_auc_suppressed = average_precision_score(y_dev, dev_proba_suppressed)
        optimal_threshold_suppressed, dev_mcc_suppressed = (
            self.find_mcc_optimal_threshold(y_dev, dev_proba_suppressed)
        )
        operating_metrics_suppressed = self.calculate_operating_point_metrics(
            y_dev, dev_proba_suppressed, optimal_threshold_suppressed
        )
        dev_brier_suppressed = brier_score_loss(y_dev, dev_proba_suppressed)
        dev_ece_suppressed = self.calculate_ece(y_dev, dev_proba_suppressed)

        # Store optimal threshold from SUPPRESSED view (used for selection)
        self.optimal_thresholds[model_name] = optimal_threshold_suppressed

        # Calculate delta (raw - suppressed)
        delta_mcc = dev_mcc_raw - dev_mcc_suppressed
        delta_auc = dev_auc_raw - dev_auc_suppressed

        logger.info(f"  Dev RAW: MCC={dev_mcc_raw:.4f}, AUC={dev_auc_raw:.4f}")
        logger.info(
            f"  Dev SUPPRESSED: MCC={dev_mcc_suppressed:.4f}, AUC={dev_auc_suppressed:.4f}"
        )
        logger.info(f"  Delta: ΔMCC={delta_mcc:.4f}, ΔAUC={delta_auc:.4f}")

        return {
            # RAW dev metrics
            "dev_mcc_raw": dev_mcc_raw,
            "dev_auc_raw": dev_auc_raw,
            "dev_pr_auc_raw": dev_pr_auc_raw,
            "dev_brier_raw": dev_brier_raw,
            "dev_ece_raw": dev_ece_raw,
            "dev_threshold_raw": optimal_threshold_raw,
            # SUPPRESSED dev metrics (PRIMARY for selection)
            "dev_mcc_suppressed": dev_mcc_suppressed,  # PRIMARY SELECTION METRIC
            "dev_auc_suppressed": dev_auc_suppressed,
            "dev_pr_auc_suppressed": dev_pr_auc_suppressed,
            "dev_brier_suppressed": dev_brier_suppressed,
            "dev_ece_suppressed": dev_ece_suppressed,
            "dev_threshold_suppressed": optimal_threshold_suppressed,
            # Delta metrics
            "delta_mcc": delta_mcc,
            "delta_auc": delta_auc,
            # Operating point metrics (suppressed view)
            "dev_precision_suppressed": operating_metrics_suppressed["precision"],
            "dev_recall_suppressed": operating_metrics_suppressed["recall"],
            "dev_specificity_suppressed": operating_metrics_suppressed["specificity"],
            "dev_confusion_matrix_suppressed": operating_metrics_suppressed[
                "confusion_matrix"
            ],
        }

    def evaluate_model_on_test(
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
        """Evaluate model on final test set with both raw and suppressed views."""
        logger.info(f"Evaluating {model_name} on final test set (raw + suppressed)...")

        # Fit on full training data (RAW features only)
        trained_model = clone(model)
        trained_model.fit(X_train, y_train)

        # Apply calibration (Platt scaling) on raw training data
        try:
            calibrated_model = CalibratedClassifierCV(
                trained_model, method="sigmoid", cv=3
            )
            calibrated_model.fit(X_train, y_train)
            trained_model = calibrated_model
            logger.info(f"Applied Platt calibration for {model_name}")
        except Exception as e:
            logger.warning(f"Calibration failed for {model_name}: {e}")

        # Compute suppression means on training data
        self.compute_suppression_means(X_train, court_ids_train, fit_global=True)

        # RAW test evaluation
        if hasattr(trained_model, "predict_proba"):
            test_proba_raw = trained_model.predict_proba(X_test)
            if test_proba_raw.shape[1] == 2:
                test_proba_raw = test_proba_raw[:, 1]
            else:
                test_proba_raw = test_proba_raw[:, 0]
        elif hasattr(trained_model, "decision_function"):
            test_proba_raw = expit(trained_model.decision_function(X_test))
        else:
            test_proba_raw = trained_model.predict(X_test).astype(float)

        # SUPPRESSED test evaluation (same model, suppressed features)
        X_test_suppressed = self.apply_identity_suppression(X_test, court_ids_test)

        if hasattr(trained_model, "predict_proba"):
            test_proba_suppressed = trained_model.predict_proba(X_test_suppressed)
            if test_proba_suppressed.shape[1] == 2:
                test_proba_suppressed = test_proba_suppressed[:, 1]
            else:
                test_proba_suppressed = test_proba_suppressed[:, 0]
        elif hasattr(trained_model, "decision_function"):
            test_proba_suppressed = expit(
                trained_model.decision_function(X_test_suppressed)
            )
        else:
            test_proba_suppressed = test_proba_raw  # Fallback

        # Use threshold from dev suppressed evaluation
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
            "predictions": (test_proba_raw >= optimal_threshold).astype(int),
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
            "test_brier_raw": test_brier_raw,
            "test_ece_raw": test_ece_raw,
            "test_precision_raw": test_operating_metrics_raw["precision"],
            "test_recall_raw": test_operating_metrics_raw["recall"],
            "test_specificity_raw": test_operating_metrics_raw["specificity"],
            "test_confusion_matrix_raw": test_operating_metrics_raw["confusion_matrix"],
            # SUPPRESSED test metrics
            "test_mcc_suppressed": test_operating_metrics_suppressed["mcc"],
            "test_auc_suppressed": test_auc_suppressed,
            "test_pr_auc_suppressed": test_pr_auc_suppressed,
            "test_brier_suppressed": test_brier_suppressed,
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
        """Run comprehensive evaluation using existing temporal splits."""
        logger.info("=" * 80)
        logger.info("STARTING TEMPORAL MODEL EVALUATION (NO RE-SPLITTING)")
        logger.info("=" * 80)

        # Load existing temporal splits
        train_df, dev_df, test_df = self.load_temporal_data()

        # Define models
        models = self.define_models()

        # Evaluate each model
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

                # Prepare features
                X_train, scaler, feature_names = self.prepare_features(
                    train_df, feature_config, fit=True
                )
                X_dev, _, _ = self.prepare_features(
                    dev_df, feature_config, scaler=scaler
                )
                X_test, _, _ = self.prepare_features(
                    test_df, feature_config, scaler=scaler
                )

                y_train = train_df["outcome_bin"].values
                y_dev = dev_df["outcome_bin"].values
                y_test = test_df["outcome_bin"].values

                court_ids_train = (
                    train_df["court_id"].values
                    if "court_id" in train_df.columns
                    else np.array(["unknown"] * len(train_df))
                )
                court_ids_dev = (
                    dev_df["court_id"].values
                    if "court_id" in dev_df.columns
                    else np.array(["unknown"] * len(dev_df))
                )
                court_ids_test = (
                    test_df["court_id"].values
                    if "court_id" in test_df.columns
                    else np.array(["unknown"] * len(test_df))
                )

                # Dev evaluation using existing temporal split (for selection)
                dev_results = self.evaluate_model_on_temporal_split(
                    model,
                    X_train,
                    y_train,
                    X_dev,
                    y_dev,
                    court_ids_train,
                    court_ids_dev,
                    model_name,
                )

                # Final test evaluation
                test_results = self.evaluate_model_on_test(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    court_ids_train,
                    court_ids_test,
                    model_name,
                )

                # Selection criteria checks (using dev suppressed metrics)
                dev_delta_mcc = dev_results.get("delta_mcc", 0.0)
                passes_delta_check = abs(dev_delta_mcc) <= 0.02  # Δ ≤ 0.02
                passes_ece_check = (
                    dev_results.get("dev_ece_suppressed", 0.0) <= 0.08
                )  # ECE ≤ 0.08

                logger.info(
                    f"   Dev Δ(MCC): {dev_delta_mcc:.4f} ({'PASS' if passes_delta_check else 'FAIL'} ≤ 0.02)"
                )
                logger.info(
                    f"   Dev ECE (suppressed): {dev_results.get('dev_ece_suppressed', 0.0):.4f} ({'PASS' if passes_ece_check else 'FAIL'} ≤ 0.08)"
                )

                # Store combined results
                self.results[model_name] = {
                    **dev_results,
                    **test_results,
                    "feature_config": feature_config,
                    "n_features": len(feature_names),
                    "model_type": type(model).__name__,
                    # Selection criteria
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
                    f"   Dev MCC (SUPPRESSED): {dev_results['dev_mcc_suppressed']:.4f} (PRIMARY SELECTION)"
                )
                logger.info(f"   Test MCC (RAW): {test_results['test_mcc_raw']:.4f}")
                logger.info(
                    f"   Test MCC (SUPPRESSED): {test_results['test_mcc_suppressed']:.4f}"
                )
                logger.info(
                    f"   Test ECE (SUPPRESSED): {test_results['test_ece_suppressed']:.4f}"
                )

            except Exception as e:
                logger.error(f"❌ {model_name} evaluation failed: {e}")
                self.results[model_name] = {"error": str(e)}

        # Generate summary and save results
        self.save_results()
        logger.info("=" * 80)
        logger.info("TEMPORAL EVALUATION COMPLETED")
        logger.info("=" * 80)

    def save_results(self):
        """Save comprehensive results."""
        logger.info("Saving results...")

        # Save raw results
        results_file = self.output_dir / "temporal_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

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
                        "dev_mcc_suppressed": result.get(
                            "dev_mcc_suppressed", 0
                        ),  # PRIMARY SELECTION
                        "test_mcc_suppressed": result.get("test_mcc_suppressed", 0),
                        "dev_auc_suppressed": result.get("dev_auc_suppressed", 0),
                        "test_auc_suppressed": result.get("test_auc_suppressed", 0),
                        "test_pr_auc_suppressed": result.get(
                            "test_pr_auc_suppressed", 0
                        ),
                        "test_brier_suppressed": result.get("test_brier_suppressed", 0),
                        "test_ece_suppressed": result.get("test_ece_suppressed", 0),
                        # RAW metrics (for comparison)
                        "dev_mcc_raw": result.get("dev_mcc_raw", 0),
                        "test_mcc_raw": result.get("test_mcc_raw", 0),
                        "dev_auc_raw": result.get("dev_auc_raw", 0),
                        "test_auc_raw": result.get("test_auc_raw", 0),
                        # Delta metrics
                        "dev_delta_mcc": result.get("delta_mcc", 0),
                        "test_delta_mcc": result.get("delta_mcc_test", 0),
                        "test_delta_auc": result.get("delta_auc_test", 0),
                        # Selection criteria
                        "passes_criteria": result.get(
                            "meets_selection_criteria", False
                        ),
                        "optimal_threshold": result.get("test_threshold_used", 0.5),
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = self.output_dir / "temporal_model_comparison.csv"
            summary_df.to_csv(summary_file, index=False)

            # Sort by dev suppressed MCC (PRIMARY SELECTION METRIC)
            summary_df_sorted = summary_df.sort_values(
                "dev_mcc_suppressed", ascending=False
            )

            logger.info("\n" + "=" * 100)
            logger.info("TEMPORAL MODEL PERFORMANCE SUMMARY")
            logger.info("=" * 100)
            logger.info(summary_df_sorted.to_string(index=False, float_format="%.4f"))

            # Save feature importance
            if self.feature_importance:
                importance_file = self.output_dir / "feature_importance.json"
                with open(importance_file, "w") as f:
                    json.dump(self.feature_importance, f, indent=2, default=str)

            # Save test predictions
            if self.test_predictions:
                predictions_file = self.output_dir / "test_predictions.json"
                with open(predictions_file, "w") as f:
                    json.dump(self.test_predictions, f, indent=2, default=str)

            logger.info(f"Results saved to {self.output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Temporal model evaluation with Legal-BERT embeddings (FIXED VERSION)"
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

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = TemporalModelEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold=args.fold,
        use_grid_search=args.grid_search,
        scout_pass=args.scout_pass,
    )

    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
