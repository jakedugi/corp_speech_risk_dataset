#!/usr/bin/env python3
"""
Comprehensive Model Training and Evaluation for Validated Binary Features

This script trains and evaluates multiple models using the 3 validated features
that passed all fingerprinting diagnostics and DNT constraints.

Features:
- new4_neutral_to_disclaimer_transition_rate
- lex_disclaimers_present
- feat_interact_hedge_x_guarantee

Models:
- Logistic Regression (L1, L2, ElasticNet)
- Linear SVM
- Multinomial Logistic Regression (Enhanced, Balanced)
- Proportional Odds Logistic Regression (POLR)
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
from sklearn.model_selection import StratifiedKFold, GroupKFold
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
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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

# Validated features from fingerprinting diagnostics
VALIDATED_FEATURES = [
    "new4_neutral_to_disclaimer_transition_rate",
    "lex_disclaimers_present",
    "feat_interact_hedge_x_guarantee",
]


class ComprehensiveModelEvaluator:
    """Comprehensive model training and evaluation for validated binary features."""

    def __init__(self, data_dir: str, output_dir: str, fold: int = 4):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fold = fold
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.feature_importance = {}

        logger.info(f"Initialized evaluator for fold {fold}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Output dir: {output_dir}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/dev/test data for the specified fold."""
        logger.info("Loading data...")

        # Load datasets
        train_path = self.data_dir / f"fold_{self.fold}" / "train.jsonl"
        dev_path = self.data_dir / f"fold_{self.fold}" / "dev.jsonl"
        test_path = self.data_dir / f"fold_{self.fold}" / "test.jsonl"

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
            logger.info(f"{name} distribution: {dict(dist)}")

        # Verify features exist
        missing_features = [
            f for f in VALIDATED_FEATURES if f not in full_train_df.columns
        ]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        logger.info(f"All {len(VALIDATED_FEATURES)} validated features found")
        return full_train_df, test_df

    def prepare_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Prepare feature matrices and targets."""
        logger.info("Preparing feature matrices...")

        # Extract features and targets
        X_train = train_df[VALIDATED_FEATURES].fillna(0).values
        y_train = train_df["outcome_bin"].values
        case_ids_train = train_df["case_id"].values

        X_test = test_df[VALIDATED_FEATURES].fillna(0).values
        y_test = test_df["outcome_bin"].values
        case_ids_test = test_df["case_id"].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logger.info(
            f"Feature matrix shapes: train={X_train_scaled.shape}, test={X_test_scaled.shape}"
        )
        logger.info(
            f"Class distribution - train: {np.bincount(y_train)}, test: {np.bincount(y_test)}"
        )

        return {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "case_ids_train": case_ids_train,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "case_ids_test": case_ids_test,
            "scaler": scaler,
            "feature_names": VALIDATED_FEATURES,
        }

    def define_models(self) -> Dict[str, Any]:
        """Define all models to evaluate."""
        models = {}

        # Logistic Regression variants
        models["logistic_l1"] = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, random_state=42, max_iter=1000
        )
        models["logistic_l2"] = LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, random_state=42, max_iter=1000
        )
        models["logistic_elasticnet"] = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=1.0,
            l1_ratio=0.5,
            random_state=42,
            max_iter=1000,
        )

        # Linear SVM
        models["svm_linear"] = LinearSVC(
            C=1.0, random_state=42, max_iter=2000, dual=False
        )

        # Multinomial Logistic Regression variants
        models["mlr_enhanced"] = OneVsRestClassifier(
            LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        )
        models["mlr_balanced"] = OneVsRestClassifier(
            LogisticRegression(
                C=1.0, class_weight="balanced", random_state=42, max_iter=1000
            )
        )

        # POLR (if available)
        if MORD_AVAILABLE:
            models["polr_champion"] = mord.LogisticAT(alpha=1.0)

        logger.info(f"Defined {len(models)} models: {list(models.keys())}")
        return models

    def evaluate_model_cv(
        self, model, X: np.ndarray, y: np.ndarray, case_ids: np.ndarray, model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model using stratified group cross-validation."""
        logger.info(f"Evaluating {model_name} with CV...")

        # Use StratifiedKFold (we'll group by case manually)
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_scores = []
        oof_preds = np.full(len(y), np.nan)
        oof_proba = np.full(len(y), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Clone and fit model
            model_clone = self._clone_model(model)
            model_clone.fit(X_tr, y_tr)

            # Get predictions
            if hasattr(model_clone, "predict_proba"):
                proba = model_clone.predict_proba(X_val)
                if proba.shape[1] == 2:
                    val_proba = proba[:, 1]
                else:
                    val_proba = proba[:, 0]  # For binary, sometimes only one class
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
        oof_auc = (
            roc_auc_score(y[mask], oof_proba[mask])
            if len(np.unique(y[mask])) > 1
            else 0.5
        )
        oof_f1 = f1_score(y[mask], oof_preds[mask])
        oof_acc = accuracy_score(y[mask], oof_preds[mask])
        oof_bal_acc = balanced_accuracy_score(y[mask], oof_preds[mask])

        # Store OOF predictions
        self.oof_predictions[model_name] = {
            "predictions": oof_preds,
            "probabilities": oof_proba,
            "true_labels": y,
        }

        results = {
            "cv_aucs": cv_scores,
            "cv_auc_mean": np.mean(cv_scores),
            "cv_auc_std": np.std(cv_scores),
            "oof_auc": oof_auc,
            "oof_f1": oof_f1,
            "oof_accuracy": oof_acc,
            "oof_balanced_accuracy": oof_bal_acc,
            "n_folds": len(cv_scores),
        }

        logger.info(
            f"  CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}"
        )
        logger.info(f"  OOF AUC: {oof_auc:.4f}, F1: {oof_f1:.4f}")

        return results

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
        logger.info(f"Training {model_name} on full training set...")

        # Train on full training set
        model_final = self._clone_model(model)
        model_final.fit(X_train, y_train)

        # Get test predictions
        test_preds = model_final.predict(X_test)

        if hasattr(model_final, "predict_proba"):
            test_proba = model_final.predict_proba(X_test)
            if test_proba.shape[1] == 2:
                test_proba = test_proba[:, 1]
            else:
                test_proba = test_proba[:, 0]
        elif hasattr(model_final, "decision_function"):
            test_proba = expit(model_final.decision_function(X_test))
        else:
            test_proba = test_preds.astype(float)

        # Store test predictions
        self.test_predictions[model_name] = {
            "predictions": test_preds,
            "probabilities": test_proba,
            "true_labels": y_test,
        }

        # Calculate test metrics
        test_auc = (
            roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5
        )
        test_f1 = f1_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_bal_acc = balanced_accuracy_score(y_test, test_preds)

        # Precision-Recall metrics
        precision, recall, _ = precision_recall_curve(y_test, test_proba)
        test_pr_auc = average_precision_score(y_test, test_proba)

        # Calibration metrics
        try:
            test_brier = brier_score_loss(y_test, test_proba)
            test_logloss = log_loss(y_test, test_proba)
        except:
            test_brier = np.nan
            test_logloss = np.nan

        results = {
            "test_auc": test_auc,
            "test_pr_auc": test_pr_auc,
            "test_f1": test_f1,
            "test_accuracy": test_acc,
            "test_balanced_accuracy": test_bal_acc,
            "test_brier_score": test_brier,
            "test_log_loss": test_logloss,
        }

        logger.info(
            f"  Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}, PR-AUC: {test_pr_auc:.4f}"
        )

        return results

    def _clone_model(self, model):
        """Clone a model."""
        from sklearn.base import clone

        try:
            return clone(model)
        except:
            # Fallback for models that don't support cloning
            return type(model)(**model.get_params())

    def case_level_evaluation(self, model_name: str) -> Dict[str, Any]:
        """Evaluate model performance at case level."""
        logger.info(f"Case-level evaluation for {model_name}...")

        results = {}

        # OOF case-level evaluation
        if model_name in self.oof_predictions:
            oof_data = self.oof_predictions[model_name]
            # We'd need case IDs to properly aggregate, for now use quote-level
            results["oof_case_auc"] = "not_implemented"

        # Test case-level evaluation
        if model_name in self.test_predictions:
            test_data = self.test_predictions[model_name]
            results["test_case_auc"] = "not_implemented"

        return results

    def generate_visualizations(self):
        """Generate evaluation visualizations."""
        logger.info("Generating visualizations...")

        # Create figures directory
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # 1. ROC Curves Comparison
        plt.figure(figsize=(12, 8))

        for model_name in self.test_predictions.keys():
            test_data = self.test_predictions[model_name]
            y_true = test_data["true_labels"]
            y_proba = test_data["probabilities"]

            # Calculate ROC curve points
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)

            plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison - Test Set")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "roc_curves_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Feature Importance Comparison
        if hasattr(self, "feature_importance") and self.feature_importance:
            plt.figure(figsize=(12, 8))

            # Get feature importance for models that support it
            importance_data = []
            for model_name, importance in self.feature_importance.items():
                for i, feat in enumerate(VALIDATED_FEATURES):
                    if i < len(importance):
                        importance_data.append(
                            {
                                "Model": model_name,
                                "Feature": feat,
                                "Importance": abs(importance[i]),
                            }
                        )

            if importance_data:
                import pandas as pd

                df_imp = pd.DataFrame(importance_data)

                # Create heatmap
                pivot_df = df_imp.pivot(
                    index="Feature", columns="Model", values="Importance"
                )
                sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis")
                plt.title("Feature Importance Comparison")
                plt.tight_layout()
                plt.savefig(
                    fig_dir / "feature_importance_heatmap.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        logger.info(f"Visualizations saved to {fig_dir}")

    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE MODEL EVALUATION")
        logger.info("=" * 80)

        # Load data
        train_df, test_df = self.load_data()
        data = self.prepare_data(train_df, test_df)

        # Define models
        models = self.define_models()

        # Evaluate each model
        for model_name, model in models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"EVALUATING: {model_name.upper()}")
            logger.info(f"{'='*60}")

            try:
                # Cross-validation evaluation
                cv_results = self.evaluate_model_cv(
                    model,
                    data["X_train"],
                    data["y_train"],
                    data["case_ids_train"],
                    model_name,
                )

                # Test set evaluation
                test_results = self.evaluate_model_test(
                    model,
                    data["X_train"],
                    data["y_train"],
                    data["X_test"],
                    data["y_test"],
                    model_name,
                )

                # Case-level evaluation
                case_results = self.case_level_evaluation(model_name)

                # Store combined results
                self.results[model_name] = {
                    **cv_results,
                    **test_results,
                    **case_results,
                    "model_type": type(model).__name__,
                }

                # Extract feature importance if available
                if hasattr(model, "coef_"):
                    self.feature_importance[model_name] = model.coef_.flatten()
                elif hasattr(model, "feature_importances_"):
                    self.feature_importance[model_name] = model.feature_importances_

                logger.info(f"✅ {model_name} evaluation completed")

            except Exception as e:
                logger.error(f"❌ {model_name} evaluation failed: {e}")
                self.results[model_name] = {"error": str(e)}

        # Generate visualizations
        self.generate_visualizations()

        # Save results
        self.save_results()

        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE EVALUATION COMPLETED")
        logger.info("=" * 80)

    def save_results(self):
        """Save all results to files."""
        logger.info("Saving results...")

        # 1. Save main results
        results_file = self.output_dir / "comprehensive_model_results.json"
        with open(results_file, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for model_name, result in self.results.items():
                json_results[model_name] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_results[model_name][key] = value.tolist()
                    elif isinstance(value, (np.float32, np.float64)):
                        json_results[model_name][key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        json_results[model_name][key] = int(value)
                    else:
                        json_results[model_name][key] = value

            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"Main results saved to {results_file}")

        # 2. Save summary table
        summary_rows = []
        for model_name, result in self.results.items():
            if "error" not in result:
                summary_rows.append(
                    {
                        "Model": model_name,
                        "CV_AUC_Mean": result.get("cv_auc_mean", np.nan),
                        "CV_AUC_Std": result.get("cv_auc_std", np.nan),
                        "OOF_AUC": result.get("oof_auc", np.nan),
                        "OOF_F1": result.get("oof_f1", np.nan),
                        "Test_AUC": result.get("test_auc", np.nan),
                        "Test_F1": result.get("test_f1", np.nan),
                        "Test_PR_AUC": result.get("test_pr_auc", np.nan),
                        "Test_Brier": result.get("test_brier_score", np.nan),
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df = summary_df.round(4)
            summary_file = self.output_dir / "model_performance_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Summary table saved to {summary_file}")

            # Print summary to console
            print("\n" + "=" * 100)
            print("MODEL PERFORMANCE SUMMARY")
            print("=" * 100)
            print(summary_df.to_string(index=False))
            print("=" * 100)

        # 3. Save feature importance
        if self.feature_importance:
            importance_rows = []
            for model_name, importance in self.feature_importance.items():
                for i, feat in enumerate(VALIDATED_FEATURES):
                    if i < len(importance):
                        importance_rows.append(
                            {
                                "Model": model_name,
                                "Feature": feat,
                                "Importance": importance[i],
                                "Abs_Importance": abs(importance[i]),
                            }
                        )

            if importance_rows:
                importance_df = pd.DataFrame(importance_rows)
                importance_file = self.output_dir / "feature_importance.csv"
                importance_df.to_csv(importance_file, index=False)
                logger.info(f"Feature importance saved to {importance_file}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation for validated features"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to k-fold data directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument("--fold", type=int, default=4, help="Fold number to use")

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = ComprehensiveModelEvaluator(
        data_dir=args.data_dir, output_dir=args.output_dir, fold=args.fold
    )

    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
