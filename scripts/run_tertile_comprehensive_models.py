#!/usr/bin/env python3
"""
Comprehensive Tertile Model Evaluation Script

This script runs all model types (POLR, MLR, ElasticNet, L1, L2, SVM) on the tertile
classification task using the complete dataset with all features.

Models to evaluate:
1. POLR (Proportional Odds Logistic Regression) - Already evaluated
2. MLR (Multinomial Logistic Regression) - Already evaluated
3. Elastic Net - NEW for tertile
4. L1 Lasso - NEW for tertile
5. L2 Ridge - NEW for tertile
6. SVM Linear - NEW for tertile

Usage:
    python scripts/run_tertile_comprehensive_models.py \
        --model-type elasticnet \
        --data-dir data/final_stratified_kfold_splits_authoritative_complete \
        --output-dir runs/tertile_elasticnet_full
"""

import argparse
import json
import orjson
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import hashlib
import joblib

from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy import sparse

from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore")

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=3):
    """Calculate Quadratic Weighted Kappa for ordinal classification."""
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2

    weights = weights / weights.max()

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Calculate expected matrix
    n = len(y_true)
    sum_rows = cm.sum(axis=1)
    sum_cols = cm.sum(axis=0)
    expected = np.outer(sum_rows, sum_cols) / n

    # Calculate weighted sums
    po = np.sum(weights * cm) / n
    pe = np.sum(weights * expected) / n

    # Calculate kappa
    kappa = 1 - (po / pe) if pe != 0 else 0

    return kappa


class TertileModelTrainer:
    """Trainer for various model types on tertile classification."""

    @staticmethod
    def _convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                k: TertileModelTrainer._convert_numpy_types(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [TertileModelTrainer._convert_numpy_types(v) for v in obj]
        return obj

    # Hyperparameter grids for each model type
    HYPERPARAMETER_GRIDS = {
        "polr": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs"],
            "max_iter": [500],
        },
        "mlr": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "newton-cg"],
            "multi_class": ["multinomial"],
            "max_iter": [500],
        },
        "elasticnet": {
            "C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.7, 0.9],
            "solver": ["saga"],
            "penalty": ["elasticnet"],
            "multi_class": ["multinomial"],
            "max_iter": [1000],
        },
        "l1": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1"],
            "solver": ["saga"],
            "multi_class": ["multinomial"],
            "max_iter": [1000],
        },
        "l2": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "multi_class": ["multinomial"],
            "max_iter": [500],
        },
        "svm": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "loss": ["squared_hinge"],  # Better for multi-class
            "max_iter": [2000],
            "random_state": [42],
        },
    }

    def __init__(self, data_dir: Path, output_dir: Path, model_type: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_type = model_type

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata = self._load_metadata()
        self.dnt_columns = self._load_dnt_columns()

        # Initialize results storage
        self.fold_results = {}
        self.oof_predictions = []
        self.oof_true = []

    def _load_metadata(self) -> Dict:
        """Load fold metadata."""
        metadata_path = self.data_dir / "per_fold_metadata.json"
        with open(metadata_path, "rb") as f:
            return orjson.loads(f.read())

    def _load_dnt_columns(self) -> set:
        """Load do-not-train columns."""
        dnt_path = self.data_dir / "dnt_manifest.json"
        with open(dnt_path, "rb") as f:
            manifest = orjson.loads(f.read())
            return set(manifest.get("do_not_train", []))

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get all trainable feature columns."""
        # All columns that are features (interpretable_ or feat_new*)
        feature_cols = [
            col
            for col in df.columns
            if (
                col.startswith("interpretable_")
                or col.startswith("feat_new")
                or col.startswith("feat_new2_")
                or col.startswith("feat_new3_")
                or col.startswith("feat_new4_")
                or col.startswith("feat_new5_")
            )
            and col not in self.dnt_columns
        ]

        logger.info(f"Found {len(feature_cols)} feature columns")
        return sorted(feature_cols)

    def load_fold_data(
        self, fold: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data for a specific fold."""
        fold_dir = self.data_dir / f"fold_{fold}"

        # Fast JSONL loading with orjson
        def load_jsonl_fast(path):
            records = []
            with open(path, "rb") as f:
                for line in f:
                    if line.strip():
                        records.append(orjson.loads(line))
            return pd.DataFrame(records)

        train_df = load_jsonl_fast(fold_dir / "train.jsonl")

        # Handle both val.jsonl and dev.jsonl naming
        if (fold_dir / "val.jsonl").exists():
            val_df = load_jsonl_fast(fold_dir / "val.jsonl")
        else:
            val_df = load_jsonl_fast(fold_dir / "dev.jsonl")

        test_df = (
            load_jsonl_fast(fold_dir / "test.jsonl")
            if (fold_dir / "test.jsonl").exists()
            else val_df
        )

        # Log class distributions
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = df["outcome_bin"].value_counts().sort_index()
            logger.info(f"{name} distribution: {dist.to_dict()}")

        return train_df, val_df, test_df

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaler: Optional[StandardScaler] = None,
        fit: bool = False,
    ) -> Tuple[np.ndarray, Optional[StandardScaler]]:
        """Prepare feature matrix with scaling."""
        X = df[feature_cols].values

        # Handle missing values by filling with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

        return X_scaled, scaler

    def get_model(self, **kwargs):
        """Get model instance based on type."""
        if self.model_type == "polr":
            # POLR uses ordinal regression (we'll approximate with LogisticRegression)
            return LogisticRegression(multi_class="multinomial", **kwargs)
        elif self.model_type == "mlr":
            return LogisticRegression(multi_class="multinomial", **kwargs)
        elif self.model_type in ["elasticnet", "l1", "l2"]:
            return LogisticRegression(**kwargs)
        elif self.model_type == "svm":
            # For multi-class SVM
            base_svm = LinearSVC(**kwargs)
            return OneVsRestClassifier(base_svm, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_fold(self, fold: int) -> Dict[str, Any]:
        """Train model on a single fold."""
        logger.info(f"\nTraining fold {fold}")

        # Load data
        train_df, val_df, test_df = self.load_fold_data(fold)

        # Get feature columns
        feature_cols = self.get_feature_columns(train_df)

        # Prepare features
        X_train, scaler = self.prepare_features(train_df, feature_cols, fit=True)
        y_train = train_df["outcome_bin"].values

        X_val, _ = self.prepare_features(val_df, feature_cols, scaler=scaler)
        y_val = val_df["outcome_bin"].values

        X_test, _ = self.prepare_features(test_df, feature_cols, scaler=scaler)
        y_test = test_df["outcome_bin"].values

        # Get class weights from metadata
        class_weights = self.metadata["weights"][f"fold_{fold}"]["class_weights"]
        class_weight_dict = {int(k): v for k, v in class_weights.items()}

        # Setup hyperparameter search
        param_grid = self.HYPERPARAMETER_GRIDS[self.model_type].copy()

        # Add class weights for applicable models
        if self.model_type != "svm":
            param_grid["class_weight"] = [class_weight_dict, "balanced", None]

        # Create model
        base_model = self.get_model()

        # Grid search with tertile-specific scoring
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Custom scorer for QWK
        from sklearn.metrics import make_scorer

        qwk_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring=qwk_scorer, n_jobs=-1, verbose=1
        )

        # Fit model
        logger.info("Starting hyperparameter search...")
        grid_search.fit(X_train, y_train)

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV QWK: {grid_search.best_score_:.4f}")

        # Get best model
        best_model = grid_search.best_estimator_

        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        val_qwk = quadratic_weighted_kappa(y_val, y_val_pred)
        val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
        val_f1_per_class = f1_score(y_val, y_val_pred, average=None)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        logger.info(f"Validation QWK: {val_qwk:.4f}")
        logger.info(f"Validation Macro F1: {val_f1_macro:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        # Evaluate on test set
        y_test_pred = best_model.predict(X_test)
        test_qwk = quadratic_weighted_kappa(y_test, y_test_pred)
        test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
        test_f1_per_class = f1_score(y_test, y_test_pred, average=None)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Store OOF predictions
        oof_df = pd.DataFrame(
            {
                "case_id": test_df["case_id"],
                "true_label": y_test,
                "pred_label": y_test_pred,
                "fold": fold,
            }
        )
        self.oof_predictions.append(oof_df)

        # Get feature importances (if available)
        feature_importance = None
        if hasattr(best_model, "coef_"):
            # For multi-class, average absolute coefficients across classes
            if len(best_model.coef_.shape) > 1:
                importance = np.abs(best_model.coef_).mean(axis=0)
            else:
                importance = np.abs(best_model.coef_)

            feature_importance = dict(zip(feature_cols, importance))

            # Log top features
            top_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            logger.info("Top 10 features by importance:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"Test Confusion Matrix:\n{cm}")

        # Save model
        model_path = self.output_dir / f"model_fold_{fold}.joblib"
        joblib.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "best_params": grid_search.best_params_,
            },
            model_path,
        )

        # Return results
        return {
            "fold": fold,
            "best_params": grid_search.best_params_,
            "cv_score": grid_search.best_score_,
            "validation": {
                "qwk": val_qwk,
                "f1_macro": val_f1_macro,
                "f1_per_class": val_f1_per_class.tolist(),
                "accuracy": val_accuracy,
            },
            "test": {
                "qwk": test_qwk,
                "f1_macro": test_f1_macro,
                "f1_per_class": test_f1_per_class.tolist(),
                "accuracy": test_accuracy,
                "confusion_matrix": cm.tolist(),
            },
            "feature_importance": feature_importance,
            "n_features": len(feature_cols),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        }

    def evaluate_oof_predictions(self) -> Dict[str, Any]:
        """Evaluate out-of-fold predictions."""
        # Combine all OOF predictions
        oof_df = pd.concat(self.oof_predictions, ignore_index=True)

        y_true = oof_df["true_label"].values
        y_pred = oof_df["pred_label"].values

        # Calculate metrics
        oof_qwk = quadratic_weighted_kappa(y_true, y_pred)
        oof_f1_macro = f1_score(y_true, y_pred, average="macro")
        oof_f1_per_class = f1_score(y_true, y_pred, average=None)
        oof_accuracy = accuracy_score(y_true, y_pred)

        # Confusion matrix
        oof_cm = confusion_matrix(y_true, y_pred)

        # Classification report
        class_report = classification_report(
            y_true, y_pred, target_names=["Low", "Medium", "High"], output_dict=True
        )

        logger.info(f"\nOOF Performance:")
        logger.info(f"QWK: {oof_qwk:.4f}")
        logger.info(f"Macro F1: {oof_f1_macro:.4f}")
        logger.info(f"Accuracy: {oof_accuracy:.4f}")
        logger.info(f"Confusion Matrix:\n{oof_cm}")

        return {
            "qwk": oof_qwk,
            "f1_macro": oof_f1_macro,
            "f1_per_class": oof_f1_per_class.tolist(),
            "accuracy": oof_accuracy,
            "confusion_matrix": oof_cm.tolist(),
            "classification_report": class_report,
            "n_samples": len(oof_df),
        }

    def run_training(self):
        """Run training across all folds."""
        logger.info(f"Starting {self.model_type} training on tertile classification")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        # Determine folds to train
        folds = []
        for fold_dir in sorted(self.data_dir.glob("fold_*")):
            if fold_dir.is_dir():
                fold_num = int(fold_dir.name.split("_")[1])
                folds.append(fold_num)

        logger.info(f"Found {len(folds)} folds: {folds}")

        # Train each fold
        for fold in folds:
            fold_result = self.train_fold(fold)
            self.fold_results[f"fold_{fold}"] = fold_result

        # Evaluate OOF performance
        oof_metrics = self.evaluate_oof_predictions()

        # Calculate average metrics across folds
        avg_metrics = {
            "cv_qwk": np.mean([r["cv_score"] for r in self.fold_results.values()]),
            "val_qwk": np.mean(
                [r["validation"]["qwk"] for r in self.fold_results.values()]
            ),
            "val_f1_macro": np.mean(
                [r["validation"]["f1_macro"] for r in self.fold_results.values()]
            ),
            "test_qwk": np.mean([r["test"]["qwk"] for r in self.fold_results.values()]),
            "test_f1_macro": np.mean(
                [r["test"]["f1_macro"] for r in self.fold_results.values()]
            ),
        }

        # Save all results
        results = {
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "fold_results": self.fold_results,
            "oof_metrics": oof_metrics,
            "average_metrics": avg_metrics,
            "hyperparameter_grid": self.HYPERPARAMETER_GRIDS[self.model_type],
        }

        # Save results with orjson
        with open(self.output_dir / "training_results.json", "wb") as f:
            # Convert numpy types before serialization
            f.write(
                orjson.dumps(
                    self._convert_numpy_types(results), option=orjson.OPT_INDENT_2
                )
            )

        # Save OOF predictions with orjson for speed
        oof_df = pd.concat(self.oof_predictions, ignore_index=True)
        with open(self.output_dir / "oof_predictions.jsonl", "wb") as f:
            for _, row in oof_df.iterrows():
                f.write(orjson.dumps(row.to_dict()))
                f.write(b"\n")

        # Create summary report
        self.create_summary_report(results)

        logger.info(f"\nTraining complete! Results saved to {self.output_dir}")
        logger.info(f"Final OOF QWK: {oof_metrics['qwk']:.4f}")
        logger.info(f"Final OOF Macro F1: {oof_metrics['f1_macro']:.4f}")

    def create_summary_report(self, results: Dict[str, Any]):
        """Create a markdown summary report."""
        report_path = self.output_dir / "TRAINING_SUMMARY.md"

        with open(report_path, "w") as f:
            f.write(f"# {self.model_type.upper()} Tertile Classification Results\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")

            f.write("## Dataset\n\n")
            f.write(f"- Data: `{self.data_dir}`\n")
            f.write(f"- Classes: 3 (Low=0, Medium=1, High=2)\n")
            f.write(f"- Features: All interpretable_* and feat_new* features\n\n")

            f.write("## Overall Performance\n\n")
            oof = results["oof_metrics"]
            avg = results["average_metrics"]

            f.write("### Out-of-Fold (OOF) Metrics\n\n")
            f.write(f"- **QWK**: {oof['qwk']:.4f}\n")
            f.write(f"- **Macro F1**: {oof['f1_macro']:.4f}\n")
            f.write(f"- **Accuracy**: {oof['accuracy']:.4f}\n")
            f.write(
                f"- **Per-Class F1**: {[f'{x:.3f}' for x in oof['f1_per_class']]}\n\n"
            )

            f.write("### Cross-Validation Averages\n\n")
            f.write(f"- **CV QWK**: {avg['cv_qwk']:.4f}\n")
            f.write(f"- **Val QWK**: {avg['val_qwk']:.4f}\n")
            f.write(f"- **Test QWK**: {avg['test_qwk']:.4f}\n\n")

            f.write("## Confusion Matrix (OOF)\n\n")
            f.write("```\n")
            f.write("Predicted →\n")
            f.write("True ↓   Low  Med  High\n")
            cm = np.array(oof["confusion_matrix"])
            f.write(f"Low    {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}\n")
            f.write(f"Med    {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}\n")
            f.write(f"High   {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}\n")
            f.write("```\n\n")

            f.write("## Per-Fold Results\n\n")
            f.write("| Fold | CV QWK | Val QWK | Test QWK | Val F1 | Test F1 |\n")
            f.write("|------|---------|---------|----------|---------|----------|\n")

            for fold_name, fold_result in results["fold_results"].items():
                fold_num = fold_result["fold"]
                f.write(f"| {fold_num} | ")
                f.write(f"{fold_result['cv_score']:.4f} | ")
                f.write(f"{fold_result['validation']['qwk']:.4f} | ")
                f.write(f"{fold_result['test']['qwk']:.4f} | ")
                f.write(f"{fold_result['validation']['f1_macro']:.4f} | ")
                f.write(f"{fold_result['test']['f1_macro']:.4f} |\n")

            f.write("\n## Hyperparameters\n\n")
            f.write("### Search Grid\n\n")
            f.write("```python\n")
            f.write(f"{results['hyperparameter_grid']}\n")
            f.write("```\n\n")

            f.write("### Selected Parameters (Per Fold)\n\n")
            for fold_name, fold_result in results["fold_results"].items():
                f.write(f"**{fold_name}**: `{fold_result['best_params']}`\n")


def main():
    parser = argparse.ArgumentParser(description="Train tertile classification models")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["polr", "mlr", "elasticnet", "l1", "l2", "svm"],
        help="Model type to train",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/final_stratified_kfold_splits_authoritative_complete",
        help="Path to tertile dataset directory",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = TertileModelTrainer(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_type=args.model_type,
    )

    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()
