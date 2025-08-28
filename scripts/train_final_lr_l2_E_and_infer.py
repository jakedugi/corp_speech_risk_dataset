#!/usr/bin/env python3
"""
Final Production Model: lr_l2_E Training and Inference

Based on deep run results:
- Winner: lr_l2_E (Legal-BERT embeddings only)
- C=0.01, lbfgs solver, sigmoid calibration
- Threshold=0.835 (MCC-optimal from dev-suppressed)
- Conservative model: High precision (~80%), low recall (~1.6%)

This script:
1. Trains on ALL folds (0-4) train+dev data
2. Applies sigmoid calibration
3. Infers on entire dataset (including training data)
4. Saves lr_l2_E_FINAL_Predictions with probabilities and binary labels
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import matthews_corrcoef

# Fast JSON loading with orjson optimization
try:
    import orjson as _json

    def _loads_bytes(data: bytes):
        return _json.loads(data)

    def _dumps(obj) -> bytes:
        return _json.dumps(obj, option=_json.OPT_SERIALIZE_NUMPY)

    ORJSON_AVAILABLE = True
except ImportError:
    import json as _json

    def _loads_bytes(data: bytes):
        return _json.loads(data.decode("utf-8"))

    def _dumps(obj) -> str:
        return _json.dumps(obj, default=str)

    ORJSON_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Validated features (the 3 that passed fingerprinting)
VALIDATED_SCALARS = [
    "feat_new4_neutral_to_disclaimer_transition_rate",
    "lex_disclaimers_present",
    "feat_interact_hedge_x_guarantee",
]

EMBEDDING_FEATURE = "legal_bert_emb"  # 768-dimensional Legal-BERT embeddings


class FinalModelTrainer:
    """Train final lr_l2_E model on all available data and infer on entire dataset."""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Winning model configuration from deep run
        self.final_config = {
            "penalty": "l2",
            "C": 0.01,
            "solver": "lbfgs",
            "max_iter": 5000,
            "random_state": 42,
            "class_weight": "balanced",
        }
        self.calibration_method = "sigmoid"
        self.optimal_threshold = 0.835  # From deep run dev-suppressed MCC optimization

        # Model components
        self.scaler = None
        self.model = None
        self.calibrated_model = None

        logger.info(f"Initialized final model trainer")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Model config: {self.final_config}")
        logger.info(f"Threshold: {self.optimal_threshold}")

    def load_metadata(self) -> Dict:
        """Load metadata for class weights."""
        metadata_path = self.data_dir / "per_fold_metadata.json"
        if metadata_path.exists():
            if ORJSON_AVAILABLE:
                with open(metadata_path, "rb") as f:
                    return _loads_bytes(f.read())
            else:
                with open(metadata_path, "r") as f:
                    return _json.load(f)
        return {}

    def load_jsonl_fast(self, path: Path) -> pd.DataFrame:
        """Fast JSONL loading with orjson."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data_rows = []
        if ORJSON_AVAILABLE:
            with open(path, "rb") as f:
                for line_bytes in f:
                    line_bytes = line_bytes.strip()
                    if line_bytes:
                        try:
                            data_rows.append(_loads_bytes(line_bytes))
                        except Exception as e:
                            try:
                                data_rows.append(
                                    _json.loads(
                                        line_bytes.decode("utf-8", errors="ignore")
                                    )
                                )
                            except Exception:
                                logger.warning(f"Failed to parse line in {path}: {e}")
        else:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data_rows.append(_json.loads(line))
                        except Exception as e:
                            logger.warning(f"Failed to parse line in {path}: {e}")

        return pd.DataFrame(data_rows)

    def load_all_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load all folds 0-4 train+dev data for final training, keep oof_test separate."""
        logger.info(
            "Loading all folds (0-4) train+dev data for final model training..."
        )

        all_train_dfs = []

        # Load train+dev from all 5 folds
        for fold in range(5):
            train_path = self.data_dir / f"fold_{fold}" / "train.jsonl"
            dev_path = self.data_dir / f"fold_{fold}" / "dev.jsonl"

            if train_path.exists():
                train_df = self.load_jsonl_fast(train_path)
                all_train_dfs.append(train_df)
                logger.info(f"Loaded fold_{fold} train: {len(train_df)} samples")

            if dev_path.exists():
                dev_df = self.load_jsonl_fast(dev_path)
                all_train_dfs.append(dev_df)
                logger.info(f"Loaded fold_{fold} dev: {len(dev_df)} samples")

        # Combine all training data
        combined_train_df = pd.concat(all_train_dfs, ignore_index=True)
        logger.info(f"Combined training data: {len(combined_train_df)} samples")

        # Load oof_test for final evaluation
        oof_test_path = self.data_dir / "oof_test" / "test.jsonl"
        oof_test_df = (
            self.load_jsonl_fast(oof_test_path)
            if oof_test_path.exists()
            else pd.DataFrame()
        )
        logger.info(f"OOF test data: {len(oof_test_df)} samples")

        return combined_train_df, oof_test_df

    def prepare_embeddings_features(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> np.ndarray:
        """Prepare Legal-BERT embeddings features (E configuration)."""
        if EMBEDDING_FEATURE not in df.columns:
            raise ValueError(f"Embedding feature {EMBEDDING_FEATURE} not found")

        # Extract embeddings
        embeddings = np.array(df[EMBEDDING_FEATURE].tolist())
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Scale features
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(embeddings)
            logger.info(f"Fitted scaler: {embeddings.shape} -> {X_scaled.shape}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            X_scaled = self.scaler.transform(embeddings)

        return X_scaled

    def find_mcc_optimal_threshold(self, y_true, y_prob) -> Tuple[float, float]:
        """Find threshold that maximizes MCC (fine-grained search)."""
        thresholds = np.linspace(0.0, 1.0, 201)  # Deep run configuration
        best_mcc = -1
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            if len(np.unique(y_pred)) > 1:
                mcc = matthews_corrcoef(y_true, y_pred)
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_threshold = threshold

        return best_threshold, best_mcc

    def train_final_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the final lr_l2_E model with sigmoid calibration."""
        logger.info("Training final lr_l2_E model...")

        # Create and train base model
        self.model = LogisticRegression(**self.final_config)
        self.model.fit(X_train, y_train)
        logger.info(f"Base model trained: {self.model}")

        # Apply sigmoid calibration
        logger.info("Applying sigmoid calibration...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method=self.calibration_method, cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        logger.info("Sigmoid calibration completed")

    def evaluate_on_oof_test(self, X_oof: np.ndarray, y_oof: np.ndarray) -> Dict:
        """Evaluate final model on OOF test data."""
        logger.info("Evaluating final model on OOF test data...")

        # Get calibrated probabilities
        test_proba = self.calibrated_model.predict_proba(X_oof)[:, 1]

        # Apply optimal threshold
        test_pred = (test_proba >= self.optimal_threshold).astype(int)

        # Calculate metrics
        test_mcc = matthews_corrcoef(y_oof, test_pred)

        # Calculate confusion matrix manually
        tp = ((test_pred == 1) & (y_oof == 1)).sum()
        fp = ((test_pred == 1) & (y_oof == 0)).sum()
        tn = ((test_pred == 0) & (y_oof == 0)).sum()
        fn = ((test_pred == 0) & (y_oof == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results = {
            "test_mcc": test_mcc,
            "test_precision": precision,
            "test_recall": recall,
            "confusion_matrix": {
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            },
            "threshold_used": self.optimal_threshold,
            "n_samples": len(y_oof),
            "n_positive_predictions": int(test_pred.sum()),
            "mean_probability": float(test_proba.mean()),
        }

        logger.info(f"OOF Test Results:")
        logger.info(f"  MCC: {test_mcc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(
            f"  Positive predictions: {test_pred.sum()}/{len(y_oof)} ({100*test_pred.sum()/len(y_oof):.1f}%)"
        )

        return results

    def infer_on_entire_dataset(self) -> pd.DataFrame:
        """Apply final model to entire dataset and return predictions."""
        logger.info("Inferring on entire dataset...")

        all_data_dfs = []

        # Load all folds + oof_test
        for fold in range(5):
            for split in ["train", "dev"]:
                path = self.data_dir / f"fold_{fold}" / f"{split}.jsonl"
                if path.exists():
                    df = self.load_jsonl_fast(path)
                    df["data_split"] = f"fold_{fold}_{split}"
                    all_data_dfs.append(df)

        # Add oof_test
        oof_path = self.data_dir / "oof_test" / "test.jsonl"
        if oof_path.exists():
            oof_df = self.load_jsonl_fast(oof_path)
            oof_df["data_split"] = "oof_test"
            all_data_dfs.append(oof_df)

        # Combine all data
        full_dataset = pd.concat(all_data_dfs, ignore_index=True)
        logger.info(f"Full dataset size: {len(full_dataset)} samples")

        # Remove duplicates based on case_id if present
        if "case_id" in full_dataset.columns:
            initial_size = len(full_dataset)
            full_dataset = full_dataset.drop_duplicates(
                subset=["case_id"], keep="first"
            )
            logger.info(
                f"Removed {initial_size - len(full_dataset)} duplicates based on case_id"
            )

        # Prepare features
        X_full = self.prepare_embeddings_features(full_dataset, fit_scaler=False)

        # Get predictions
        logger.info("Generating predictions...")
        full_proba = self.calibrated_model.predict_proba(X_full)[:, 1]
        full_pred = (full_proba >= self.optimal_threshold).astype(int)

        # Add predictions to dataframe
        full_dataset["lr_l2_E_probability"] = full_proba
        full_dataset["lr_l2_E_prediction"] = full_pred
        full_dataset["lr_l2_E_threshold"] = self.optimal_threshold
        full_dataset["model_config"] = f"C={self.final_config['C']}_sigmoid_calibrated"

        logger.info(f"Predictions generated:")
        logger.info(
            f"  Positive predictions: {full_pred.sum()}/{len(full_pred)} ({100*full_pred.sum()/len(full_pred):.1f}%)"
        )
        logger.info(f"  Mean probability: {full_proba.mean():.4f}")
        logger.info(
            f"  Probability range: [{full_proba.min():.4f}, {full_proba.max():.4f}]"
        )

        return full_dataset

    def save_predictions(
        self, dataset_with_predictions: pd.DataFrame, oof_results: Dict
    ) -> None:
        """Save final predictions and model info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Save full dataset with predictions
        predictions_file = (
            self.output_dir / f"lr_l2_E_FINAL_Predictions_{timestamp}.jsonl"
        )

        logger.info(f"Saving predictions to {predictions_file}...")
        with open(predictions_file, "w") as f:
            for _, row in dataset_with_predictions.iterrows():
                row_dict = row.to_dict()
                # Convert numpy types to native Python types for JSON serialization
                for key, value in row_dict.items():
                    if isinstance(value, (np.int64, np.int32)):
                        row_dict[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        row_dict[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        row_dict[key] = value.tolist()

                f.write(
                    _json.dumps(row_dict)
                    if not ORJSON_AVAILABLE
                    else _json.dumps(row_dict).decode("utf-8")
                )
                f.write("\n")

        # Save model metadata and results
        model_info = {
            "model_name": "lr_l2_E_FINAL",
            "timestamp": timestamp,
            "configuration": self.final_config,
            "calibration_method": self.calibration_method,
            "optimal_threshold": self.optimal_threshold,
            "oof_test_results": oof_results,
            "training_data_size": len(
                dataset_with_predictions[
                    dataset_with_predictions["data_split"] != "oof_test"
                ]
            ),
            "total_predictions": len(dataset_with_predictions),
            "feature_config": "embeddings_only_768D",
            "embedding_feature": EMBEDDING_FEATURE,
        }

        model_info_file = self.output_dir / f"lr_l2_E_FINAL_Model_Info_{timestamp}.json"
        with open(model_info_file, "w") as f:
            if ORJSON_AVAILABLE:
                f.write(_dumps(model_info).decode("utf-8"))
            else:
                _json.dump(model_info, f, indent=2, default=str)

        logger.info(f"Model info saved to {model_info_file}")
        logger.info(f"Predictions saved to {predictions_file}")

        # Save summary statistics
        summary = {
            "total_samples": len(dataset_with_predictions),
            "positive_predictions": int(
                dataset_with_predictions["lr_l2_E_prediction"].sum()
            ),
            "positive_rate": float(
                dataset_with_predictions["lr_l2_E_prediction"].mean()
            ),
            "mean_probability": float(
                dataset_with_predictions["lr_l2_E_probability"].mean()
            ),
            "median_probability": float(
                dataset_with_predictions["lr_l2_E_probability"].median()
            ),
            "probability_percentiles": {
                "p90": float(
                    dataset_with_predictions["lr_l2_E_probability"].quantile(0.9)
                ),
                "p95": float(
                    dataset_with_predictions["lr_l2_E_probability"].quantile(0.95)
                ),
                "p99": float(
                    dataset_with_predictions["lr_l2_E_probability"].quantile(0.99)
                ),
            },
        }

        summary_file = self.output_dir / f"lr_l2_E_FINAL_Summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            if ORJSON_AVAILABLE:
                f.write(_dumps(summary).decode("utf-8"))
            else:
                _json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary statistics saved to {summary_file}")

    def run_final_pipeline(self) -> None:
        """Run the complete final model training and inference pipeline."""
        logger.info("🚀 STARTING FINAL lr_l2_E PRODUCTION PIPELINE")
        logger.info("=" * 80)

        # 1. Load all training data
        train_df, oof_test_df = self.load_all_training_data()

        # 2. Prepare training features
        X_train = self.prepare_embeddings_features(train_df, fit_scaler=True)
        y_train = train_df["outcome_bin"].values

        logger.info(f"Training data: {X_train.shape}, Labels: {np.bincount(y_train)}")

        # 3. Train final model
        self.train_final_model(X_train, y_train)

        # 4. Evaluate on OOF test if available
        oof_results = {}
        if not oof_test_df.empty and "outcome_bin" in oof_test_df.columns:
            X_oof = self.prepare_embeddings_features(oof_test_df, fit_scaler=False)
            y_oof = oof_test_df["outcome_bin"].values
            oof_results = self.evaluate_on_oof_test(X_oof, y_oof)

        # 5. Infer on entire dataset
        dataset_with_predictions = self.infer_on_entire_dataset()

        # 6. Save everything
        self.save_predictions(dataset_with_predictions, oof_results)

        logger.info("=" * 80)
        logger.info("✅ FINAL lr_l2_E PRODUCTION PIPELINE COMPLETED")
        logger.info("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train final lr_l2_E model and infer on entire dataset"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to k-fold data directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for predictions"
    )

    args = parser.parse_args()

    # Create trainer and run pipeline
    trainer = FinalModelTrainer(data_dir=args.data_dir, output_dir=args.output_dir)

    trainer.run_final_pipeline()


if __name__ == "__main__":
    main()
