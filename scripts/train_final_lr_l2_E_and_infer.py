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
from sklearn.metrics import matthews_corrcoef, recall_score

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

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        target_recall: float = 0.10,
        topk_percent: float | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Threshold tuning knobs
        self.target_recall = float(target_recall)
        self.topk_percent = float(topk_percent) if topk_percent is not None else None
        # Storage for auxiliary thresholds computed from OOF
        self.recall_threshold = None

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

    def find_threshold_for_recall(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_recall: float,
        n_steps: int = 401,
    ) -> float:
        """Smallest threshold achieving at least target_recall; closest otherwise."""
        thresholds = np.linspace(0.0, 1.0, n_steps)
        best_thr, best_gap, found = 0.5, 1e9, False
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0.0)
            gap = abs(rec - target_recall)
            if rec >= target_recall and gap < best_gap:
                best_thr, best_gap, found = float(thr), gap, True
        if not found:
            # fallback: closest recall overall
            for thr in thresholds:
                y_pred = (y_prob >= thr).astype(int)
                rec = recall_score(y_true, y_pred, zero_division=0.0)
                gap = abs(rec - target_recall)
                if gap < best_gap:
                    best_thr, best_gap = float(thr), gap
        return best_thr

    def find_threshold_for_topk_percent(
        self, y_prob: np.ndarray, topk_percent: float
    ) -> float:
        """Quantile cutoff to surface the top-k% most confident as positive."""
        topk_percent = max(0.0, min(1.0, float(topk_percent)))
        if topk_percent <= 0.0:
            return 1.0
        q = 1.0 - topk_percent
        return float(np.quantile(y_prob, q))

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

        # --- Auxiliary thresholds for higher candidate yield ---
        # Threshold to hit target recall on OOF (for reporting + optional production use)
        try:
            thr_recall = self.find_threshold_for_recall(
                y_oof, test_proba, self.target_recall
            )
            self.recall_threshold = thr_recall
            y_pred_recall = (test_proba >= thr_recall).astype(int)
            tp_r = ((y_pred_recall == 1) & (y_oof == 1)).sum()
            fp_r = ((y_pred_recall == 1) & (y_oof == 0)).sum()
            tn_r = ((y_pred_recall == 0) & (y_oof == 0)).sum()
            fn_r = ((y_pred_recall == 0) & (y_oof == 1)).sum()
            precision_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
            recall_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
            mcc_r = (
                matthews_corrcoef(y_oof, y_pred_recall)
                if len(np.unique(y_pred_recall)) > 1
                else 0.0
            )
            logger.info(
                f"  Aux threshold (recall@{self.target_recall:.2f}) on OOF: thr={thr_recall:.4f} | P={precision_r:.4f}, R={recall_r:.4f}, MCC={mcc_r:.4f} | PosRate={(y_pred_recall.mean()*100):.2f}%"
            )
        except Exception as e:
            logger.warning(f"Failed to compute recall-target threshold: {e}")
            thr_recall = None

        # If top-k% requested, estimate top-k threshold from OOF for logging
        thr_topk = None
        if self.topk_percent is not None:
            try:
                thr_topk = self.find_threshold_for_topk_percent(
                    test_proba, self.topk_percent
                )
                y_pred_topk = (test_proba >= thr_topk).astype(int)
                logger.info(
                    f"  Aux threshold (top-k={self.topk_percent*100:.2f}%) on OOF: thr={thr_topk:.4f} | PosRate={(y_pred_topk.mean()*100):.2f}%"
                )
            except Exception as e:
                logger.warning(f"Failed to compute top-k threshold: {e}")

        # Augment results
        results.update(
            {
                "aux_thresholds": {
                    "mcc": self.optimal_threshold,
                    "recall_target": (
                        float(thr_recall) if thr_recall is not None else None
                    ),
                    "topk_percent": float(thr_topk) if thr_topk is not None else None,
                    "target_recall": self.target_recall,
                    "topk": self.topk_percent,
                }
            }
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
        # Strict (MCC-optimal) threshold
        full_pred_strict = (full_proba >= self.optimal_threshold).astype(int)

        # Recall-target threshold (computed on OOF); if absent, fall back to strict
        thr_recall = self.recall_threshold
        full_pred_recall = None
        if thr_recall is not None:
            full_pred_recall = (full_proba >= thr_recall).astype(int)

        # Top-k% threshold computed on the full dataset distribution
        thr_topk = None
        full_pred_topk = None
        if self.topk_percent is not None:
            thr_topk = self.find_threshold_for_topk_percent(
                full_proba, self.topk_percent
            )
            full_pred_topk = (full_proba >= thr_topk).astype(int)

        # Add predictions to dataframe
        full_dataset["lr_l2_E_probability"] = full_proba
        full_dataset["lr_l2_E_prediction_strict"] = full_pred_strict
        full_dataset["lr_l2_E_threshold_strict"] = self.optimal_threshold
        if full_pred_recall is not None:
            full_dataset["lr_l2_E_prediction_recallT"] = full_pred_recall
            full_dataset["lr_l2_E_threshold_recallT"] = thr_recall
        if full_pred_topk is not None:
            full_dataset["lr_l2_E_prediction_topk"] = full_pred_topk
            full_dataset["lr_l2_E_threshold_topk"] = thr_topk
        full_dataset["model_config"] = f"C={self.final_config['C']}_sigmoid_calibrated"

        logger.info("Predictions generated:")
        logger.info(
            f"  Strict@MCC: {full_pred_strict.sum()}/{len(full_pred_strict)} ({100*full_pred_strict.mean():.1f}%)"
        )
        if full_pred_recall is not None:
            logger.info(
                f"  Recall@{self.target_recall:.2f}: {full_pred_recall.sum()}/{len(full_pred_recall)} ({100*full_pred_recall.mean():.1f}%)"
            )
        if full_pred_topk is not None:
            logger.info(
                f"  Top-k@{self.topk_percent*100:.2f}%: {full_pred_topk.sum()}/{len(full_pred_topk)} ({100*full_pred_topk.mean():.1f}%)"
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
            "aux_thresholds": {
                "recall_target": (
                    float(self.recall_threshold)
                    if self.recall_threshold is not None
                    else None
                ),
                "target_recall": self.target_recall,
                "topk_percent": self.topk_percent,
            },
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
            "positive_predictions": (
                int(
                    dataset_with_predictions.get(
                        "lr_l2_E_prediction", pd.Series(dtype=int)
                    ).sum()
                )
                if "lr_l2_E_prediction" in dataset_with_predictions.columns
                else None
            ),
            "positive_rate": (
                float(
                    dataset_with_predictions.get(
                        "lr_l2_E_prediction", pd.Series(dtype=float)
                    ).mean()
                )
                if "lr_l2_E_prediction" in dataset_with_predictions.columns
                else None
            ),
            "positive_predictions_recallT": (
                int(
                    dataset_with_predictions.get(
                        "lr_l2_E_prediction_recallT", pd.Series(dtype=int)
                    ).sum()
                )
                if "lr_l2_E_prediction_recallT" in dataset_with_predictions.columns
                else None
            ),
            "positive_rate_recallT": (
                float(
                    dataset_with_predictions.get(
                        "lr_l2_E_prediction_recallT", pd.Series(dtype=float)
                    ).mean()
                )
                if "lr_l2_E_prediction_recallT" in dataset_with_predictions.columns
                else None
            ),
            "positive_predictions_topk": (
                int(
                    dataset_with_predictions.get(
                        "lr_l2_E_prediction_topk", pd.Series(dtype=int)
                    ).sum()
                )
                if "lr_l2_E_prediction_topk" in dataset_with_predictions.columns
                else None
            ),
            "positive_rate_topk": (
                float(
                    dataset_with_predictions.get(
                        "lr_l2_E_prediction_topk", pd.Series(dtype=float)
                    ).mean()
                )
                if "lr_l2_E_prediction_topk" in dataset_with_predictions.columns
                else None
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
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.10,
        help="Target recall for auxiliary threshold (on OOF)",
    )
    parser.add_argument(
        "--topk-percent",
        type=float,
        default=None,
        help="Top-k fraction (0..1) for auxiliary threshold (on full dataset)",
    )

    args = parser.parse_args()

    # Create trainer and run pipeline
    trainer = FinalModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_recall=args.target_recall,
        topk_percent=args.topk_percent,
    )

    trainer.run_final_pipeline()


if __name__ == "__main__":
    main()
