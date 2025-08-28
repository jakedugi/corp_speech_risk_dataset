#!/usr/bin/env python3
"""
Frozen LegalBERT MLP Training - OPTIMIZED PYTORCH VERSION

CRITICAL IMPROVEMENTS:
- PyTorch MLP models for E and E+3 variants
- Frozen LegalBERT embeddings with shallow MLP
- Court-based identity suppression (reused from existing logic)
- Early stopping on dev-suppressed MCC
- StandardScaler for feature normalization
- orjson optimization and efficient data loading
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

warnings.filterwarnings("ignore")

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# Sklearn imports (reusing existing preprocessing and metrics)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

# Fast JSON loading with orjson optimization (reused from existing)
try:
    import orjson as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data)

    def _loads_str(data: str) -> Any:
        return _json.loads(data.encode("utf-8"))

    def _dumps(obj: Any) -> bytes:
        return _json.dumps(obj, option=_json.OPT_SERIALIZE_NUMPY)

    ORJSON_AVAILABLE = True
except ImportError:
    import json as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data.decode("utf-8"))

    def _loads_str(data: str) -> Any:
        return _json.loads(data)

    def _dumps(obj: Any) -> str:
        return _json.dumps(obj, default=str)

    ORJSON_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Runtime knobs for Colab T4 / CPU fallback ----
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # Allow TF32 on Ampere/Turing; safe and faster for matmul
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
# Prevent thread oversubscription on Colab
try:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
except Exception:
    pass

# Validated features (the 3 that passed fingerprinting)
VALIDATED_SCALARS = [
    "feat_new4_neutral_to_disclaimer_transition_rate",
    "interpretable_lex_disclaimers_present",
    "feat_interact_hedge_x_guarantee",
]

EMBEDDING_FEATURE = "legal_bert_emb"  # 768-dimensional Legal-BERT embeddings


class FrozenEmbedMLP(nn.Module):
    """
    Shallow MLP for frozen LegalBERT embeddings with optional scalar features.

    E variant: LayerNorm(768) → Dropout(0.2) → Linear(768→256) → GELU → Dropout(0.2) → Linear(256→1)
    E+3 variant: Late fusion with projected scalars
    """

    def __init__(
        self,
        emb_dim: int = 768,
        use_scalars: bool = False,
        scalar_proj_dim: int = 16,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_scalars = use_scalars
        self.emb_dim = emb_dim

        if use_scalars:
            # Scalar projection with higher dropout for regularization
            self.scalar_proj = nn.Sequential(
                nn.Linear(3, scalar_proj_dim),
                nn.GELU(),
                nn.Dropout(0.33),  # Higher dropout for scalar features
            )
            in_dim = emb_dim + scalar_proj_dim
        else:
            self.scalar_proj = None
            in_dim = emb_dim

        # Main MLP with LayerNorm for stability
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, emb: torch.Tensor, scalars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            emb: (batch_size, emb_dim) - LegalBERT embeddings
            scalars: (batch_size, 3) - Optional scalar features for E+3 variant

        Returns:
            logits: (batch_size,) - Raw logits for binary classification
        """
        if self.use_scalars:
            if scalars is None:
                raise ValueError("scalars required when use_scalars=True")
            p = self.scalar_proj(scalars)
            x = torch.cat([emb, p], dim=1)
        else:
            x = emb

        logits = self.net(x).squeeze(1)  # Remove last dimension
        return logits


class PyTorchMLPEvaluator:
    """PyTorch MLP training and evaluation with court-based suppression (reusing existing logic)."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        fold: int = 4,
        device: Optional[str] = None,
        grid_search: bool = False,
        amp: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fold = fold
        self.grid_search = grid_search
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Mixed precision (AMP) toggle: default True on CUDA unless explicitly disabled
        if amp is None:
            self.amp = self.device.type == "cuda"
        else:
            self.amp = bool(amp) and (self.device.type == "cuda")
        logger.info(f"Using device: {self.device}")
        logger.info(f"AMP (mixed precision) enabled: {self.amp}")

        # Load metadata (reusing existing logic)
        self.metadata = self.load_metadata()

        # Results storage
        self.results = {}
        self.test_predictions = {}
        self.optimal_thresholds = {}

        # Court-based identity suppression storage (reusing existing logic)
        self.court_means_emb_ = None
        self.court_means_scal_ = None
        self.circuit_means_emb_ = None
        self.circuit_means_scal_ = None
        self.global_mean_emb_ = None
        self.global_mean_scal_ = None

        logger.info(f"Initialized PyTorch MLP evaluator for fold {fold}")
        logger.info(f"Data dir: {data_dir}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Grid search: {grid_search}")
        logger.info(f"orjson available: {ORJSON_AVAILABLE}")

    def load_metadata(self) -> Dict:
        """Load the per-fold metadata with class weights (reusing existing logic)."""
        metadata_path = self.data_dir / "per_fold_metadata.json"
        if metadata_path.exists():
            if ORJSON_AVAILABLE:
                with open(metadata_path, "rb") as f:
                    metadata = _loads_bytes(f.read())
            else:
                with open(metadata_path, "r") as f:
                    metadata = _json.load(f)
            logger.info(f"Loaded metadata: {metadata.get('methodology', 'unknown')}")
            return metadata
        else:
            logger.warning("No metadata found - using default weights")
            return {}

    def get_fold_class_weights(self) -> Union[Dict[int, float], str]:
        """Get the class weights for the current fold from metadata (reusing existing logic)."""
        if (
            "weights" in self.metadata
            and f"fold_{self.fold}" in self.metadata["weights"]
        ):
            class_weights = self.metadata["weights"][f"fold_{self.fold}"][
                "class_weights"
            ]
            # Convert string keys to int keys
            return {int(k): float(v) for k, v in class_weights.items()}
        else:
            logger.warning(
                f"No class weights found for fold {self.fold}, using balanced"
            )
            return "balanced"

    def load_temporal_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the existing temporal train/dev/test splits with orjson optimization (reusing existing logic)."""
        logger.info("Loading existing temporal splits with orjson...")

        # Load training data for this fold
        train_path = self.data_dir / f"fold_{self.fold}" / "train.jsonl"
        dev_path = self.data_dir / f"fold_{self.fold}" / "dev.jsonl"
        test_path = self.data_dir / "oof_test" / "test.jsonl"

        def load_jsonl_fast(path: Path) -> pd.DataFrame:
            """Fast JSONL loading with orjson (reusing existing logic)."""
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
                                        _loads_str(
                                            line_bytes.decode("utf-8", errors="ignore")
                                        )
                                    )
                                except Exception:
                                    logger.warning(
                                        f"Failed to parse line in {path}: {e}"
                                    )
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

        # Load with optimized JSONL loading
        train_df = load_jsonl_fast(train_path)
        dev_df = load_jsonl_fast(dev_path) if dev_path.exists() else train_df
        test_df = load_jsonl_fast(test_path) if test_path.exists() else dev_df

        logger.info(
            f"Loaded temporal splits: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}"
        )

        # Verify the binary outcome distribution
        for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            if "outcome_bin" in df.columns:
                dist = df["outcome_bin"].value_counts().sort_index()
                logger.info(f"{name} outcome distribution: {dist.to_dict()}")

        return train_df, dev_df, test_df

    def compute_suppression_means(
        self,
        X_emb: np.ndarray,
        court_ids: np.ndarray,
        X_scal: Optional[np.ndarray] = None,
    ):
        """Train-only: compute per-court and per-circuit means by SAMPLE counts (efficient, leak-free)."""
        logger.info("Computing court/circuit suppression means (train-only)...")
        self.court_means_emb_, self.circuit_means_emb_ = {}, {}
        self.court_means_scal_, self.circuit_means_scal_ = (
            ({}, {}) if X_scal is not None else (None, None)
        )

        # Court means
        court_unique = pd.Series(court_ids).fillna("").values
        for c in np.unique(court_unique):
            if c == "":
                continue
            idx = court_unique == c
            n = int(idx.sum())
            if n >= 10:
                self.court_means_emb_[c] = X_emb[idx].mean(axis=0)
                if X_scal is not None:
                    self.court_means_scal_[c] = X_scal[idx].mean(axis=0)

        # Circuit accumulators by SAMPLE (letters only)
        circ_sum_emb, circ_cnt = {}, {}
        circ_sum_scal = {} if X_scal is not None else None
        for i, c in enumerate(court_unique):
            if c == "":
                continue
            circ = "".join(ch for ch in str(c).lower() if ch.isalpha())
            if not circ:
                continue
            if circ not in circ_sum_emb:
                circ_sum_emb[circ] = X_emb[i].copy()
                circ_cnt[circ] = 1
                if X_scal is not None:
                    circ_sum_scal[circ] = X_scal[i].copy()
            else:
                circ_sum_emb[circ] += X_emb[i]
                circ_cnt[circ] += 1
                if X_scal is not None:
                    circ_sum_scal[circ] += X_scal[i]

        for circ, cnt in circ_cnt.items():
            if cnt >= 10:
                key = f"circuit_{circ}"
                self.circuit_means_emb_[key] = circ_sum_emb[circ] / cnt
                if X_scal is not None:
                    self.circuit_means_scal_[key] = circ_sum_scal[circ] / cnt

        self.global_mean_emb_ = X_emb.mean(axis=0)
        if X_scal is not None:
            self.global_mean_scal_ = X_scal.mean(axis=0)

        logger.info(
            f"Computed court means: {len(self.court_means_emb_)} (min size: 10)"
        )
        logger.info(
            f"Computed circuit means: {len(self.circuit_means_emb_)} (min size: 10)"
        )

    def apply_identity_suppression(
        self,
        X_emb: np.ndarray,
        court_ids: np.ndarray,
        X_scal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Eval-time: subtract train means; fallback hierarchy (reusing existing logic)."""
        if self.court_means_emb_ is None or self.global_mean_emb_ is None:
            raise RuntimeError("Call compute_suppression_means on train split first.")

        # Enhanced fallback hierarchy: court → circuit → global
        mu_emb = []
        mu_scal = [] if X_scal is not None else None
        fallback_stats = {"court": 0, "circuit": 0, "global": 0}

        for court_id in court_ids:
            if court_id in self.court_means_emb_:
                mu_emb.append(self.court_means_emb_[court_id])
                if X_scal is not None and self.court_means_scal_ is not None:
                    mu_scal.append(self.court_means_scal_[court_id])
                fallback_stats["court"] += 1
            else:
                # Extract circuit from court_id letters for fallback
                circuit = (
                    "".join([c for c in str(court_id).lower() if c.isalpha()])
                    if court_id
                    else "unk"
                )
                circuit_key = f"circuit_{circuit}"

                if circuit_key in self.circuit_means_emb_:
                    mu_emb.append(self.circuit_means_emb_[circuit_key])
                    if X_scal is not None and self.circuit_means_scal_ is not None:
                        mu_scal.append(self.circuit_means_scal_[circuit_key])
                    fallback_stats["circuit"] += 1
                else:
                    mu_emb.append(self.global_mean_emb_)
                    if X_scal is not None:
                        mu_scal.append(self.global_mean_scal_)
                    fallback_stats["global"] += 1

        # Log fallback usage for transparency
        total_samples = len(court_ids)
        if total_samples > 0:
            court_pct = (fallback_stats["court"] / total_samples) * 100
            circuit_pct = (fallback_stats["circuit"] / total_samples) * 100
            global_pct = (fallback_stats["global"] / total_samples) * 100
            logger.info(
                f"Identity suppression fallback: Court={court_pct:.1f}%, Circuit={circuit_pct:.1f}%, Global={global_pct:.1f}%"
            )

        X_emb_suppressed = X_emb - np.stack(mu_emb, axis=0)
        X_scal_suppressed = None
        if X_scal is not None and mu_scal is not None:
            X_scal_suppressed = X_scal - np.stack(mu_scal, axis=0)

        return X_emb_suppressed, X_scal_suppressed

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_config: str,
        emb_scaler: Optional[StandardScaler] = None,
        scal_scaler: Optional[StandardScaler] = None,
        fit: bool = False,
    ) -> Tuple[
        np.ndarray, Optional[np.ndarray], StandardScaler, Optional[StandardScaler]
    ]:
        """Prepare features based on configuration."""
        logger.info(f"Preparing features for config: {feature_config}")

        # Extract embeddings
        if EMBEDDING_FEATURE in df.columns:
            embeddings = np.asarray(df[EMBEDDING_FEATURE].tolist(), dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[1] != 768:
                raise ValueError(
                    f"Expected {EMBEDDING_FEATURE} to be (N,768) got {embeddings.shape}"
                )
        else:
            raise ValueError(f"Embedding feature {EMBEDDING_FEATURE} not found")

        # Extract scalars if needed
        scalars = None
        if feature_config == "E+3":
            scalar_features = []
            for feature in VALIDATED_SCALARS:
                if feature in df.columns:
                    scalar_features.append(
                        df[feature].to_numpy(dtype=np.float32, copy=False)
                    )
                else:
                    logger.warning(
                        f"Feature {feature} not found in data; filling zeros"
                    )
                    scalar_features.append(np.zeros(len(df), dtype=np.float32))
            scalars = np.column_stack(scalar_features).astype(np.float32, copy=False)
            if scalars.ndim != 2 or scalars.shape[1] != 3:
                raise ValueError(f"Expected 3 scalar features, got {scalars.shape}")

        # Scale features
        if fit:
            emb_scaler = StandardScaler()
            embeddings_scaled = emb_scaler.fit_transform(embeddings)
            logger.info(f"Fitted embedding scaler: {embeddings.shape}")

            if scalars is not None:
                scal_scaler = StandardScaler()
                scalars_scaled = scal_scaler.fit_transform(scalars)
                logger.info(f"Fitted scalar scaler: {scalars.shape}")
            else:
                scalars_scaled = None
        else:
            if emb_scaler is None:
                raise ValueError("Embedding scaler required when fit=False")
            embeddings_scaled = emb_scaler.transform(embeddings)

            if scalars is not None:
                if scal_scaler is None:
                    raise ValueError(
                        "Scalar scaler required when fit=False and scalars present"
                    )
                scalars_scaled = scal_scaler.transform(scalars)
            else:
                scalars_scaled = None

        logger.info(f"Embeddings shape: {embeddings_scaled.shape}")
        if scalars_scaled is not None:
            logger.info(f"Scalars shape: {scalars_scaled.shape}")

        return embeddings_scaled, scalars_scaled, emb_scaler, scal_scaler

    def calculate_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (reusing existing logic)."""
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
        self, y_true: np.ndarray, y_prob: np.ndarray, n_steps: int = 201
    ) -> Tuple[float, float]:
        """Find threshold that maximizes MCC with dense search (reusing existing logic)."""
        thresholds = np.linspace(0.0, 1.0, n_steps)
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
        self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float
    ) -> Dict[str, Any]:
        """Calculate precision, recall, specificity at given threshold (reusing existing logic)."""
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)

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

    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for PyTorch models."""
        if self.grid_search:
            # Full grid search
            return {
                "E": {
                    "hidden": [128, 256],
                    "dropout": [0.1, 0.2],
                    "lr": [1e-3, 3e-4],
                    "weight_decay": [1e-4, 1e-3],
                    "batch_size": [128],
                    "max_epochs": [30],
                    "patience": [5],
                },
                "E+3": {
                    "hidden": [128, 256],
                    "dropout": [0.1, 0.2],
                    "lr": [1e-3, 3e-4],
                    "weight_decay": [1e-4, 1e-3],
                    "batch_size": [128],
                    "max_epochs": [30],
                    "patience": [5],
                    "scalar_proj_dim": [16],  # Fixed as specified
                },
            }
        else:
            # Quick single configuration
            return {
                "E": {
                    "hidden": [256],
                    "dropout": [0.2],
                    "lr": [1e-3],
                    "weight_decay": [1e-4],
                    "batch_size": [128],
                    "max_epochs": [30],
                    "patience": [5],
                },
                "E+3": {
                    "hidden": [256],
                    "dropout": [0.2],
                    "lr": [1e-3],
                    "weight_decay": [1e-4],
                    "batch_size": [128],
                    "max_epochs": [30],
                    "patience": [5],
                    "scalar_proj_dim": [16],
                },
            }

    def train_mlp_model(
        self,
        X_emb_train: np.ndarray,
        y_train: np.ndarray,
        court_train: np.ndarray,
        X_emb_dev: np.ndarray,
        y_dev: np.ndarray,
        court_dev: np.ndarray,
        X_scal_train: Optional[np.ndarray] = None,
        X_scal_dev: Optional[np.ndarray] = None,
        **hyperparams,
    ) -> Dict[str, Any]:
        """Train MLP model with early stopping on dev-suppressed MCC."""

        # Extract hyperparameters with defaults
        hidden = hyperparams.get("hidden", 256)
        dropout = hyperparams.get("dropout", 0.2)
        lr = hyperparams.get("lr", 1e-3)
        weight_decay = hyperparams.get("weight_decay", 1e-4)
        batch_size = hyperparams.get("batch_size", 128)
        max_epochs = hyperparams.get("max_epochs", 30)
        patience = hyperparams.get("patience", 5)
        scalar_proj_dim = hyperparams.get("scalar_proj_dim", 16)

        use_scalars = X_scal_train is not None
        emb_dim = X_emb_train.shape[1]

        logger.info(
            f"Training MLP: hidden={hidden}, dropout={dropout}, lr={lr}, use_scalars={use_scalars}"
        )

        # Compute court suppression baselines (train-only)
        self.compute_suppression_means(X_emb_train, court_train, X_scal_train)
        X_emb_dev_sup, X_scal_dev_sup = self.apply_identity_suppression(
            X_emb_dev, court_dev, X_scal_dev
        )

        # Convert to PyTorch tensors
        def to_tensor(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=self.device)

        X_emb_train_t = to_tensor(X_emb_train)
        y_train_t = to_tensor(y_train)
        # Ensure binary labels in {0,1}
        if not set(np.unique(y_train)).issubset({0, 1}):
            raise ValueError("y_train must be binary in {0,1}")
        X_emb_dev_t = to_tensor(X_emb_dev)
        X_emb_dev_sup_t = to_tensor(X_emb_dev_sup)
        y_dev_t = to_tensor(y_dev)

        if use_scalars:
            X_scal_train_t = to_tensor(X_scal_train)
            X_scal_dev_t = to_tensor(X_scal_dev)
            X_scal_dev_sup_t = to_tensor(X_scal_dev_sup)

            # Create dataset with scalars
            train_dataset = TensorDataset(X_emb_train_t, X_scal_train_t, y_train_t)
        else:
            X_scal_train_t = X_scal_dev_t = X_scal_dev_sup_t = None

            # Create dataset without scalars
            train_dataset = TensorDataset(X_emb_train_t, y_train_t)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # Initialize model
        model = FrozenEmbedMLP(
            emb_dim=emb_dim,
            use_scalars=use_scalars,
            scalar_proj_dim=scalar_proj_dim,
            hidden=hidden,
            dropout=dropout,
        ).to(self.device)

        # Loss function with class weights
        class_weights = self.get_fold_class_weights()
        if isinstance(class_weights, dict):
            pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
        else:
            # Calculate from data
            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            pos_weight = float(neg / max(pos, 1))

        pos_weight_tensor = torch.tensor(pos_weight, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # AMP scaler (active only on CUDA when self.amp is True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # Early stopping on dev-suppressed MCC
        best_mcc = -1.0
        best_state = None
        best_threshold = 0.5
        no_improve = 0

        model.train()
        for epoch in range(1, max_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if use_scalars:
                    batch_emb, batch_scal, batch_y = batch
                    with torch.cuda.amp.autocast(enabled=self.amp):
                        logits = model(batch_emb, batch_scal)
                        loss = criterion(logits, batch_y)
                else:
                    batch_emb, batch_y = batch
                    with torch.cuda.amp.autocast(enabled=self.amp):
                        logits = model(batch_emb)
                        loss = criterion(logits, batch_y)

                # Scaled backward + safe clipping
                scaler.scale(loss).backward()
                if self.amp:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

            # Evaluate on dev-suppressed (selection view)
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.amp):
                    if use_scalars:
                        logits_sup = model(X_emb_dev_sup_t, X_scal_dev_sup_t)
                    else:
                        logits_sup = model(X_emb_dev_sup_t)

                probs_sup = torch.sigmoid(logits_sup).cpu().numpy()

            threshold, mcc = self.find_mcc_optimal_threshold(y_dev, probs_sup)

            logger.info(
                f"Epoch {epoch}: loss={epoch_loss/num_batches:.4f}, dev_mcc_suppressed={mcc:.4f}, threshold={threshold:.4f}"
            )

            if mcc > best_mcc:
                best_mcc = mcc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_threshold = threshold
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

            model.train()

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Return training artifacts
        return {
            "model": model,
            "best_threshold": best_threshold,
            "best_mcc_dev_suppressed": best_mcc,
            "hyperparams": hyperparams,
            "use_scalars": use_scalars,
        }

    def evaluate_model_with_suppressed_cv(
        self,
        artifacts: Dict[str, Any],
        X_emb_train: np.ndarray,
        y_train: np.ndarray,
        court_train: np.ndarray,
        X_emb_dev: np.ndarray,
        y_dev: np.ndarray,
        court_dev: np.ndarray,
        X_scal_train: Optional[np.ndarray] = None,
        X_scal_dev: Optional[np.ndarray] = None,
        model_name: str = "mlp",
    ) -> Dict[str, Any]:
        """Evaluate model using temporal train/dev split with SUPPRESSED CV selection."""
        logger.info(f"Evaluating {model_name} with suppressed CV selection...")

        model = artifacts["model"]
        threshold = artifacts["best_threshold"]

        # Compute suppression (already done during training, but ensure consistency)
        self.compute_suppression_means(X_emb_train, court_train, X_scal_train)
        X_emb_dev_sup, X_scal_dev_sup = self.apply_identity_suppression(
            X_emb_dev, court_dev, X_scal_dev
        )

        # Convert to tensors
        def to_tensor(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=self.device)

        X_emb_dev_t = to_tensor(X_emb_dev)
        X_emb_dev_sup_t = to_tensor(X_emb_dev_sup)

        if X_scal_dev is not None:
            X_scal_dev_t = to_tensor(X_scal_dev)
            X_scal_dev_sup_t = to_tensor(X_scal_dev_sup)
        else:
            X_scal_dev_t = X_scal_dev_sup_t = None

        # Get predictions
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp):
                if artifacts["use_scalars"]:
                    logits_raw = model(X_emb_dev_t, X_scal_dev_t)
                    logits_sup = model(X_emb_dev_sup_t, X_scal_dev_sup_t)
                else:
                    logits_raw = model(X_emb_dev_t)
                    logits_sup = model(X_emb_dev_sup_t)

            probs_raw = torch.sigmoid(logits_raw).cpu().numpy()
            probs_sup = torch.sigmoid(logits_sup).cpu().numpy()

        # Calculate metrics for both views
        # RAW metrics
        dev_auc_raw = (
            roc_auc_score(y_dev, probs_raw) if len(np.unique(y_dev)) > 1 else 0.5
        )
        dev_pr_auc_raw = average_precision_score(y_dev, probs_raw)
        optimal_threshold_raw, dev_mcc_raw = self.find_mcc_optimal_threshold(
            y_dev, probs_raw
        )
        operating_metrics_raw = self.calculate_operating_point_metrics(
            y_dev, probs_raw, optimal_threshold_raw
        )
        dev_brier_raw = brier_score_loss(y_dev, probs_raw)
        dev_ece_raw = self.calculate_ece(y_dev, probs_raw)

        # SUPPRESSED metrics (PRIMARY for selection)
        dev_auc_suppressed = (
            roc_auc_score(y_dev, probs_sup) if len(np.unique(y_dev)) > 1 else 0.5
        )
        dev_pr_auc_suppressed = average_precision_score(y_dev, probs_sup)
        optimal_threshold_suppressed, dev_mcc_suppressed = (
            self.find_mcc_optimal_threshold(y_dev, probs_sup)
        )
        operating_metrics_suppressed = self.calculate_operating_point_metrics(
            y_dev, probs_sup, optimal_threshold_suppressed
        )
        dev_brier_suppressed = brier_score_loss(y_dev, probs_sup)
        dev_ece_suppressed = self.calculate_ece(y_dev, probs_sup)

        # Store optimal threshold from training (already optimized on suppressed view)
        self.optimal_thresholds[model_name] = threshold

        # Calculate delta (raw - suppressed)
        delta_mcc = dev_mcc_raw - dev_mcc_suppressed
        delta_auc = dev_auc_raw - dev_auc_suppressed

        logger.info(f"  Dev RAW: MCC={dev_mcc_raw:.4f}, AUC={dev_auc_raw:.4f}")
        logger.info(
            f"  Dev SUPPRESSED: MCC={dev_mcc_suppressed:.4f}, AUC={dev_auc_suppressed:.4f} (SELECTION)"
        )
        logger.info(f"  Delta: ΔMCC={delta_mcc:.4f}, ΔAUC={delta_auc:.4f}")

        return {
            # RAW dev metrics (for reporting)
            "dev_mcc_raw": dev_mcc_raw,
            "dev_auc_raw": dev_auc_raw,
            "dev_pr_auc_raw": dev_pr_auc_raw,
            "dev_brier_raw": dev_brier_raw,
            "dev_ece_raw": dev_ece_raw,
            # SUPPRESSED dev metrics (PRIMARY for selection)
            "dev_mcc_suppressed": dev_mcc_suppressed,  # PRIMARY SELECTION METRIC
            "dev_auc_suppressed": dev_auc_suppressed,
            "dev_pr_auc_suppressed": dev_pr_auc_suppressed,
            "dev_brier_suppressed": dev_brier_suppressed,
            "dev_ece_suppressed": dev_ece_suppressed,
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
            # Store best hyperparameters
            "best_params": artifacts["hyperparams"],
        }

    def evaluate_model_on_test(
        self,
        artifacts: Dict[str, Any],
        X_emb_train: np.ndarray,
        y_train: np.ndarray,
        court_train: np.ndarray,
        X_emb_test: np.ndarray,
        y_test: np.ndarray,
        court_test: np.ndarray,
        X_scal_train: Optional[np.ndarray] = None,
        X_scal_test: Optional[np.ndarray] = None,
        model_name: str = "mlp",
    ) -> Dict[str, Any]:
        """Evaluate model on final test set with both raw and suppressed views."""
        logger.info(f"Evaluating {model_name} on final test set (raw + suppressed)...")

        model = artifacts["model"]
        threshold = artifacts["best_threshold"]

        # Compute suppression means on training data
        self.compute_suppression_means(X_emb_train, court_train, X_scal_train)
        X_emb_test_sup, X_scal_test_sup = self.apply_identity_suppression(
            X_emb_test, court_test, X_scal_test
        )

        # Convert to tensors
        def to_tensor(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=self.device)

        X_emb_test_t = to_tensor(X_emb_test)
        X_emb_test_sup_t = to_tensor(X_emb_test_sup)

        if X_scal_test is not None:
            X_scal_test_t = to_tensor(X_scal_test)
            X_scal_test_sup_t = to_tensor(X_scal_test_sup)
        else:
            X_scal_test_t = X_scal_test_sup_t = None

        # Get predictions
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp):
                if artifacts["use_scalars"]:
                    logits_raw = model(X_emb_test_t, X_scal_test_t)
                    logits_sup = model(X_emb_test_sup_t, X_scal_test_sup_t)
                else:
                    logits_raw = model(X_emb_test_t)
                    logits_sup = model(X_emb_test_sup_t)

            probs_raw = torch.sigmoid(logits_raw).cpu().numpy()
            probs_sup = torch.sigmoid(logits_sup).cpu().numpy()

        # Calculate metrics for both views
        # RAW metrics
        test_operating_metrics_raw = self.calculate_operating_point_metrics(
            y_test, probs_raw, threshold
        )
        test_auc_raw = (
            roc_auc_score(y_test, probs_raw) if len(np.unique(y_test)) > 1 else 0.5
        )
        test_pr_auc_raw = average_precision_score(y_test, probs_raw)
        test_brier_raw = brier_score_loss(y_test, probs_raw)
        test_ece_raw = self.calculate_ece(y_test, probs_raw)

        # SUPPRESSED metrics
        test_operating_metrics_suppressed = self.calculate_operating_point_metrics(
            y_test, probs_sup, threshold
        )
        test_auc_suppressed = (
            roc_auc_score(y_test, probs_sup) if len(np.unique(y_test)) > 1 else 0.5
        )
        test_pr_auc_suppressed = average_precision_score(y_test, probs_sup)
        test_brier_suppressed = brier_score_loss(y_test, probs_sup)
        test_ece_suppressed = self.calculate_ece(y_test, probs_sup)

        # Delta calculations
        delta_mcc_test = (
            test_operating_metrics_raw["mcc"] - test_operating_metrics_suppressed["mcc"]
        )
        delta_auc_test = test_auc_raw - test_auc_suppressed

        # Store test predictions (convert numpy arrays to lists for JSON serialization)
        self.test_predictions[model_name] = {
            "predictions": (probs_raw >= threshold).astype(int).tolist(),
            "probabilities_raw": probs_raw.tolist(),
            "probabilities_suppressed": probs_sup.tolist(),
            "true_labels": y_test.tolist(),
        }

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
            "test_threshold_used": threshold,
            "best_params": artifacts["hyperparams"],
        }

    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation using existing temporal splits with PyTorch MLPs."""
        logger.info("=" * 80)
        logger.info("STARTING PYTORCH MLP TEMPORAL MODEL EVALUATION")
        logger.info("=" * 80)

        # Load existing temporal splits with orjson optimization
        train_df, dev_df, test_df = self.load_temporal_data()
        # Drop rows with missing or malformed embeddings
        for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            bad = (
                df[EMBEDDING_FEATURE].isna()
                if EMBEDDING_FEATURE in df.columns
                else pd.Series(False, index=df.index)
            )
            if bad.any():
                logger.warning(
                    f"Dropping {bad.sum()} rows with missing {EMBEDDING_FEATURE} from {name}"
                )
                df.drop(index=df[bad].index, inplace=True)

        # Get hyperparameter grids
        param_grids = self.get_hyperparameter_grids()

        # Evaluate both E and E+3 variants
        for variant in ["E", "E+3"]:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"EVALUATING VARIANT: {variant}")
                logger.info(f"{'='*60}")

                # Prepare features
                X_emb_train, X_scal_train, emb_scaler, scal_scaler = (
                    self.prepare_features(train_df, variant, fit=True)
                )
                X_emb_dev, X_scal_dev, _, _ = self.prepare_features(
                    dev_df, variant, emb_scaler=emb_scaler, scal_scaler=scal_scaler
                )
                X_emb_test, X_scal_test, _, _ = self.prepare_features(
                    test_df, variant, emb_scaler=emb_scaler, scal_scaler=scal_scaler
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

                # Grid search over hyperparameters
                best_artifacts = None
                best_dev_mcc = -1.0
                best_params = None

                grid = param_grids[variant]
                param_combinations = self._get_param_combinations(grid)

                logger.info(
                    f"Testing {len(param_combinations)} hyperparameter combinations..."
                )

                for i, params in enumerate(param_combinations):
                    logger.info(
                        f"  Combination {i+1}/{len(param_combinations)}: {params}"
                    )

                    try:
                        # Train model
                        artifacts = self.train_mlp_model(
                            X_emb_train,
                            y_train,
                            court_ids_train,
                            X_emb_dev,
                            y_dev,
                            court_ids_dev,
                            X_scal_train,
                            X_scal_dev,
                            **params,
                        )

                        dev_mcc = artifacts["best_mcc_dev_suppressed"]
                        logger.info(f"    Dev MCC (suppressed): {dev_mcc:.4f}")

                        if dev_mcc > best_dev_mcc:
                            best_dev_mcc = dev_mcc
                            best_artifacts = artifacts
                            best_params = params
                            logger.info(f"    New best: {dev_mcc:.4f}")

                    except Exception as e:
                        logger.error(f"    Failed: {e}")
                        continue

                if best_artifacts is None:
                    logger.error(f"❌ {variant} evaluation failed - no successful runs")
                    self.results[variant] = {
                        "error": "No successful hyperparameter combinations"
                    }
                    continue

                logger.info(f"Best hyperparameters for {variant}: {best_params}")
                logger.info(f"Best dev MCC (suppressed): {best_dev_mcc:.4f}")

                # Dev evaluation with SUPPRESSED CV selection
                dev_results = self.evaluate_model_with_suppressed_cv(
                    best_artifacts,
                    X_emb_train,
                    y_train,
                    court_ids_train,
                    X_emb_dev,
                    y_dev,
                    court_ids_dev,
                    X_scal_train,
                    X_scal_dev,
                    variant,
                )

                # Final test evaluation
                test_results = self.evaluate_model_on_test(
                    best_artifacts,
                    X_emb_train,
                    y_train,
                    court_ids_train,
                    X_emb_test,
                    y_test,
                    court_ids_test,
                    X_scal_train,
                    X_scal_test,
                    variant,
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
                self.results[variant] = {
                    **dev_results,
                    **test_results,
                    "feature_config": variant,
                    "n_features": X_emb_train.shape[1]
                    + (X_scal_train.shape[1] if X_scal_train is not None else 0),
                    "model_type": "PyTorchMLP",
                    # Selection criteria
                    "passes_delta_check": passes_delta_check,
                    "passes_ece_check": passes_ece_check,
                    "meets_selection_criteria": passes_delta_check and passes_ece_check,
                }

                logger.info(f"✅ {variant} evaluation completed")
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
                logger.error(f"❌ {variant} evaluation failed: {e}")
                self.results[variant] = {"error": str(e)}

        # Generate summary and save results with orjson optimization
        self.save_results_optimized()
        logger.info("=" * 80)
        logger.info("PYTORCH MLP TEMPORAL EVALUATION COMPLETED")
        logger.info("=" * 80)

    def _get_param_combinations(self, grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        import itertools

        keys = list(grid.keys())
        values = list(grid.values())

        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))

        return combinations

    def save_results_optimized(self):
        """Save comprehensive results with orjson optimization (reusing existing logic)."""
        logger.info("Saving results with orjson optimization...")

        # Save raw results with orjson
        results_file = self.output_dir / "pytorch_mlp_results.json"
        if ORJSON_AVAILABLE:
            with open(results_file, "wb") as f:
                f.write(_dumps(self.results))
        else:
            with open(results_file, "w") as f:
                _json.dump(self.results, f, indent=2, default=str)

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
            summary_file = self.output_dir / "pytorch_mlp_comparison.csv"
            summary_df.to_csv(summary_file, index=False)

            # Sort by dev suppressed MCC (PRIMARY SELECTION METRIC)
            summary_df_sorted = summary_df.sort_values(
                "dev_mcc_suppressed", ascending=False
            )

            logger.info("\n" + "=" * 100)
            logger.info(
                "PYTORCH MLP MODEL PERFORMANCE SUMMARY (SORTED BY SUPPRESSED MCC)"
            )
            logger.info("=" * 100)
            logger.info(summary_df_sorted.to_string(index=False, float_format="%.4f"))

            # Save test predictions with orjson
            if self.test_predictions:
                predictions_file = self.output_dir / "pytorch_mlp_test_predictions.json"
                if ORJSON_AVAILABLE:
                    with open(predictions_file, "wb") as f:
                        f.write(_dumps(self.test_predictions))
                else:
                    with open(predictions_file, "w") as f:
                        _json.dump(self.test_predictions, f, indent=2, default=str)

            logger.info(f"Results saved to {self.output_dir} with orjson optimization")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch MLP training with frozen LegalBERT embeddings"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to k-fold data directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument("--fold", type=int, default=4, help="Fold number to use")
    parser.add_argument(
        "--grid-search", action="store_true", help="Enable hyperparameter grid search"
    )
    parser.add_argument("--device", type=str, help="PyTorch device (cuda/cpu)")
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable mixed precision on CUDA"
    )

    args = parser.parse_args()

    # Create evaluator and run with PyTorch optimizations
    evaluator = PyTorchMLPEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold=args.fold,
        device=args.device,
        grid_search=args.grid_search,
        amp=(not args.no_amp),
    )

    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
