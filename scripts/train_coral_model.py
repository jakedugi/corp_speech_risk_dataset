#!/usr/bin/env python3
"""
Train CORAL ordinal model with comprehensive tracking and evaluation.

This script:
1. Loads prepared CORAL data with train/validation/test splits
2. Trains CORAL ordinal regression model with progress tracking
3. Evaluates model performance with multiple metrics
4. Saves model checkpoints and training history
5. Maintains traceability back to original embeddings

Usage:
    python scripts/train_coral_model.py \
        --data data/coral_training_data.jsonl \
        --output runs/coral_experiment \
        --epochs 50 \
        --batch-size 64 \
        --lr 3e-4 \
        --val-split 0.2 \
        --test-split 0.1 \
        --seed 42
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedShuffleSplit
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import CORAL components
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from corp_speech_risk_dataset.coral_ordinal.config import Config
from corp_speech_risk_dataset.coral_ordinal.model import CORALMLP, HybridOrdinalMLP
from corp_speech_risk_dataset.coral_ordinal.losses import (
    coral_loss,
    coral_threshold_prevalences,
    hybrid_loss,
)
from corp_speech_risk_dataset.coral_ordinal.metrics import compute_metrics
from corp_speech_risk_dataset.coral_ordinal.utils import (
    set_seed,
    choose_device,
    save_checkpoint,
)


class CORALDataset(Dataset):
    """Dataset for CORAL ordinal regression with traceability.

    Supports dynamic feature selection via concatenation of multiple keys and
    optional scalar features flattened from ``raw_features``.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        buckets: List[str],
        feature_keys: List[str] | None = None,
        include_scalars: bool = False,
    ):
        self.data = data
        self.buckets = buckets
        self.label2idx = {b: i for i, b in enumerate(buckets)}

        # Selection
        self.feature_keys: List[str] = feature_keys or ["fused_emb"]
        self.include_scalars: bool = include_scalars

        # Determine per-key dimensions by scanning once
        self.key_dims: Dict[str, int] = {}
        for key in self.feature_keys:
            dim = 0
            for rec in data:
                if key in rec and isinstance(rec[key], (list, tuple)):
                    dim = len(rec[key])
                    break
            self.key_dims[key] = dim

        # Determine scalar feature dimension from raw_features if requested
        self.scalar_dim: int = 0
        if self.include_scalars:
            for rec in data:
                raw = rec.get("raw_features")
                if isinstance(raw, dict):
                    self.scalar_dim = self._compute_scalar_vec(raw).shape[0]
                    break

        # Store metadata for traceability
        self.metadata: List[Dict[str, Any]] = []
        self.features: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        for record in data:
            # Build concatenated feature vector
            parts: List[torch.Tensor] = []
            for key in self.feature_keys:
                vals = record.get(key)
                if isinstance(vals, (list, tuple)):
                    parts.append(torch.tensor(vals, dtype=torch.float32))
                else:
                    # zero-pad if key missing using learned dim
                    pad_dim = self.key_dims.get(key, 0)
                    if pad_dim > 0:
                        parts.append(torch.zeros(pad_dim, dtype=torch.float32))
            if self.include_scalars:
                scal = self._compute_scalar_vec(record.get("raw_features"))
                if scal is not None:
                    parts.append(scal)
                elif self.scalar_dim > 0:
                    parts.append(torch.zeros(self.scalar_dim, dtype=torch.float32))

            if not parts:
                raise ValueError(
                    "No features assembled. Check feature_keys and input data fields."
                )

            features = torch.cat(parts, dim=0)
            label = torch.tensor(self.label2idx[record["bucket"]], dtype=torch.long)

            self.features.append(features)
            self.labels.append(label)

            # Store metadata
            metadata = {
                "doc_id": record.get("doc_id"),
                "text": record.get("text"),
                "speaker": record.get("speaker"),
                "final_judgement_real": record.get("final_judgement_real"),
                "bucket": record.get("bucket"),
                "_src": record.get("_src"),
            }
            self.metadata.append(metadata)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_metadata(self, idx):
        """Get metadata for traceability."""
        return self.metadata[idx]

    @staticmethod
    def _pad_or_list(val: Any, expected_len: int) -> List[float]:
        if val is None:
            return [0.0] * expected_len
        if isinstance(val, (list, tuple)):
            arr = list(val)[:expected_len]
            if len(arr) < expected_len:
                arr += [0.0] * (expected_len - len(arr))
            return [float(x) for x in arr]
        # scalar
        return [float(val)] + [0.0] * (expected_len - 1)

    def _compute_scalar_vec(self, raw: Dict[str, Any] | None) -> torch.Tensor | None:
        """Flatten selected raw feature groups into a fixed-order vector.

        Uses a robust layout aligned with upstream graph/global features:
        [quote_sentiment(3), context_sentiment(3), quote_deontic(1), context_deontic(1),
         quote_pos(11), context_pos(11), quote_ner(7), context_ner(7),
         quote_deps(23), context_deps(23), quote_wl(1), context_wl(1)] => 92 dims.
        """
        if not isinstance(raw, dict):
            return None

        # Infer lengths from present values, fallback to canonical sizes
        def infer_len(key: str, default_len: int) -> int:
            v = raw.get(key)
            if isinstance(v, (list, tuple)):
                return len(v)
            return default_len

        q_sent_len = infer_len("quote_sentiment", 3)
        c_sent_len = infer_len("context_sentiment", 3)
        q_pos_len = infer_len("quote_pos", 11)
        c_pos_len = infer_len("context_pos", 11)
        q_ner_len = infer_len("quote_ner", 7)
        c_ner_len = infer_len("context_ner", 7)
        q_dep_len = infer_len("quote_deps", 23)
        c_dep_len = infer_len("context_deps", 23)

        parts: List[float] = []
        parts += self._pad_or_list(raw.get("quote_sentiment"), q_sent_len)
        parts += self._pad_or_list(raw.get("context_sentiment"), c_sent_len)
        parts += self._pad_or_list(raw.get("quote_deontic_count"), 1)
        parts += self._pad_or_list(raw.get("context_deontic_count"), 1)
        parts += self._pad_or_list(raw.get("quote_pos"), q_pos_len)
        parts += self._pad_or_list(raw.get("context_pos"), c_pos_len)
        parts += self._pad_or_list(raw.get("quote_ner"), q_ner_len)
        parts += self._pad_or_list(raw.get("context_ner"), c_ner_len)
        parts += self._pad_or_list(raw.get("quote_deps"), q_dep_len)
        parts += self._pad_or_list(raw.get("context_deps"), c_dep_len)
        parts += self._pad_or_list(raw.get("quote_wl"), 1)
        parts += self._pad_or_list(raw.get("context_wl"), 1)

        return torch.tensor(parts, dtype=torch.float32)


class TrainingTracker:
    """Track training progress and metrics."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_exact": [],
            "val_off_by_one": [],
            "val_spearman": [],
            "epochs": [],
            "learning_rate": [],
        }
        self.best_metrics = {}
        self.start_time = time.time()

    def log_epoch(self, epoch: int, train_loss: float, val_metrics: Dict, lr: float):
        """Log metrics for one epoch."""
        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_metrics.get("loss", 0.0))
        self.history["val_exact"].append(val_metrics["exact"].value)
        self.history["val_off_by_one"].append(val_metrics["off_by_one"].value)
        self.history["val_spearman"].append(val_metrics["spearman_r"].value)
        self.history["learning_rate"].append(lr)

        # Update best metrics
        if (
            not self.best_metrics
            or val_metrics["exact"].value > self.best_metrics["exact"]
        ):
            self.best_metrics = {
                "epoch": epoch,
                "exact": val_metrics["exact"].value,
                "off_by_one": val_metrics["off_by_one"].value,
                "spearman": val_metrics["spearman_r"].value,
                "train_loss": train_loss,
            }

        # Print progress
        elapsed = time.time() - self.start_time
        logger.info(
            f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
            f"Exact: {val_metrics['exact'].value:.3f} | "
            f"Off1: {val_metrics['off_by_one'].value:.3f} | "
            f"Spearman: {val_metrics['spearman_r'].value:.3f} | "
            f"Time: {elapsed:.1f}s"
        )

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(
            self.history["epochs"], self.history["train_loss"], label="Train"
        )
        axes[0, 0].plot(
            self.history["epochs"], self.history["val_loss"], label="Validation"
        )
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Exact accuracy
        axes[0, 1].plot(self.history["epochs"], self.history["val_exact"])
        axes[0, 1].set_title("Exact Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].grid(True)

        # Off-by-one accuracy
        axes[1, 0].plot(self.history["epochs"], self.history["val_off_by_one"])
        axes[1, 0].set_title("Off-by-One Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].grid(True)

        # Spearman correlation
        axes[1, 1].plot(self.history["epochs"], self.history["val_spearman"])
        axes[1, 1].set_title("Spearman Correlation")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Correlation")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved training curves to {plot_path}")


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load prepared CORAL data."""
    data = []
    with open(data_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)

    logger.info(f"Loaded {len(data)} records from {data_path}")
    return data


def create_datasets(
    data: List[Dict[str, Any]],
    buckets: List[str],
    val_split: float,
    test_split: float,
    seed: int,
    feature_keys: List[str] | None = None,
    include_scalars: bool = False,
) -> Tuple[CORALDataset, CORALDataset, CORALDataset]:
    """Create train/validation/test datasets with stratified splits."""
    # Use stratified splits to preserve bucket distribution
    y = np.array([buckets.index(d["bucket"]) for d in data])

    # First split: train+val vs test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_val_idx, test_idx = next(sss.split(np.zeros_like(y), y))

    # Second split: train vs val
    y_tv = y[train_val_idx]
    val_split_adjusted = val_split / (1 - test_split)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split_adjusted, random_state=seed
    )
    train_idx_rel, val_idx_rel = next(sss2.split(np.zeros_like(y_tv), y_tv))

    # Map back to absolute indices
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    logger.info(
        f"Stratified splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    # Create datasets
    def to_dataset(idxs):
        return CORALDataset(
            [data[i] for i in idxs],
            buckets,
            feature_keys=feature_keys,
            include_scalars=include_scalars,
        )

    train_dataset = to_dataset(train_idx)
    val_dataset = to_dataset(val_idx)
    test_dataset = to_dataset(test_idx)

    return train_dataset, val_dataset, test_dataset


def choose_best_threshold(model, val_loader, device, config, metric="exact"):
    """Auto-tune threshold on validation set"""
    model.eval()
    best_t, best_score = 0.5, -1

    for t in torch.linspace(0.3, 0.7, 21):
        m, _, _ = evaluate_model(model, val_loader, device, config, threshold=float(t))
        score = m["exact"].value if metric == "exact" else m["spearman_r"].value
        if score > best_score:
            best_t, best_score = float(t), score

    logger.info(f"Best threshold: {best_t:.3f} with {metric}={best_score:.3f}")
    return best_t


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Config,
    threshold: float = 0.5,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics with predictions."""
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device), labels.to(device)

            # Forward pass and loss calculation
            if config.model_type == "hybrid":
                class_logits, reg_output = model(features)
                loss = hybrid_loss(
                    class_logits,
                    reg_output,
                    labels,
                    lambda_cls=config.lambda_cls,
                    lambda_reg=config.lambda_reg,
                )
                # For hybrid model, use classification logits for predictions
                logits = class_logits
            else:
                logits = model(features)
                loss = coral_loss(logits, labels, model.num_classes)

            total_loss += loss.item()

            # Get predictions
            if config.model_type == "hybrid":
                # For hybrid model, use direct classification predictions
                predictions = torch.argmax(logits, dim=1)
            else:
                # For CORAL model, use threshold-based predictions
                predictions = (torch.sigmoid(logits) > threshold).sum(1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics, y_true, y_pred


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, buckets: List[str], save_path: Path
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=buckets, yticklabels=buckets
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {save_path}")


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    tracker: TrainingTracker,
    input_dim: int,
    train_dataset=None,
) -> torch.nn.Module:
    """Train the CORAL model with progress tracking."""
    device = choose_device(config.device)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Choose scheduler based on config
    if config.use_onecycle_lr:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr * 3,  # 3x peak learning rate
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )
        step_per_batch = True
    else:
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                return (epoch + 1) / config.warmup_epochs
            else:
                # Cosine annealing after warmup
                import math

                return 0.5 * (
                    1
                    + math.cos(
                        math.pi
                        * (epoch - config.warmup_epochs)
                        / (config.num_epochs - config.warmup_epochs)
                    )
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        step_per_batch = False

    # Enable mixed precision for GPU, disable for MPS/CPU due to compatibility issues
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    set_seed(config.seed)
    best_val_score = -1
    patience_counter = 0

    # Compute threshold reweighting if enabled
    lambda_k = None
    if config.use_imbalance_weights:
        train_labels = [
            int(train_dataset[i][1].item()) for i in range(len(train_dataset))
        ]
        prev = coral_threshold_prevalences(train_labels, len(config.buckets))
        eps = 1e-3
        lambda_k = 1.0 / np.clip(prev * (1 - prev), eps, None)
        lambda_k = torch.tensor(lambda_k, dtype=torch.float32, device=device)
        logger.info(f"Threshold prevalences: {prev}")
        logger.info(f"Threshold weights: {lambda_k.cpu().numpy()}")

    # Initialize bias with class priors (only for CORALMLP which has .head)
    if config.use_imbalance_weights and lambda_k is not None and hasattr(model, "head"):
        with torch.no_grad():
            p = torch.clamp(torch.tensor(prev, dtype=torch.float32), 1e-4, 1 - 1e-4)
            model.head.bias.copy_(torch.log(p / (1 - p)))
            logger.info("Initialized bias with class priors")

    logger.info(f"Training on device: {device}")
    logger.info(f"Mixed precision enabled: {use_amp}")
    logger.info(f"OneCycle LR: {config.use_onecycle_lr}")
    logger.info(f"Label smoothing: {config.label_smoothing}")
    logger.info(f"Feature noise: {config.feature_noise}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, config.num_epochs + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # Add feature noise for regularization during training
            if model.training and config.feature_noise > 0:
                noise = config.feature_noise * torch.randn_like(features)
                features = features + noise

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                # Handle different model outputs
                if config.model_type == "hybrid":
                    class_logits, reg_output = model(features)
                    loss = hybrid_loss(
                        class_logits,
                        reg_output,
                        labels,
                        lambda_cls=config.lambda_cls,
                        lambda_reg=config.lambda_reg,
                        label_smoothing=config.label_smoothing,
                    )
                else:
                    logits = model(features)
                    loss = coral_loss(
                        logits,
                        labels,
                        model.num_classes,
                        lambda_k=lambda_k,
                        label_smoothing=config.label_smoothing,
                    )

            # Use gradient scaling for mixed precision
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_train_loss += loss.item() * features.size(0)

            # Step OneCycle scheduler per batch
            if step_per_batch:
                scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation phase
        val_metrics, _, _ = evaluate_model(
            model, val_loader, device, config, config.prob_threshold
        )

        # Track progress
        current_lr = optimizer.param_groups[0]["lr"]
        tracker.log_epoch(epoch, avg_train_loss, val_metrics, current_lr)

        # Save best model and early stopping
        val_score = val_metrics["exact"].value
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            save_checkpoint(
                model, tracker.output_dir / "best_model.pt", config, input_dim
            )
            logger.info(f"New best model saved (exact accuracy: {val_score:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs (patience: {config.patience})"
                )
                break

        # Step learning rate scheduler
        if step_per_batch:
            # OneCycle steps every batch - already done in training loop
            pass
        else:
            scheduler.step()

    # Save final model
    save_checkpoint(model, tracker.output_dir / "final_model.pt", config, input_dim)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train CORAL ordinal model")
    parser.add_argument("--data", required=True, help="Path to prepared CORAL data")
    parser.add_argument(
        "--output", default="runs/coral_experiment", help="Output directory"
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["low", "medium", "high"],
        help="Ordinal buckets",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[768, 512, 256],
        help="Hidden layer dimensions (wider & deeper with residual connections)",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="Learning rate warmup epochs"
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.03, help="Label smoothing epsilon"
    )
    parser.add_argument(
        "--feature-noise",
        type=float,
        default=0.01,
        help="Input noise for regularization",
    )
    parser.add_argument(
        "--no-imbalance-weights",
        action="store_true",
        help="Disable threshold reweighting",
    )
    parser.add_argument(
        "--no-onecycle-lr",
        action="store_true",
        help="Use cosine LR instead of OneCycle",
    )
    parser.add_argument(
        "--no-tune-threshold", action="store_true", help="Disable auto threshold tuning"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda/mps/auto)")

    # Hybrid model arguments
    parser.add_argument(
        "--model-type",
        choices=["coral", "hybrid"],
        default="coral",
        help="Model type to use",
    )
    parser.add_argument(
        "--lambda-cls",
        type=float,
        default=0.7,
        help="Classification loss weight for hybrid model",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.3,
        help="Regression loss weight for hybrid model",
    )

    # Feature selection arguments
    parser.add_argument(
        "--feature-keys",
        nargs="+",
        default=None,
        help=(
            "List of feature keys to concatenate (defaults to fused_emb). "
            "Examples: fused_emb legal_bert_emb gph_emb"
        ),
    )
    parser.add_argument(
        "--add-scalars",
        action="store_true",
        help="Append flattened raw_features scalars",
    )
    parser.add_argument(
        "--add-legal-bert", action="store_true", help="Append legal_bert_emb"
    )
    parser.add_argument("--add-graph", action="store_true", help="Append gph_emb")
    parser.add_argument(
        "--add-lb-aux",
        action="store_true",
        help=(
            "Append LB auxiliary embeddings: quote_top_keywords, context_top_keywords, "
            "speaker, section_headers"
        ),
    )
    parser.add_argument(
        "--replace-features",
        action="store_true",
        help="Replace default fused_emb instead of appending to it",
    )

    args = parser.parse_args()

    # Build selected feature keys list
    selected_feature_keys: List[str] = []
    # Default: fused only
    selected_feature_keys.append("fused_emb")

    # Use CLI feature-keys if provided
    if args.feature_keys:
        selected_feature_keys = [] if args.replace_features else selected_feature_keys
        for k in args.feature_keys:
            if k not in selected_feature_keys:
                selected_feature_keys.append(k)

    # Expand convenience toggles
    if args.add_legal_bert:
        if "legal_bert_emb" not in selected_feature_keys:
            selected_feature_keys.append("legal_bert_emb")
    if args.add_graph and "gph_emb" not in selected_feature_keys:
        selected_feature_keys.append("gph_emb")
    if args.add_lb_aux:
        for k in [
            "legal_bert_quote_top_keywords_emb",
            "legal_bert_context_top_keywords_emb",
            "legal_bert_speaker_emb",
            "legal_bert_section_headers_emb",
        ]:
            if k not in selected_feature_keys:
                selected_feature_keys.append(k)

    # Optional additive flags via environment variables (advanced users)
    # Allow env to augment or replace selections without breaking CLI usage
    import os

    env_keys = os.getenv("CORAL_FEATURE_KEYS")
    env_replace = os.getenv("CORAL_REPLACE_FEATURES", "0") == "1"
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        if keys:
            if env_replace:
                selected_feature_keys = []
            for k in keys:
                if k not in selected_feature_keys:
                    selected_feature_keys.append(k)

    add_scalars = bool(args.add_scalars or os.getenv("CORAL_ADD_SCALARS", "0") == "1")

    # If user explicitly wants to replace and didn't add any vector feature via CLI/env,
    # allow scalars-only by clearing the base fused key.
    if (
        args.replace_features
        and not args.feature_keys
        and not args.add_legal_bert
        and not args.add_graph
        and not args.add_lb_aux
        and not env_keys
    ):
        selected_feature_keys = []

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger.add(output_dir / "training.log", rotation="100 MB")

    logger.info("Starting CORAL model training")
    logger.info(f"Output directory: {output_dir}")

    # Create config
    config = Config(
        data_path=args.data,
        feature_key="fused_emb",
        feature_keys=selected_feature_keys,
        label_key="bucket",
        buckets=args.buckets,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        output_dir=str(output_dir),
        prob_threshold=args.threshold,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        use_imbalance_weights=not args.no_imbalance_weights,
        use_onecycle_lr=not args.no_onecycle_lr,
        feature_noise=args.feature_noise,
        tune_threshold=not args.no_tune_threshold,
        model_type=args.model_type,
        lambda_cls=args.lambda_cls,
        lambda_reg=args.lambda_reg,
        include_scalars=add_scalars,
    )

    # Save config
    config.save(output_dir / "config.json")

    # Load and split data
    data = load_data(args.data)
    train_dataset, val_dataset, test_dataset = create_datasets(
        data,
        args.buckets,
        args.val_split,
        args.test_split,
        args.seed,
        feature_keys=selected_feature_keys,
        include_scalars=add_scalars,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Get input dimension from first sample
    sample_features, _ = train_dataset[0]
    input_dim = len(sample_features)
    logger.info(f"Input dimension: {input_dim}")

    # Create model based on model type
    if args.model_type == "hybrid":
        model = HybridOrdinalMLP(
            in_dim=input_dim,
            num_classes=len(args.buckets),
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        )
        logger.info(
            f"Created hybrid model with λ_cls={args.lambda_cls}, λ_reg={args.lambda_reg}"
        )
    else:
        model = CORALMLP(
            in_dim=input_dim,
            num_classes=len(args.buckets),
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        )
        logger.info("Created CORAL model")

    # Initialize training tracker
    tracker = TrainingTracker(output_dir)

    # Train model
    logger.info("Starting training...")
    trained_model = train_model(
        model, train_loader, val_loader, config, tracker, input_dim, train_dataset
    )

    # val_loader will be used for threshold tuning below

    # Save training history and plots
    tracker.save_history()
    tracker.plot_training_curves()

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    device = choose_device(config.device)

    # Load best model for testing
    if args.model_type == "hybrid":
        best_model = HybridOrdinalMLP(
            in_dim=input_dim,
            num_classes=len(args.buckets),
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        )
    else:
        best_model = CORALMLP(
            in_dim=input_dim,
            num_classes=len(args.buckets),
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        )
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.to(device)  # Ensure model is on correct device

    # Auto-tune threshold on validation set if enabled
    test_threshold = args.threshold
    if config.tune_threshold:
        test_threshold = choose_best_threshold(best_model, val_loader, device, config)

    test_metrics, y_true, y_pred = evaluate_model(
        best_model, test_loader, device, config, test_threshold
    )

    # Log test results
    logger.info(f"Test Results:")
    logger.info(f"  Exact Accuracy: {test_metrics['exact'].value:.3f}")
    logger.info(f"  Off-by-One Accuracy: {test_metrics['off_by_one'].value:.3f}")
    logger.info(f"  Spearman Correlation: {test_metrics['spearman_r'].value:.3f}")

    # Save test results
    test_results = {
        "test_metrics": {
            k: float(v.value) if hasattr(v, "value") else float(v)
            for k, v in test_metrics.items()
        },
        "best_val_metrics": tracker.best_metrics,
        "config": vars(args),
    }

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, args.buckets, output_dir / "confusion_matrix.png"
    )

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=args.buckets)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    logger.success("Training completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
