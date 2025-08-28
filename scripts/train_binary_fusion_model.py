#!/usr/bin/env python3
"""
Train Binary Fusion Model with Cross-Validation
Adapts CORAL ordinal architecture for binary classification
Uses cross-modal fusion (Legal-BERT + GraphSAGE) with pure MLP head
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Fast JSON loading with orjson optimization
try:
    import orjson as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data)

    def _loads_str(data: str) -> Any:
        # orjson only accepts bytes; encode
        return _json.loads(data.encode("utf-8"))

except ImportError:
    import json as _json  # type: ignore

    def _loads_bytes(data: bytes) -> Any:  # type: ignore
        return _json.loads(data.decode("utf-8"))

    def _loads_str(data: str) -> Any:  # type: ignore
        return _json.loads(data)


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corp_speech_risk_dataset.coral_ordinal.binary_fusion_model import (
    BinaryFusionMLP,
    get_binary_fusion_model,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BinaryFusionDataset(Dataset):
    """Dataset for binary classification with fused embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
        doc_ids: Optional[List[str]] = None,
    ):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(labels))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "doc_id": self.doc_ids[idx],
        }
        if self.weights is not None:
            sample["weight"] = self.weights[idx]
        return sample


def load_fold_data(
    fold_path: Path,
) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    """Load data from a fold directory with optimized JSON parsing."""
    logger.info(f"Loading data from {fold_path}")

    # Read JSONL files with orjson optimization
    data_rows = []
    for jsonl_file in sorted(fold_path.glob("*.jsonl")):
        with open(jsonl_file, "rb") as f:
            for line_bytes in f:
                line_bytes = line_bytes.strip()
                if line_bytes:
                    try:
                        data_rows.append(_loads_bytes(line_bytes))
                    except Exception as e:
                        # Fallback to string parsing for problematic lines
                        try:
                            data_rows.append(
                                _loads_str(line_bytes.decode("utf-8", errors="ignore"))
                            )
                        except Exception:
                            logger.warning(f"Failed to parse line in {jsonl_file}: {e}")
                            continue

    logger.info(f"Loaded {len(data_rows)} rows from {fold_path.name}")

    # Extract embeddings, labels, and weights
    embeddings = []
    labels = []
    weights = []
    doc_ids = []

    for row in data_rows:
        # Use fused embeddings if available, otherwise fallback
        if "fused_emb" in row and row["fused_emb"] is not None:
            embeddings.append(row["fused_emb"])
        elif "legal_bert_emb" in row and row["legal_bert_emb"] is not None:
            embeddings.append(row["legal_bert_emb"])
        elif "st_emb" in row and row["st_emb"] is not None:
            embeddings.append(row["st_emb"])
        else:
            logger.warning(f"No embedding found for {row.get('doc_id', 'unknown')}")
            continue

        # Binary label
        labels.append(row["outcome_bin"])

        # Support weight
        weights.append(row.get("support_weight", 1.0))

        # Document ID
        doc_ids.append(row.get("doc_id", f"doc_{len(doc_ids)}"))

    return (data_rows, np.array(embeddings), np.array(labels), np.array(weights))


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_weights = []

    for batch in tqdm(dataloader, desc="Training"):
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        weights = batch.get("weight", torch.ones_like(labels)).to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(embeddings)

        # Binary cross-entropy with class weights
        if class_weights is not None:
            # Apply class weights
            batch_class_weights = class_weights[labels]
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), reduction="none"
            )
            loss = (loss * weights * batch_class_weights).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), weight=weights
            )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Store predictions
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_weights.extend(weights.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)

    # Weighted metrics
    auc = roc_auc_score(all_labels, all_preds, sample_weight=all_weights)
    avg_precision = average_precision_score(
        all_labels, all_preds, sample_weight=all_weights
    )

    # Binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds, sample_weight=all_weights)

    return {
        "loss": total_loss / len(dataloader),
        "auc": auc,
        "avg_precision": avg_precision,
        "f1": f1,
    }


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_weights = []
    all_doc_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch["embedding"].to(device)
            labels = batch["label"]
            weights = batch.get("weight", torch.ones_like(labels))

            # Forward pass
            logits = model(embeddings)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_weights.extend(weights.numpy())
            all_doc_ids.extend(batch["doc_id"])

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds, sample_weight=all_weights)
    avg_precision = average_precision_score(
        all_labels, all_preds, sample_weight=all_weights
    )

    # Binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds, sample_weight=all_weights)

    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_preds, sample_weight=all_weights)

    return {
        "auc": auc,
        "avg_precision": avg_precision,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "weights": all_weights,
        "doc_ids": all_doc_ids,
    }


def train_fold(
    fold_idx: int,
    train_path: Path,
    val_path: Path,
    config: Dict,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    """Train on one fold."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Fold {fold_idx}")
    logger.info(f"{'='*50}")

    # Load data
    _, train_emb, train_labels, train_weights = load_fold_data(train_path)
    _, val_emb, val_labels, val_weights = load_fold_data(val_path)

    # Create datasets
    train_dataset = BinaryFusionDataset(train_emb, train_labels, train_weights)
    val_dataset = BinaryFusionDataset(val_emb, val_labels, val_weights)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # M1 Mac optimization
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # Initialize model
    model = get_binary_fusion_model(
        embed_dim=config.get("embed_dim", 768),
        hidden_dims=config.get("hidden_dims", (768, 512, 256)),
        head_hidden=config.get("head_hidden", 128),
        dropout=config.get("dropout", 0.1),
        head_dropout=config.get("head_dropout", 0.3),
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # Class weights (from fold metadata)
    if config.get("use_class_weights", True):
        # Calculate class weights from training data
        unique, counts = np.unique(train_labels, return_counts=True)
        class_weights = len(train_labels) / (len(unique) * counts)
        class_weights = torch.FloatTensor(class_weights).to(device)
        logger.info(f"Class weights: {class_weights}")
    else:
        class_weights = None

    # Training loop
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(config["epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        history["train"].append(train_metrics)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        history["val"].append(val_metrics)

        # Log metrics
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"AUC: {train_metrics['auc']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}"
        )
        logger.info(
            f"Val - AUC: {val_metrics['auc']:.4f}, "
            f"AP: {val_metrics['avg_precision']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}"
        )

        # Update scheduler
        scheduler.step(val_metrics["auc"])

        # Early stopping
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            best_model_path = output_dir / f"fold_{fold_idx}_best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "config": config,
                },
                best_model_path,
            )
            logger.info(f"Saved best model with AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 5):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation
    final_metrics = evaluate(model, val_loader, device)

    # Save results
    results = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "final_metrics": {
            k: v
            for k, v in final_metrics.items()
            if k not in ["predictions", "labels", "weights", "doc_ids"]
        },
        "history": history,
        "config": config,
    }

    # Save fold results
    with open(output_dir / f"fold_{fold_idx}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot training curves
    plot_training_curves(history, fold_idx, output_dir)

    # Plot confusion matrix
    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        ["Lower Risk", "Higher Risk"],
        fold_idx,
        output_dir,
    )

    return final_metrics


def plot_training_curves(history: Dict, fold_idx: int, output_dir: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    epochs = range(1, len(history["train"]) + 1)
    axes[0].plot(
        epochs, [h["loss"] for h in history["train"]], "b-", label="Train Loss"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fold {fold_idx} - Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC curves
    axes[1].plot(epochs, [h["auc"] for h in history["train"]], "b-", label="Train AUC")
    axes[1].plot(epochs, [h["auc"] for h in history["val"]], "r-", label="Val AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title(f"Fold {fold_idx} - AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"fold_{fold_idx}_training_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], fold_idx: int, output_dir: Path
):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(f"Fold {fold_idx} - Confusion Matrix (Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Add counts as text
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j + 0.5,
                i + 0.7,
                f"n={cm[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )

    plt.tight_layout()
    plt.savefig(output_dir / f"fold_{fold_idx}_confusion_matrix.png", dpi=150)
    plt.close()


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Binary Fusion Model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to binary k-fold data directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Base dropout rate")
    parser.add_argument(
        "--head-dropout", type=float, default=0.3, help="Head dropout rate"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu/cuda/mps/auto)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific folds to run (default: all)",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training configuration
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "head_dropout": args.head_dropout,
        "patience": args.patience,
        "embed_dim": 768,  # Legal-BERT dimension
        "hidden_dims": (768, 512, 256),
        "head_hidden": 128,
        "weight_decay": 1e-4,
        "use_class_weights": True,
    }

    # Save configuration
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load fold metadata
    with open(args.data_dir / "fold_statistics.json", "r") as f:
        fold_stats = json.load(f)

    # Determine folds to run
    if args.folds is None:
        # Run all CV folds (not including fold_4 which is final training)
        num_folds = fold_stats["folds"]
        folds_to_run = list(range(num_folds))
    else:
        folds_to_run = args.folds

    logger.info(f"Running folds: {folds_to_run}")

    # Run cross-validation
    cv_results = []

    for fold_idx in folds_to_run:
        train_path = args.data_dir / f"fold_{fold_idx}" / "train"
        val_path = args.data_dir / f"fold_{fold_idx}" / "dev"

        if not train_path.exists() or not val_path.exists():
            logger.warning(f"Skipping fold {fold_idx} - paths don't exist")
            continue

        fold_results = train_fold(
            fold_idx, train_path, val_path, config, device, args.output_dir
        )

        cv_results.append(
            {
                "fold": fold_idx,
                "auc": fold_results["auc"],
                "avg_precision": fold_results["avg_precision"],
                "f1": fold_results["f1"],
            }
        )

    # Calculate CV statistics
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        cv_summary = {
            "num_folds": len(cv_results),
            "mean_auc": cv_df["auc"].mean(),
            "std_auc": cv_df["auc"].std(),
            "mean_avg_precision": cv_df["avg_precision"].mean(),
            "std_avg_precision": cv_df["avg_precision"].std(),
            "mean_f1": cv_df["f1"].mean(),
            "std_f1": cv_df["f1"].std(),
            "per_fold_results": cv_results,
        }

        # Save CV summary
        with open(args.output_dir / "cv_summary.json", "w") as f:
            json.dump(cv_summary, f, indent=2)

        logger.info("\n" + "=" * 50)
        logger.info("Cross-Validation Summary")
        logger.info("=" * 50)
        logger.info(
            f"Mean AUC: {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}"
        )
        logger.info(
            f"Mean AP: {cv_summary['mean_avg_precision']:.4f} ± {cv_summary['std_avg_precision']:.4f}"
        )
        logger.info(
            f"Mean F1: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
