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
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import CORAL components
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from corp_speech_risk_dataset.coral_ordinal.config import Config
from corp_speech_risk_dataset.coral_ordinal.model import CORALMLP
from corp_speech_risk_dataset.coral_ordinal.losses import coral_loss
from corp_speech_risk_dataset.coral_ordinal.metrics import compute_metrics
from corp_speech_risk_dataset.coral_ordinal.utils import (
    set_seed,
    choose_device,
    save_checkpoint,
)


class CORALDataset(Dataset):
    """Dataset for CORAL ordinal regression with traceability."""

    def __init__(self, data: List[Dict[str, Any]], buckets: List[str]):
        self.data = data
        self.buckets = buckets
        self.label2idx = {b: i for i, b in enumerate(buckets)}

        # Store metadata for traceability
        self.metadata = []
        self.features = []
        self.labels = []

        for record in data:
            # Extract features and labels
            features = torch.tensor(record["fused_emb"], dtype=torch.float32)
            label = torch.tensor(self.label2idx[record["bucket"]], dtype=torch.long)

            self.features.append(features)
            self.labels.append(label)

            # Store metadata for traceability
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
) -> Tuple[CORALDataset, CORALDataset, CORALDataset]:
    """Create train/validation/test datasets."""
    dataset = CORALDataset(data, buckets)

    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    logger.info(
        f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}"
    )

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
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

            # Forward pass
            logits = model(features)
            loss = coral_loss(logits, labels, model.num_classes)
            total_loss += loss.item()

            # Get predictions
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
) -> torch.nn.Module:
    """Train the CORAL model with progress tracking."""
    device = choose_device(config.device)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Enable mixed precision for GPU, disable for MPS/CPU due to compatibility issues
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    set_seed(config.seed)
    best_val_score = -1

    logger.info(f"Training on device: {device}")
    logger.info(f"Mixed precision enabled: {use_amp}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, config.num_epochs + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits = model(features)
                loss = coral_loss(logits, labels, model.num_classes)

            # Use gradient scaling for mixed precision
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item() * features.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation phase
        val_metrics, _, _ = evaluate_model(
            model, val_loader, device, config.prob_threshold
        )

        # Track progress
        current_lr = optimizer.param_groups[0]["lr"]
        tracker.log_epoch(epoch, avg_train_loss, val_metrics, current_lr)

        # Save best model
        val_score = val_metrics["exact"].value
        if val_score > best_val_score:
            best_val_score = val_score
            save_checkpoint(
                model, tracker.output_dir / "best_model.pt", config, input_dim
            )
            logger.info(f"New best model saved (exact accuracy: {val_score:.3f})")

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
        default=[512, 128],
        help="Hidden layer dimensions",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda/mps/auto)")

    args = parser.parse_args()

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
    )

    # Save config
    config.save(output_dir / "config.json")

    # Load and split data
    data = load_data(args.data)
    train_dataset, val_dataset, test_dataset = create_datasets(
        data, args.buckets, args.val_split, args.test_split, args.seed
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Get input dimension from first sample
    sample_features, _ = train_dataset[0]
    input_dim = len(sample_features)
    logger.info(f"Input dimension: {input_dim}")

    # Create model
    model = CORALMLP(
        in_dim=input_dim,
        num_classes=len(args.buckets),
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )

    # Initialize training tracker
    tracker = TrainingTracker(output_dir)

    # Train model
    logger.info("Starting training...")
    trained_model = train_model(
        model, train_loader, val_loader, config, tracker, input_dim
    )

    # Save training history and plots
    tracker.save_history()
    tracker.plot_training_curves()

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    device = choose_device(config.device)

    # Load best model for testing
    best_model = CORALMLP(
        in_dim=input_dim,
        num_classes=len(args.buckets),
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred = evaluate_model(
        best_model, test_loader, device, args.threshold
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
