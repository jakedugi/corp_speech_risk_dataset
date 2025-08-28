#!/usr/bin/env python3
"""
End-to-End Binary Fusion Model Training
Trains Legal-BERT, GraphSAGE, CrossModalFusion, and Binary MLP per fold
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

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
import networkx as nx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corp_speech_risk_dataset.coral_ordinal.binary_fusion_model import (
    BinaryFusionMLP,
    get_binary_fusion_model,
)
from src.corp_speech_risk_dataset.encoding.legal_bert_embedder import (
    LegalBertEmbedder,
    get_legal_bert_embedder,
)
from src.corp_speech_risk_dataset.encoding.graphembedder import (
    get_graphsage_embedder,
    train_graphsage_model,
    CrossModalFusion,
    train_crossmodal_fusion,
    compute_graph_embedding,
    parse_deps_to_nx,
    GRAPH_INPUT_DIM,
)
from src.corp_speech_risk_dataset.encoding.tokenizer import decode_sp_ids
from torch_geometric.data import Data, Batch
import torch_geometric


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EndToEndSample:
    """Sample containing all data needed for end-to-end training."""

    doc_id: str
    text: str
    deps: List[List[Any]]
    raw_features: Optional[Dict[str, float]]
    label: int
    weight: float

    @classmethod
    def from_json(cls, row: Dict) -> "EndToEndSample":
        """Create sample from JSON row."""
        # Decode text if needed
        if "text" in row and row["text"]:
            text = row["text"]
        elif "sp_ids" in row:
            text = decode_sp_ids(row["sp_ids"])
        else:
            raise ValueError(f"No text found in row {row.get('doc_id', 'unknown')}")

        return cls(
            doc_id=row["doc_id"],
            text=text,
            deps=row.get("deps", []),
            raw_features=row.get("raw_features"),
            label=row["outcome_bin"],
            weight=row.get("support_weight", 1.0),
        )


class EndToEndDataset(Dataset):
    """Dataset for end-to-end training with raw text and graphs."""

    def __init__(self, samples: List[EndToEndSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_end2end(samples: List[EndToEndSample]) -> Dict:
    """Custom collate function for end-to-end samples."""
    texts = [s.text for s in samples]
    labels = torch.LongTensor([s.label for s in samples])
    weights = torch.FloatTensor([s.weight for s in samples])
    doc_ids = [s.doc_id for s in samples]
    deps_list = [s.deps for s in samples]
    raw_features_list = [s.raw_features for s in samples]

    return {
        "texts": texts,
        "labels": labels,
        "weights": weights,
        "doc_ids": doc_ids,
        "deps_list": deps_list,
        "raw_features_list": raw_features_list,
    }


class EndToEndModel(nn.Module):
    """Complete end-to-end model with all components."""

    def __init__(
        self,
        legal_bert_model: str = "nlpaueb/legal-bert-base-uncased",
        graph_hidden_dim: int = 256,
        fusion_dropout: float = 0.1,
        mlp_hidden_dims: tuple = (768, 512, 256),
        mlp_head_hidden: int = 128,
        mlp_dropout: float = 0.1,
        mlp_head_dropout: float = 0.3,
        device: torch.device = torch.device("cpu"),
        use_amp: bool = True,
    ):
        super().__init__()
        self.device = device
        self.use_amp = use_amp

        # Legal-BERT embedder
        logger.info("Initializing Legal-BERT embedder...")
        self.legal_bert = get_legal_bert_embedder(
            model_name=legal_bert_model, device=device, use_amp=use_amp
        )

        # GraphSAGE embedder
        logger.info("Initializing GraphSAGE embedder...")
        self.graphsage = get_graphsage_embedder(
            in_channels=GRAPH_INPUT_DIM, hidden_channels=graph_hidden_dim, num_layers=3
        ).to(device)

        # Cross-modal fusion
        logger.info("Initializing CrossModalFusion...")
        self.fusion = CrossModalFusion(
            text_dim=768,  # Legal-BERT dimension
            graph_dim=graph_hidden_dim,
            fusion_dim=768,
            dropout=fusion_dropout,
            num_heads=8,
        ).to(device)

        # Binary MLP classifier
        logger.info("Initializing Binary MLP classifier...")
        self.classifier = get_binary_fusion_model(
            embed_dim=768,
            hidden_dims=mlp_hidden_dims,
            head_hidden=mlp_head_hidden,
            dropout=mlp_dropout,
            head_dropout=mlp_head_dropout,
        ).to(device)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts using Legal-BERT."""
        embeddings = self.legal_bert.encode(
            texts, batch_size=batch_size, convert_to_numpy=False, show_progress=False
        )
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings)
        return embeddings.to(self.device)

    def encode_graphs(
        self, deps_list: List[List[List]], raw_features_list: List[Optional[Dict]]
    ) -> torch.Tensor:
        """Encode dependency graphs using GraphSAGE."""
        graphs = []

        for deps, raw_features in zip(deps_list, raw_features_list):
            # Parse dependencies to NetworkX graph
            nx_graph = parse_deps_to_nx(deps)

            # Convert to PyTorch Geometric format
            num_nodes = nx_graph.number_of_nodes()

            # Extract features
            if raw_features and "wl_features" in raw_features:
                wl_feats = raw_features["wl_features"]
                # Ensure correct dimension
                if len(wl_feats) < GRAPH_INPUT_DIM:
                    wl_feats.extend([0.0] * (GRAPH_INPUT_DIM - len(wl_feats)))
                elif len(wl_feats) > GRAPH_INPUT_DIM:
                    wl_feats = wl_feats[:GRAPH_INPUT_DIM]
                x = torch.tensor([wl_feats] * num_nodes, dtype=torch.float32)
            else:
                # Use degree-based features as fallback
                degrees = dict(nx_graph.degree())
                x = torch.zeros((num_nodes, GRAPH_INPUT_DIM))
                for i in range(num_nodes):
                    x[i, 0] = degrees.get(i, 0)
                    if i < GRAPH_INPUT_DIM - 1:
                        x[i, i + 1] = 1.0

            # Edge indices
            edge_list = list(nx_graph.edges())
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            graphs.append(Data(x=x, edge_index=edge_index))

        # Batch graphs
        batch = Batch.from_data_list(graphs).to(self.device)

        # Encode with GraphSAGE
        self.graphsage.eval()
        with torch.no_grad():
            node_embeddings = self.graphsage(batch.x, batch.edge_index)

            # Mean pool per graph
            graph_embeddings = torch_geometric.nn.global_mean_pool(
                node_embeddings, batch.batch
            )

        return graph_embeddings

    def forward(
        self,
        texts: List[str],
        deps_list: List[List[List]],
        raw_features_list: List[Optional[Dict]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Full forward pass from raw inputs to predictions."""
        # Encode modalities
        text_embeddings = self.encode_texts(texts, batch_size)
        graph_embeddings = self.encode_graphs(deps_list, raw_features_list)

        # Fuse embeddings
        fused_embeddings = self.fusion(text_embeddings, graph_embeddings)

        # Classify
        logits = self.classifier(fused_embeddings)

        return logits


def pretrain_graphsage(
    model: nn.Module,
    train_samples: List[EndToEndSample],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> nn.Module:
    """Pretrain GraphSAGE on reconstruction task."""
    logger.info("Pretraining GraphSAGE...")

    # Prepare graphs
    graphs = []
    for sample in tqdm(
        train_samples[:5000], desc="Preparing graphs"
    ):  # Limit for speed
        nx_graph = parse_deps_to_nx(sample.deps)
        num_nodes = nx_graph.number_of_nodes()

        if sample.raw_features and "wl_features" in sample.raw_features:
            wl_feats = sample.raw_features["wl_features"]
            if len(wl_feats) < GRAPH_INPUT_DIM:
                wl_feats.extend([0.0] * (GRAPH_INPUT_DIM - len(wl_feats)))
            elif len(wl_feats) > GRAPH_INPUT_DIM:
                wl_feats = wl_feats[:GRAPH_INPUT_DIM]
            x = torch.tensor([wl_feats] * num_nodes, dtype=torch.float32)
        else:
            degrees = dict(nx_graph.degree())
            x = torch.zeros((num_nodes, GRAPH_INPUT_DIM))
            for i in range(num_nodes):
                x[i, 0] = degrees.get(i, 0)
                if i < GRAPH_INPUT_DIM - 1:
                    x[i, i + 1] = 1.0

        edge_list = list(nx_graph.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        graphs.append(Data(x=x, edge_index=edge_index))

    # Train
    model = train_graphsage_model(
        model,
        graphs,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=5,
        loss_type="hybrid",
        dgi_weight=0.1,
    )

    return model


def pretrain_fusion(
    fusion_model: CrossModalFusion,
    legal_bert: LegalBertEmbedder,
    graphsage: nn.Module,
    train_samples: List[EndToEndSample],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 128,
) -> CrossModalFusion:
    """Pretrain fusion module with contrastive learning."""
    logger.info("Pretraining CrossModal Fusion...")

    # Prepare embeddings
    texts = [s.text for s in train_samples[:3000]]  # Limit for speed

    # Get text embeddings
    text_embeddings = []
    for i in range(0, len(texts), 32):
        batch_texts = texts[i : i + 32]
        batch_emb = legal_bert.encode(
            batch_texts, convert_to_numpy=False, show_progress=False
        )
        if isinstance(batch_emb, np.ndarray):
            batch_emb = torch.tensor(batch_emb)
        text_embeddings.append(batch_emb)
    text_embeddings = torch.cat(text_embeddings)

    # Get graph embeddings
    graph_embeddings = []
    for sample in tqdm(train_samples[:3000], desc="Computing graph embeddings"):
        nx_graph = parse_deps_to_nx(sample.deps)
        num_nodes = nx_graph.number_of_nodes()

        # Prepare features
        if sample.raw_features and "wl_features" in sample.raw_features:
            wl_feats = sample.raw_features["wl_features"]
            if len(wl_feats) < GRAPH_INPUT_DIM:
                wl_feats.extend([0.0] * (GRAPH_INPUT_DIM - len(wl_feats)))
            elif len(wl_feats) > GRAPH_INPUT_DIM:
                wl_feats = wl_feats[:GRAPH_INPUT_DIM]
            x = torch.tensor([wl_feats] * num_nodes, dtype=torch.float32)
        else:
            x = torch.zeros((num_nodes, GRAPH_INPUT_DIM))
            x[:, 0] = 1.0  # Simple feature

        edge_list = list(nx_graph.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Forward through GraphSAGE
        x = x.to(device)
        edge_index = edge_index.to(device)

        with torch.no_grad():
            node_emb = graphsage(x, edge_index)
            graph_emb = node_emb.mean(dim=0)  # Mean pool

        graph_embeddings.append(graph_emb)

    graph_embeddings = torch.stack(graph_embeddings)

    # Train fusion
    fusion_model = train_crossmodal_fusion(
        fusion_model,
        text_embeddings.split(256),  # Convert to list of tensors
        graph_embeddings.split(256),
        epochs=epochs,
        batch_size=batch_size,
        temperature=0.07,
        patience=2,
        use_amp=True,
    )

    return fusion_model


def train_epoch_end2end(
    model: EndToEndModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Train for one epoch end-to-end."""
    model.classifier.train()  # Only train classifier
    model.fusion.eval()  # Keep fusion frozen after pretraining
    model.graphsage.eval()  # Keep GraphSAGE frozen after pretraining

    total_loss = 0
    all_preds = []
    all_labels = []
    all_weights = []

    for batch in tqdm(dataloader, desc="Training"):
        texts = batch["texts"]
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        deps_list = batch["deps_list"]
        raw_features_list = batch["raw_features_list"]

        optimizer.zero_grad()

        # Forward pass
        logits = model(texts, deps_list, raw_features_list)

        # Binary cross-entropy with class weights
        if class_weights is not None:
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
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
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

    auc = roc_auc_score(all_labels, all_preds, sample_weight=all_weights)
    avg_precision = average_precision_score(
        all_labels, all_preds, sample_weight=all_weights
    )

    binary_preds = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds, sample_weight=all_weights)

    return {
        "loss": total_loss / len(dataloader),
        "auc": auc,
        "avg_precision": avg_precision,
        "f1": f1,
    }


def evaluate_end2end(
    model: EndToEndModel, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluate model end-to-end."""
    model.eval()
    all_preds = []
    all_labels = []
    all_weights = []
    all_doc_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = batch["texts"]
            labels = batch["labels"]
            weights = batch["weights"]
            doc_ids = batch["doc_ids"]
            deps_list = batch["deps_list"]
            raw_features_list = batch["raw_features_list"]

            # Forward pass
            logits = model(texts, deps_list, raw_features_list)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_weights.extend(weights.numpy())
            all_doc_ids.extend(doc_ids)

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds, sample_weight=all_weights)
    avg_precision = average_precision_score(
        all_labels, all_preds, sample_weight=all_weights
    )

    binary_preds = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds, sample_weight=all_weights)

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


def load_fold_data_end2end(fold_path: Path) -> List[EndToEndSample]:
    """Load data from a fold directory for end-to-end training."""
    logger.info(f"Loading data from {fold_path}")

    samples = []
    for jsonl_file in sorted(fold_path.glob("*.jsonl")):
        with open(jsonl_file, "r") as f:
            for line in f:
                row = json.loads(line)
                try:
                    sample = EndToEndSample.from_json(row)
                    samples.append(sample)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse row {row.get('doc_id', 'unknown')}: {e}"
                    )

    logger.info(f"Loaded {len(samples)} samples from {fold_path.name}")
    return samples


def train_fold_end2end(
    fold_idx: int,
    train_path: Path,
    val_path: Path,
    config: Dict,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    """Train on one fold with full end-to-end pipeline."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Fold {fold_idx} - End-to-End")
    logger.info(f"{'='*50}")

    # Load data
    train_samples = load_fold_data_end2end(train_path)
    val_samples = load_fold_data_end2end(val_path)

    # Create datasets
    train_dataset = EndToEndDataset(train_samples)
    val_dataset = EndToEndDataset(val_samples)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_end2end,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_end2end,
        num_workers=0,
    )

    # Initialize model
    model = EndToEndModel(
        legal_bert_model=config.get(
            "legal_bert_model", "nlpaueb/legal-bert-base-uncased"
        ),
        graph_hidden_dim=config.get("graph_hidden_dim", 256),
        fusion_dropout=config.get("fusion_dropout", 0.1),
        mlp_hidden_dims=config.get("mlp_hidden_dims", (768, 512, 256)),
        mlp_head_hidden=config.get("mlp_head_hidden", 128),
        mlp_dropout=config.get("mlp_dropout", 0.1),
        mlp_head_dropout=config.get("mlp_head_dropout", 0.3),
        device=device,
        use_amp=config.get("use_amp", True),
    )

    # Pretrain GraphSAGE
    if config.get("pretrain_graphsage", True):
        model.graphsage = pretrain_graphsage(
            model.graphsage,
            train_samples,
            device,
            epochs=config.get("graphsage_epochs", 10),
            batch_size=256,
            lr=3e-4,
        )

    # Pretrain Fusion
    if config.get("pretrain_fusion", True):
        model.fusion = pretrain_fusion(
            model.fusion,
            model.legal_bert,
            model.graphsage,
            train_samples,
            device,
            epochs=config.get("fusion_epochs", 5),
            batch_size=128,
        )

    # Initialize optimizer (only for classifier)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # Class weights
    if config.get("use_class_weights", True):
        train_labels = [s.label for s in train_samples]
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
        train_metrics = train_epoch_end2end(
            model, train_loader, optimizer, device, class_weights
        )
        history["train"].append(train_metrics)

        # Validate
        val_metrics = evaluate_end2end(model, val_loader, device)
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
    final_metrics = evaluate_end2end(model, val_loader, device)

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "doc_id": final_metrics["doc_ids"],
            "true_label": final_metrics["labels"],
            "predicted_prob": final_metrics["predictions"],
            "weight": final_metrics["weights"],
        }
    )
    predictions_df.to_csv(output_dir / f"fold_{fold_idx}_predictions.csv", index=False)

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

    with open(output_dir / f"fold_{fold_idx}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return final_metrics


def inference_on_test_set(
    model_path: Path,
    test_path: Path,
    config: Dict,
    device: torch.device,
    output_path: Path,
):
    """Run inference on test/oof set using trained model."""
    logger.info(f"\nRunning inference on {test_path}")

    # Load test data
    test_samples = load_fold_data_end2end(test_path)
    test_dataset = EndToEndDataset(test_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_end2end,
        num_workers=0,
    )

    # Initialize model
    model = EndToEndModel(
        legal_bert_model=config.get(
            "legal_bert_model", "nlpaueb/legal-bert-base-uncased"
        ),
        graph_hidden_dim=config.get("graph_hidden_dim", 256),
        fusion_dropout=config.get("fusion_dropout", 0.1),
        mlp_hidden_dims=config.get("mlp_hidden_dims", (768, 512, 256)),
        mlp_head_hidden=config.get("mlp_head_hidden", 128),
        mlp_dropout=config.get("mlp_dropout", 0.1),
        mlp_head_dropout=config.get("mlp_head_dropout", 0.3),
        device=device,
        use_amp=config.get("use_amp", True),
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run evaluation
    test_metrics = evaluate_end2end(model, test_loader, device)

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "doc_id": test_metrics["doc_ids"],
            "true_label": test_metrics["labels"],
            "predicted_prob": test_metrics["predictions"],
            "weight": test_metrics["weights"],
        }
    )
    predictions_df.to_csv(output_path, index=False)

    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Predictions saved to {output_path}")

    return test_metrics


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-End Binary Fusion Model Training"
    )
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
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (reduced for end-to-end)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
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
    parser.add_argument(
        "--pretrain-graphsage",
        action="store_true",
        default=True,
        help="Pretrain GraphSAGE",
    )
    parser.add_argument(
        "--pretrain-fusion",
        action="store_true",
        default=True,
        help="Pretrain fusion module",
    )
    parser.add_argument(
        "--run-oof-inference",
        action="store_true",
        help="Run inference on OOF test set using last fold",
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
        "patience": args.patience,
        "legal_bert_model": "nlpaueb/legal-bert-base-uncased",
        "graph_hidden_dim": 256,
        "fusion_dropout": 0.1,
        "mlp_hidden_dims": (768, 512, 256),
        "mlp_head_hidden": 128,
        "mlp_dropout": 0.1,
        "mlp_head_dropout": 0.3,
        "weight_decay": 1e-4,
        "use_class_weights": True,
        "use_amp": True,
        "pretrain_graphsage": args.pretrain_graphsage,
        "pretrain_fusion": args.pretrain_fusion,
        "graphsage_epochs": 10,
        "fusion_epochs": 5,
    }

    # Save configuration
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load fold metadata
    with open(args.data_dir / "fold_statistics.json", "r") as f:
        fold_stats = json.load(f)

    # Determine folds to run
    if args.folds is None:
        num_folds = fold_stats["folds"]
        folds_to_run = list(range(num_folds))
    else:
        folds_to_run = args.folds

    logger.info(f"Running folds: {folds_to_run}")

    # Run cross-validation
    cv_results = []
    last_model_path = None

    for fold_idx in folds_to_run:
        train_path = args.data_dir / f"fold_{fold_idx}" / "train"
        val_path = args.data_dir / f"fold_{fold_idx}" / "dev"

        if not train_path.exists() or not val_path.exists():
            logger.warning(f"Skipping fold {fold_idx} - paths don't exist")
            continue

        fold_results = train_fold_end2end(
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

        last_model_path = args.output_dir / f"fold_{fold_idx}_best_model.pt"

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

    # Run OOF inference if requested
    if args.run_oof_inference and last_model_path:
        oof_path = args.data_dir / "oof_test" / "test"
        if oof_path.exists():
            oof_metrics = inference_on_test_set(
                last_model_path,
                oof_path,
                config,
                device,
                args.output_dir / "oof_predictions.csv",
            )

            # Save OOF results
            with open(args.output_dir / "oof_results.json", "w") as f:
                json.dump(
                    {
                        "auc": oof_metrics["auc"],
                        "avg_precision": oof_metrics["avg_precision"],
                        "f1": oof_metrics["f1"],
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
