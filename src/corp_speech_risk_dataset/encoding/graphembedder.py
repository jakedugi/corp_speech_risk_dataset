from __future__ import annotations
from functools import lru_cache
from typing import Literal, Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .parser import to_dependency_graph

from contextlib import nullcontext


def get_best_device() -> torch.device:
    """Get best available device with consistent priority: CUDA → CPU → MPS"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")  # Prefer CPU over MPS for stability


def _nx_to_pyg(g: nx.DiGraph) -> Data:
    # Convert NetworkX DiGraph to a PyG Data object
    # Nodes are indexed 0..N-1
    mapping = {n: i for i, n in enumerate(g.nodes())}
    if len(g.edges()) == 0:
        # Handle empty graphs
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            [[mapping[u] for u, v in g.edges()], [mapping[v] for u, v in g.edges()]],
            dtype=torch.long,
        )

    # Create fixed-size node features (degree + POS tag embeddings)
    node_features = []
    num_nodes = (
        g.number_of_nodes()
    )  # Fix: use number_of_nodes() instead of len(g.nodes())
    for node_id in range(num_nodes):
        # Get node attributes
        node_data = g.nodes[node_id] if node_id in g.nodes else {"pos": "X", "text": ""}

        # Simple degree feature - fix degree calculation
        degree = g.degree(node_id) if node_id in g.nodes else 0
        degree_val = degree if isinstance(degree, int) else 0  # Handle DiDegreeView

        # Simple POS tag embedding (map common POS tags to indices)
        pos_map = {
            "NOUN": 0,
            "VERB": 1,
            "ADJ": 2,
            "ADV": 3,
            "PRON": 4,
            "DET": 5,
            "ADP": 6,
            "CONJ": 7,
            "NUM": 8,
            "PART": 9,
            "PUNCT": 10,
            "X": 11,
        }
        pos_idx = pos_map.get(node_data.get("pos", "X"), 11)

        # Create fixed 16-dimensional feature vector
        # [degree, pos_one_hot(12), position_in_sentence(3)]
        feature = torch.zeros(16)
        feature[0] = min(degree_val / 10.0, 1.0)  # normalized degree
        feature[1 + pos_idx] = 1.0  # POS one-hot
        feature[14] = (
            min(node_id / num_nodes, 1.0) if num_nodes > 0 else 0
        )  # relative position
        feature[15] = 1.0  # bias term

        node_features.append(feature)

    if not node_features:
        # Empty graph fallback
        node_features = [torch.zeros(16)]

    x = torch.stack(node_features)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


@lru_cache(maxsize=None)
def get_node2vec_embedder(
    dim: int = 128, walk_length: int = 10, context_size: int = 5
) -> Node2Vec:
    """
    Returns a PyG Node2Vec model (transductive).
    You must call `model.train()` and fit or load pre-trained weights offline.
    """
    # Create a dummy graph to instantiate; real graph will be set at runtime
    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
    node2vec = Node2Vec(
        dummy_edge_index,
        embedding_dim=dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=1,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    )
    return node2vec


def train_graphsage_model(
    model: nn.Module,
    graphs: list[Data],
    epochs: int = 40,  # Increased from 15 to 40 for better convergence
    neighbor_samples: List[int] = [25, 10],  # GraphSAGE fan-out
    val_split: float = 0.2,
    batch_size: int = 512,  # Added batch processing
    eval_mode: bool = False,
    val_labels: Optional[List[int]] = None,
    loss_type: str = "hybrid",  # "mse", "cosine", or "hybrid"
    dgi_weight: float = 0.1,  # DGI auxiliary loss weight
    num_negative: int = 30,  # Increased negative sampling
) -> Tuple[nn.Module, Dict]:
    """
    Train GraphSAGE model with improved loss functions for 10^-4 MSE target.

    Args:
        model: GraphSAGE model to train
        graphs: List of PyG Data objects for training
        epochs: Number of training epochs (default 40)
        neighbor_samples: Fan-out for neighbor sampling [layer1, layer2]
        val_split: Fraction of data to use for validation
        batch_size: Batch size for training (default 512)
        eval_mode: Whether to run evaluation metrics
        val_labels: Optional labels for downstream classification eval
        loss_type: "mse", "cosine", or "hybrid" reconstruction loss
        dgi_weight: Weight for DGI contrastive loss (default 0.1)
        num_negative: Number of negative samples for DGI (default 30)

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    # Use consistent device priority: CUDA → MPS → CPU
    device = get_best_device()
    model = model.to(device)
    model.train()

    # Split into train/validation sets
    if len(graphs) < 10:
        print(
            f"[GRAPHSAGE WARNING] Only {len(graphs)} graphs available, skipping validation split"
        )
        train_graphs = graphs
        val_graphs = []
        train_labels = val_labels
        val_labels_split = None
    else:
        if val_labels is not None:
            train_graphs, val_graphs, train_labels, val_labels_split = train_test_split(
                graphs,
                val_labels,
                test_size=val_split,
                random_state=42,
                stratify=val_labels,
            )
        else:
            train_graphs, val_graphs = train_test_split(
                graphs, test_size=val_split, random_state=42
            )
            train_labels = val_labels_split = None

    print(
        f"[GRAPHSAGE TRAINING] Training on {len(train_graphs)} graphs, validating on {len(val_graphs)}"
    )
    print(f"[GRAPHSAGE TRAINING] Epochs: {epochs}, Batch size: {batch_size}")
    print(f"[GRAPHSAGE TRAINING] Neighbor sampling: {neighbor_samples}")
    print(f"[GRAPHSAGE TRAINING] Loss type: {loss_type.upper()}")
    print(
        f"[GRAPHSAGE TRAINING] DGI weight: {dgi_weight}, Negative samples: {num_negative}"
    )
    print(f"[GRAPHSAGE TRAINING] Device: {device}")
    print(f"[GRAPHSAGE TRAINING] Targeting 10^-4 MSE with enhanced objectives")

    # Optimized for extended training - lower LR, higher weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Persistent decoder for reconstruction loss - ensure on same device
    decoder = nn.Linear(256, 16).to(device)  # Updated for 256D hidden
    decoder_opt = torch.optim.AdamW(decoder.parameters(), lr=0.0005)

    # Move graphs to device - fix device assignment
    train_graphs = [g.to(device) for g in train_graphs if g.num_nodes > 0]
    val_graphs = [g.to(device) for g in val_graphs if g.num_nodes > 0]

    best_val_loss = float("inf")
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = 8  # Increased patience for longer training

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        total_samples = 0

        # Process in batches
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i : i + batch_size]
            batch_loss = None  # Initialize as None

            optimizer.zero_grad()
            decoder_opt.zero_grad()

            for pyg_data in batch_graphs:
                if pyg_data.num_nodes == 0 or pyg_data.x is None:
                    continue

                # Forward pass
                embeddings = model(pyg_data.x, pyg_data.edge_index)

                # Enhanced reconstruction loss based on type
                reconstructed = decoder(embeddings)

                if loss_type == "hybrid":
                    recon_loss = compute_hybrid_loss(reconstructed, pyg_data.x)
                elif loss_type == "cosine":
                    recon_loss = compute_scaled_cosine_loss(reconstructed, pyg_data.x)
                else:  # "mse"
                    recon_loss = F.mse_loss(reconstructed, pyg_data.x)

                # DGI contrastive loss for structural consistency
                if dgi_weight > 0:
                    dgi_loss = compute_dgi_loss(
                        embeddings, pyg_data.edge_index, num_negative
                    )
                    total_loss_batch = recon_loss + dgi_weight * dgi_loss
                else:
                    total_loss_batch = recon_loss

                # Accumulate batch loss
                if batch_loss is None:
                    batch_loss = total_loss_batch
                else:
                    batch_loss += total_loss_batch
                total_samples += 1

            if batch_loss is not None:
                batch_loss.backward()
                optimizer.step()
                decoder_opt.step()
                total_loss += batch_loss.item()

        scheduler.step()
        avg_train_loss = total_loss / max(total_samples, 1)
        train_losses.append(avg_train_loss)

        # Validation phase
        if val_graphs:
            model.eval()
            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for pyg_data in val_graphs:
                    if pyg_data.num_nodes == 0 or pyg_data.x is None:
                        continue

                    embeddings = model(pyg_data.x, pyg_data.edge_index)
                    reconstructed = decoder(embeddings)

                    # Use MSE for validation regardless of training loss
                    recon_loss = F.mse_loss(reconstructed, pyg_data.x)
                    val_loss += recon_loss.item()
                    val_samples += 1

            avg_val_loss = val_loss / max(val_samples, 1)
            val_losses.append(avg_val_loss)
        else:
            avg_val_loss = avg_train_loss  # Fallback if no validation set

        # Progress logging - show MSE progress toward 10^-4
        if epoch % 5 == 0 or epoch == epochs - 1:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[GRAPHSAGE TRAINING] Epoch {epoch:2d}: "
                f"train_loss = {avg_train_loss:.4f}, "
                f"val_MSE = {avg_val_loss:.2e}, "  # Scientific notation for MSE tracking
                f"lr = {lr:.6f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch > 20:  # Allow longer training
            print(f"[GRAPHSAGE TRAINING] Early stopping at epoch {epoch}")
            break

    model.eval()

    # Evaluation metrics - keep on same device
    eval_metrics = {}
    if eval_mode and val_graphs:
        print(f"[GRAPHSAGE EVALUATION] Running evaluation metrics...")
        eval_metrics = evaluate_graphsage_model(
            model, decoder, val_graphs, val_labels_split
        )

    # Move model back to CPU for inference to avoid device issues
    model = model.cpu()
    decoder = decoder.cpu()

    # Ensure epoch is defined for the final metrics
    final_epoch = epoch if "epoch" in locals() else epochs - 1

    eval_metrics.update(
        {
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "final_val_loss": val_losses[-1] if val_losses else 0.0,
            "best_val_loss": best_val_loss,
            "epochs_trained": final_epoch + 1,
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
            "loss_type": loss_type,
            "target_mse": "10^-4",
        }
    )

    # Check if we hit the 10^-4 target
    if best_val_loss < 1e-4:
        print(
            f"[GRAPHSAGE TRAINING] 🎯 TARGET ACHIEVED! Final val MSE: {best_val_loss:.2e} < 10^-4"
        )
    else:
        print(
            f"[GRAPHSAGE TRAINING] Final val MSE: {best_val_loss:.2e} (target: 10^-4)"
        )

    return model, eval_metrics


def evaluate_graphsage_model(
    model: nn.Module,
    decoder: nn.Module,
    val_graphs: List[Data],
    val_labels: Optional[List[int]] = None,
) -> Dict:
    """
    Comprehensive evaluation of GraphSAGE model.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    # Get device from model parameters
    device = next(model.parameters()).device

    metrics = {}

    # 1. Reconstruction Loss
    total_recon_loss = 0.0
    val_embeddings = []

    with torch.no_grad():
        for graph in val_graphs:
            if graph.num_nodes == 0 or graph.x is None:
                continue

            # Ensure graph is on same device as model
            graph = graph.to(device)
            embeddings = model(graph.x, graph.edge_index)

            # Store embeddings for downstream tasks
            val_embeddings.append(embeddings.mean(dim=0).cpu().numpy())

            # Reconstruction loss - ensure decoder is on same device
            if decoder is not None:
                # Ensure decoder is on same device as embeddings
                if next(decoder.parameters()).device != device:
                    decoder = decoder.to(device)
                reconstructed = decoder(embeddings)
                recon_loss = F.mse_loss(reconstructed, graph.x)
                total_recon_loss += recon_loss.item()

    metrics["reconstruction_loss"] = total_recon_loss / max(len(val_graphs), 1)

    # 2. Downstream Classification (if labels provided)
    if val_labels is not None and len(val_embeddings) == len(val_labels):
        try:
            embeddings_array = np.array(val_embeddings)
            labels_array = np.array(val_labels)

            # Train simple classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(embeddings_array, labels_array)

            # Predictions
            y_pred = clf.predict(embeddings_array)
            y_pred_proba = (
                clf.predict_proba(embeddings_array)[:, 1]
                if len(np.unique(labels_array)) == 2
                else None
            )

            # Metrics
            metrics["classification_f1"] = f1_score(
                labels_array, y_pred, average="weighted"
            )
            if y_pred_proba is not None:
                metrics["classification_auc"] = roc_auc_score(
                    labels_array, y_pred_proba
                )
            else:
                metrics["classification_auc"] = 0.0

            print(
                f"[GRAPHSAGE EVAL] Classification F1: {metrics['classification_f1']:.3f}"
            )
            print(
                f"[GRAPHSAGE EVAL] Classification AUC: {metrics['classification_auc']:.3f}"
            )

        except Exception as e:
            print(f"[GRAPHSAGE EVAL] Classification evaluation failed: {e}")
            metrics["classification_f1"] = 0.0
            metrics["classification_auc"] = 0.0

    # 3. Clustering Quality (if we have enough samples)
    if len(val_embeddings) >= 10:
        try:
            embeddings_array = np.array(val_embeddings)

            # K-means clustering
            n_clusters = min(5, len(val_embeddings) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(embeddings_array)

            # Silhouette score
            if len(np.unique(cluster_labels)) > 1:
                sil_score = silhouette_score(embeddings_array, cluster_labels)
                metrics["silhouette_score"] = sil_score
                print(f"[GRAPHSAGE EVAL] Silhouette Score: {sil_score:.3f}")
            else:
                metrics["silhouette_score"] = 0.0

        except Exception as e:
            print(f"[GRAPHSAGE EVAL] Clustering evaluation failed: {e}")
            metrics["silhouette_score"] = 0.0

    print(f"[GRAPHSAGE EVAL] Reconstruction Loss: {metrics['reconstruction_loss']:.4e}")

    return metrics


@lru_cache(maxsize=None)
def get_graphsage_embedder(
    in_channels: int = 16,
    hidden_channels: int = 256,
    num_layers: int = 3,  # Increased to 256D and 3 layers
) -> nn.Module:
    """
    Returns a GraphSAGE model optimized for legal text dependency graphs.
    Default: 16→256→256→256 (3 layers) for capturing multi-hop legal patterns.
    """

    class GraphSAGEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.dropout = nn.Dropout(
                0.3
            )  # Increased dropout for legal domain regularization

            if num_layers == 1:
                self.convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.convs.append(SAGEConv(in_channels, hidden_channels))
                for _ in range(num_layers - 2):
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if (
                    i < len(self.convs) - 1
                ):  # Apply ReLU and dropout to all but last layer
                    x = F.relu(x)
                    x = self.dropout(x)
            return x

    return GraphSAGEModel()


class CrossModalFusion(nn.Module):
    """
    Enhanced two-stream cross-attention fusion with stability improvements:
      - text_embs: [B, Tdim]
      - graph_embs: [B, Gdim]
    returns fused [B, Fdim] where Fdim = fusion_dim

    Enhancements for Legal-BERT integration:
    - LayerNorm for gradient stability on legal quotes
    - Dropout for regularization during long training runs
    - Adaptive dimensions for Legal-BERT (768D) compatibility
    """

    def __init__(
        self,
        text_dim: int,
        graph_dim: Optional[int] = None,
        fusion_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        # Set default dimensions - prioritize Legal-BERT compatibility
        graph_dim = graph_dim or text_dim
        fusion_dim = fusion_dim or max(
            text_dim, graph_dim
        )  # Use larger dimension as default (768 for Legal-BERT)

        self.fusion_dim = fusion_dim

        # Project text and graph embeddings into the same fusion space
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)

        # Stability improvements: dropout after projections
        self.text_dropout = nn.Dropout(dropout)
        self.graph_dropout = nn.Dropout(dropout)

        # Cross-attention with configurable heads
        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        # LayerNorm for gradient stability (critical for legal quote complexity)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Final projection with residual connection
        self.proj = nn.Linear(fusion_dim * 2, fusion_dim)
        self.final_norm = nn.LayerNorm(fusion_dim)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, txt: torch.Tensor, gph: torch.Tensor) -> torch.Tensor:
        # Project into shared fusion dimension with dropout
        txt_up = self.text_dropout(self.text_proj(txt))  # [B, fusion_dim]
        gph_up = self.graph_dropout(self.graph_proj(gph))  # [B, fusion_dim]

        # Add a sequence dimension of 1 for attention
        txt_seq = txt_up.unsqueeze(1)  # [B,1,fusion_dim]
        gph_seq = gph_up.unsqueeze(1)  # [B,1,fusion_dim]

        # Cross-attend: text attends to graph with normalization
        o1, _ = self.attn(txt_seq, gph_seq, gph_seq)
        o1_norm = self.norm1(o1.squeeze(1))  # [B, fusion_dim]

        # Graph attends to text with normalization
        o2, _ = self.attn(gph_seq, txt_seq, txt_seq)
        o2_norm = self.norm2(o2.squeeze(1))  # [B, fusion_dim]

        # Concatenate attended outputs
        fused = torch.cat([o1_norm, o2_norm], dim=-1)  # [B, 2*fusion_dim]

        # Final projection with normalization and dropout
        output = self.proj(fused)  # [B, fusion_dim]
        output = self.final_norm(output)
        output = self.final_dropout(output)

        return output

    def get_fusion_dim(self) -> int:
        """Return the fusion dimension for external compatibility checks."""
        return self.fusion_dim


def train_crossmodal_fusion(
    fusion_model: CrossModalFusion,
    text_embeddings: List[torch.Tensor],
    graph_embeddings: List[torch.Tensor],
    epochs: int = 12,  # Increased from 5 to 12
    batch_size: int = 256,  # Added batch processing
    temperature: float = 0.07,  # InfoNCE temperature
    patience: int = 3,  # Early stopping patience
    use_amp: bool = False,  # Automatic Mixed Precision
    adaptive_temperature: bool = True,  # New: Adaptive temperature for legal text
    hard_negative_weight: float = 1.2,  # New: Hard negative mining weight
) -> CrossModalFusion:
    """
    Train CrossModalFusion with InfoNCE contrastive learning - optimized for legal text.

    Args:
        fusion_model: CrossModalFusion model to train
        text_embeddings: List of text embedding tensors
        graph_embeddings: List of graph embedding tensors
        epochs: Number of training epochs (default 12)
        batch_size: Batch size for contrastive learning (default 256)
        temperature: Temperature parameter for InfoNCE (default 0.07)
        patience: Early stopping patience (default 3)
        use_amp: Automatic Mixed Precision for speedup (default False)
        adaptive_temperature: Use adaptive temperature in InfoNCE (default True)
        hard_negative_weight: Weight for hard negative mining (default 1.2)

    Returns:
        Trained fusion model
    """
    device = get_best_device()

    fusion_model = fusion_model.to(device)
    fusion_model.train()

    # Convert to tensors and move to device
    text_tensor = torch.stack(text_embeddings).to(device)
    graph_tensor = torch.stack(graph_embeddings).to(device)

    # Optimized for cross-modal learning
    optimizer = torch.optim.AdamW(
        fusion_model.parameters(), lr=0.001, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(
        f"[CROSSMODAL TRAINING] Training fusion on {len(text_embeddings)} pairs for {epochs} epochs"
    )
    print(f"[CROSSMODAL TRAINING] Batch size: {batch_size}, Temperature: {temperature}")
    print(f"[CROSSMODAL TRAINING] Device: {device}")
    print(
        f"[CROSSMODAL TRAINING] Using InfoNCE contrastive loss for legal text alignment"
    )

    best_loss = float("inf")
    patience_counter = 0

    # Enhanced tracking for alignment monitoring
    epoch_metrics = {
        "text_alignment": [],
        "graph_alignment": [],
        "adaptive_temperature": [],
    }

    # Enable AMP if requested for T4 efficiency
    if use_amp and device.type == "cuda":
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        print(f"[CROSSMODAL TRAINING] AMP enabled for CUDA speedup")
    elif use_amp and device.type == "mps":
        try:
            autocast_context = torch.autocast(device_type="mps", dtype=torch.float16)
            print(f"[CROSSMODAL TRAINING] AMP enabled for MPS speedup")
        except:
            autocast_context = nullcontext()
    else:
        autocast_context = nullcontext()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        epoch_text_alignment = []
        epoch_graph_alignment = []
        epoch_adaptive_temp = []

        # Process in batches for better gradients
        for i in range(0, len(text_embeddings), batch_size):
            end_idx = min(i + batch_size, len(text_embeddings))
            batch_text = text_tensor[i:end_idx]
            batch_graph = graph_tensor[i:end_idx]

            optimizer.zero_grad()

            # Use autocast or nullcontext for forward pass
            with autocast_context:
                # Forward pass
                fused = fusion_model(batch_text, batch_graph)

                # Enhanced InfoNCE contrastive loss - legal text optimized
                batch_loss, batch_metrics = compute_infonce_loss(
                    fused,
                    batch_text,
                    batch_graph,
                    temperature=temperature,
                    use_nt_xent=use_amp,
                    adaptive_temp=adaptive_temperature,
                    hard_negative_weight=hard_negative_weight,
                )

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

            # Collect alignment metrics for monitoring
            epoch_text_alignment.append(batch_metrics["text_alignment"])
            epoch_graph_alignment.append(batch_metrics["graph_alignment"])
            epoch_adaptive_temp.append(batch_metrics["adaptive_temperature"])

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Compute epoch-level alignment metrics
        avg_text_alignment = (
            sum(epoch_text_alignment) / len(epoch_text_alignment)
            if epoch_text_alignment
            else 0.0
        )
        avg_graph_alignment = (
            sum(epoch_graph_alignment) / len(epoch_graph_alignment)
            if epoch_graph_alignment
            else 0.0
        )
        avg_adaptive_temp = (
            sum(epoch_adaptive_temp) / len(epoch_adaptive_temp)
            if epoch_adaptive_temp
            else temperature
        )

        # Store epoch metrics
        epoch_metrics["text_alignment"].append(avg_text_alignment)
        epoch_metrics["graph_alignment"].append(avg_graph_alignment)
        epoch_metrics["adaptive_temperature"].append(avg_adaptive_temp)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Enhanced progress reporting with alignment metrics
        if epoch % 3 == 0 or epoch == epochs - 1:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[CROSSMODAL TRAINING] Epoch {epoch:2d}: loss = {avg_loss:.4f}, lr = {lr:.6f}"
            )
            print(
                f"                     Text align: {avg_text_alignment:.3f}, "
                f"Graph align: {avg_graph_alignment:.3f}, "
                f"Temp: {avg_adaptive_temp:.3f}"
            )

            # Alignment quality assessment (target >0.7 for good alignment)
            if avg_text_alignment > 0.7 and avg_graph_alignment > 0.7:
                print(f"                     ✓ Strong cross-modal alignment achieved")
            elif avg_text_alignment > 0.5 and avg_graph_alignment > 0.5:
                print(
                    f"                     ⚠ Moderate alignment - consider more epochs"
                )
            else:
                print(
                    f"                     ⚠ Weak alignment - may need hyperparameter tuning"
                )

        if patience_counter >= patience and epoch > 5:
            print(f"[CROSSMODAL TRAINING] Early stopping at epoch {epoch}")
            break

    fusion_model.eval()

    # Final evaluation on full dataset
    print(f"[CROSSMODAL TRAINING] Evaluating final alignment quality...")
    final_metrics = evaluate_crossmodal_alignment(
        fusion_model, text_tensor, graph_tensor, device
    )

    fusion_model = fusion_model.cpu()  # Move back for inference
    print(f"[CROSSMODAL TRAINING] Training complete! Best loss: {best_loss:.4f}")
    print(
        f"[CROSSMODAL TRAINING] Final alignment: Text={final_metrics['text_alignment']:.3f}, Graph={final_metrics['graph_alignment']:.3f}"
    )

    return fusion_model


def evaluate_crossmodal_alignment(
    fusion_model: CrossModalFusion,
    text_embeddings: torch.Tensor,
    graph_embeddings: torch.Tensor,
    device: torch.device,
    eval_batch_size: int = 512,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of cross-modal alignment quality.

    Computes alignment metrics for unsupervised assessment:
    - Positive pair similarities (text-fused, graph-fused alignment)
    - Negative pair discrimination capability
    - Cross-modal retrieval accuracy (text→graph, graph→text)
    - Alignment consistency across legal quote complexity

    Args:
        fusion_model: Trained CrossModalFusion model
        text_embeddings: Text embeddings tensor [N, text_dim]
        graph_embeddings: Graph embeddings tensor [N, graph_dim]
        device: Compute device
        eval_batch_size: Batch size for evaluation

    Returns:
        Dictionary of alignment metrics
    """
    fusion_model.eval()
    fusion_model = fusion_model.to(device)

    all_fused = []
    all_text_norm = []
    all_graph_norm = []

    # Process in batches to avoid memory issues
    with torch.no_grad():
        for i in range(0, len(text_embeddings), eval_batch_size):
            end_idx = min(i + eval_batch_size, len(text_embeddings))
            batch_text = text_embeddings[i:end_idx].to(device)
            batch_graph = graph_embeddings[i:end_idx].to(device)

            # Generate fused embeddings
            batch_fused = fusion_model(batch_text, batch_graph)

            # Normalize all embeddings for cosine similarity
            batch_fused_norm = F.normalize(batch_fused, dim=1, eps=1e-8)
            batch_text_norm = F.normalize(batch_text, dim=1, eps=1e-8)
            batch_graph_norm = F.normalize(batch_graph, dim=1, eps=1e-8)

            all_fused.append(batch_fused_norm.cpu())
            all_text_norm.append(batch_text_norm.cpu())
            all_graph_norm.append(batch_graph_norm.cpu())

    # Concatenate all batches
    fused_norm = torch.cat(all_fused, dim=0)
    text_norm = torch.cat(all_text_norm, dim=0)
    graph_norm = torch.cat(all_graph_norm, dim=0)

    n_samples = fused_norm.shape[0]

    # 1. Positive Pair Alignment (diagonal similarities)
    text_fused_sim = torch.matmul(text_norm, fused_norm.T)
    graph_fused_sim = torch.matmul(graph_norm, fused_norm.T)

    pos_text_similarities = torch.diagonal(text_fused_sim)
    pos_graph_similarities = torch.diagonal(graph_fused_sim)

    # 2. Negative Pair Discrimination (off-diagonal vs diagonal)
    text_neg_mask = ~torch.eye(n_samples, dtype=torch.bool)
    graph_neg_mask = ~torch.eye(n_samples, dtype=torch.bool)

    text_neg_similarities = text_fused_sim[text_neg_mask]
    graph_neg_similarities = graph_fused_sim[graph_neg_mask]

    # 3. Cross-Modal Retrieval Accuracy (top-k precision)
    def compute_retrieval_accuracy(
        similarity_matrix: torch.Tensor, k: int = 5
    ) -> float:
        """Compute top-k retrieval accuracy."""
        n = similarity_matrix.shape[0]
        correct = 0
        for i in range(n):
            # Get top-k most similar items for sample i
            _, top_indices = torch.topk(similarity_matrix[i], k=min(k, n))
            # Check if correct match (index i) is in top-k
            if i in top_indices:
                correct += 1
        return correct / n

    text_to_fused_acc = compute_retrieval_accuracy(text_fused_sim, k=5)
    graph_to_fused_acc = compute_retrieval_accuracy(graph_fused_sim, k=5)

    # 4. Alignment Consistency (standard deviation of positive similarities)
    text_alignment_consistency = (
        1.0 - pos_text_similarities.std().item()
    )  # Lower std = higher consistency
    graph_alignment_consistency = 1.0 - pos_graph_similarities.std().item()

    # 5. Discriminative Power (ratio of positive to negative similarities)
    text_discrimination = (
        pos_text_similarities.mean() - text_neg_similarities.mean()
    ).item()
    graph_discrimination = (
        pos_graph_similarities.mean() - graph_neg_similarities.mean()
    ).item()

    metrics = {
        "text_alignment": pos_text_similarities.mean().item(),
        "graph_alignment": pos_graph_similarities.mean().item(),
        "text_alignment_std": pos_text_similarities.std().item(),
        "graph_alignment_std": pos_graph_similarities.std().item(),
        "text_retrieval_acc_top5": text_to_fused_acc,
        "graph_retrieval_acc_top5": graph_to_fused_acc,
        "text_alignment_consistency": max(
            0.0, text_alignment_consistency
        ),  # Clamp to [0,1]
        "graph_alignment_consistency": max(0.0, graph_alignment_consistency),
        "text_discrimination_power": text_discrimination,
        "graph_discrimination_power": graph_discrimination,
        "overall_alignment": (
            pos_text_similarities.mean() + pos_graph_similarities.mean()
        ).item()
        / 2.0,
        "overall_retrieval_acc": (text_to_fused_acc + graph_to_fused_acc) / 2.0,
    }

    return metrics


def compute_infonce_loss(
    fused: torch.Tensor,
    text: torch.Tensor,
    graph: torch.Tensor,
    temperature: float = 0.07,
    use_nt_xent: bool = False,
    adaptive_temp: bool = True,
    hard_negative_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Enhanced InfoNCE contrastive loss for cross-modal alignment.
    Optimized for legal text dependency graph alignment with stability improvements.

    Args:
        fused: Fused embeddings [B, fusion_dim]
        text: Text embeddings [B, text_dim]
        graph: Graph embeddings [B, graph_dim]
        temperature: Base temperature for InfoNCE
        use_nt_xent: Use NT-Xent normalization for stability
        adaptive_temp: Dynamically adjust temperature based on alignment
        hard_negative_weight: Weight for hard negative mining

    Returns:
        Tuple of (loss, metrics_dict)
    """
    batch_size = fused.shape[0]
    fused_dim = fused.shape[1]
    device = fused.device

    # Project text and graph to fused dimension if needed (with proper initialization)
    if text.shape[1] != fused_dim:
        text_proj = nn.Linear(text.shape[1], fused_dim).to(device)
        # Xavier initialization for stability
        nn.init.xavier_uniform_(text_proj.weight)
        nn.init.zeros_(text_proj.bias)
        text = text_proj(text)
    if graph.shape[1] != fused_dim:
        graph_proj = nn.Linear(graph.shape[1], fused_dim).to(device)
        # Xavier initialization for stability
        nn.init.xavier_uniform_(graph_proj.weight)
        nn.init.zeros_(graph_proj.bias)
        graph = graph_proj(graph)

    # Normalize embeddings for cosine similarity
    fused_norm = F.normalize(fused, dim=1, eps=1e-8)
    text_norm = F.normalize(text, dim=1, eps=1e-8)
    graph_norm = F.normalize(graph, dim=1, eps=1e-8)

    # Adaptive temperature based on current alignment (for legal quote complexity)
    if adaptive_temp:
        # Compute mean positive similarity to adjust temperature
        pos_text_sim = torch.diagonal(torch.matmul(text_norm, fused_norm.T))
        pos_graph_sim = torch.diagonal(torch.matmul(graph_norm, fused_norm.T))
        mean_pos_sim = (pos_text_sim.mean() + pos_graph_sim.mean()) / 2

        # Adaptive temperature: lower temp when alignment is good (higher temp for hard negatives)
        adaptive_temperature = temperature * (2.0 - mean_pos_sim.clamp(0.1, 0.9))
    else:
        adaptive_temperature = temperature

    # Text-to-fused similarity matrix
    text_fused_sim = torch.matmul(text_norm, fused_norm.T) / adaptive_temperature

    # Graph-to-fused similarity matrix
    graph_fused_sim = torch.matmul(graph_norm, fused_norm.T) / adaptive_temperature

    # Hard negative mining for legal quotes (emphasize challenging cases)
    if hard_negative_weight > 1.0:
        # Find hardest negatives (highest similarity non-diagonal elements)
        text_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        graph_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        text_hard_negs = text_fused_sim.masked_fill(~text_mask, -float("inf"))
        graph_hard_negs = graph_fused_sim.masked_fill(~graph_mask, -float("inf"))

        # Weight hard negatives more heavily
        text_fused_sim = text_fused_sim + (
            text_hard_negs * (hard_negative_weight - 1.0)
        ).masked_fill(text_mask, 0)
        graph_fused_sim = graph_fused_sim + (
            graph_hard_negs * (hard_negative_weight - 1.0)
        ).masked_fill(graph_mask, 0)

    # Positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=device)

    # NT-Xent normalization for stability (critical for legal text variance)
    if use_nt_xent:
        text_fused_sim = F.normalize(text_fused_sim, p=2, dim=1)
        graph_fused_sim = F.normalize(graph_fused_sim, p=2, dim=1)

    # InfoNCE loss for both directions
    text_to_fused_loss = F.cross_entropy(text_fused_sim, labels)
    graph_to_fused_loss = F.cross_entropy(graph_fused_sim, labels)

    # Symmetric contrastive loss
    total_loss = (text_to_fused_loss + graph_to_fused_loss) / 2

    # Compute alignment metrics for monitoring
    with torch.no_grad():
        # Positive pair similarities (alignment quality)
        pos_text_similarities = torch.diagonal(torch.matmul(text_norm, fused_norm.T))
        pos_graph_similarities = torch.diagonal(torch.matmul(graph_norm, fused_norm.T))

        metrics = {
            "text_alignment": pos_text_similarities.mean().item(),
            "graph_alignment": pos_graph_similarities.mean().item(),
            "text_alignment_std": pos_text_similarities.std().item(),
            "graph_alignment_std": pos_graph_similarities.std().item(),
            "adaptive_temperature": (
                adaptive_temperature
                if isinstance(adaptive_temperature, float)
                else adaptive_temperature.item()
            ),
            "text_loss": text_to_fused_loss.item(),
            "graph_loss": graph_to_fused_loss.item(),
        }

    return total_loss, metrics


def compute_scaled_cosine_loss(
    reconstructed: torch.Tensor, target: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    """
    GraphMAE scaled-cosine loss for better reconstruction.
    Maps features to unit sphere and scales easy samples down.
    """
    # Normalize to unit sphere
    recon_norm = F.normalize(reconstructed, dim=1)
    target_norm = F.normalize(target, dim=1)

    # Cosine similarity
    cos_sim = F.cosine_similarity(recon_norm, target_norm, dim=1)

    # Scaled loss - reduce easy samples (high similarity)
    scaled_loss = (1 - cos_sim) ** gamma

    return scaled_loss.mean()


def compute_hybrid_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.7,
    cosine_weight: float = 0.3,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Hybrid MSE + scaled-cosine loss for balanced reconstruction.
    """
    mse_loss = F.mse_loss(reconstructed, target)
    cosine_loss = compute_scaled_cosine_loss(reconstructed, target, gamma)

    return mse_weight * mse_loss + cosine_weight * cosine_loss


def compute_dgi_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    num_negative: int = 30,  # Increased negative sampling
) -> torch.Tensor:
    """
    Deep Graph Infomax auxiliary loss for structural consistency.
    """
    batch_size = embeddings.shape[0]

    # Global summary via mean pooling
    global_summary = embeddings.mean(dim=0, keepdim=True)

    # Positive scores (local nodes vs global)
    pos_scores = torch.mm(embeddings, global_summary.t()).squeeze()

    # Negative sampling - corrupt node features
    neg_indices = torch.randperm(batch_size)[:num_negative]
    neg_embeddings = embeddings[neg_indices]
    neg_scores = torch.mm(neg_embeddings, global_summary.t()).squeeze()

    # Binary classification loss
    pos_loss = F.logsigmoid(pos_scores).mean()
    neg_loss = F.logsigmoid(-neg_scores).mean()

    return -(pos_loss + neg_loss)


def compute_graph_embedding(
    text: str,
    method: Literal["wl", "node2vec", "graphsage"],
    node2vec_model: Node2Vec | None = None,
    graphsage_model: nn.Module | None = None,
    fuse_model: nn.Module | None = None,
) -> torch.Tensor:
    """
    Compute a graph embedding vector for a single sentence:
      - wl: converts via wl_vector (counts → sparse → to_dense)
      - node2vec: runs random walks (requires trained model)
      - graphsage: runs mini-batch through GNN (trained model)
    """
    g_nx = to_dependency_graph(text)

    if method == "wl":
        # reuse existing wl_vector and densify
        from .wl_features import wl_vector

        sp = wl_vector(text).toarray().ravel()
        return torch.from_numpy(sp).float()

    pyg = _nx_to_pyg(g_nx)

    if method == "node2vec":
        if node2vec_model is None:
            raise ValueError("Node2Vec model not initialized")
        # run one forward pass
        return node2vec_model(pyg.edge_index)

    if method == "graphsage":
        # Get or create GraphSAGE model with fixed input dimensions
        if graphsage_model is None:
            graphsage_model = get_graphsage_embedder(
                in_channels=16, hidden_channels=128, num_layers=2
            )

        # Handle empty graphs
        if pyg.num_nodes == 0 or pyg.x is None:
            return torch.zeros(128)  # Return zero vector with correct hidden dimension

        # Run GNN and mean-pool node embeddings (model should be trained by now)
        with torch.no_grad():
            node_embs = graphsage_model(pyg.x, pyg.edge_index)
            return node_embs.mean(dim=0)  # Mean pooling across nodes

    raise ValueError(f"Unknown graph embedding method: {method}")
