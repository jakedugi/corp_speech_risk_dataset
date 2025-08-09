from __future__ import annotations
from functools import lru_cache
from typing import Literal, Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Node2Vec, SAGEConv, global_mean_pool
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .parser import to_dependency_graph

from contextlib import nullcontext

# --------------------------------------------------------------------------- #
# Feature dimensionality configuration for GraphSAGE fusion
# --------------------------------------------------------------------------- #
# Base token-level feature vector used historically by GraphSAGE
# [degree (1), pos_one_hot (12), position (1), bias (1)] = 16
BASE_NODE_DIM: int = 16

# Global quote/context feature groups (kept in raw form; no compression):
#  - quote_sentiment (3) + context_sentiment (3)
#  - quote_deontic_count (1) + context_deontic_count (1)
#  - quote_pos (11) + context_pos (11)
#  - quote_ner (7) + context_ner (7)
#  - quote_deps (23) + context_deps (23)
#  - quote_wl (1) + context_wl (1)
# Total extra = 92
GLOBAL_FEATURE_DIM: int = 3 + 3 + 1 + 1 + 11 + 11 + 7 + 7 + 23 + 23 + 1 + 1

# Optional node-type indicator to allow GraphSAGE to distinguish node roles:
# [is_token, is_quote_node, is_context_node]
NODE_TYPE_DIM: int = 3

# Final per-node input dimensionality seen by GraphSAGE
GRAPH_INPUT_DIM: int = BASE_NODE_DIM + GLOBAL_FEATURE_DIM + NODE_TYPE_DIM


def _extract_raw_features(raw: Optional[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-layout global feature vectors for quote and context.

    The layout is strictly ordered to ensure consistent dimensionality and
    compatibility across training/inference runs.

    Returns a tuple of tensors: (quote_vec_46, context_vec_46).
    Missing keys default to zeros. No normalization/scaling is applied
    (values are passed in current raw form by design).
    """
    # Initialize zero vectors for the two halves (46 dims each)
    quote_vec = torch.zeros(46, dtype=torch.float32)
    context_vec = torch.zeros(46, dtype=torch.float32)

    if raw is None:
        return quote_vec, context_vec

    # Helper to safely get list/number
    def _to_list(val, expected_len: int) -> List[float]:
        if val is None:
            return [0.0] * expected_len
        if isinstance(val, (list, tuple)):
            # Pad/trim to expected length for strict shape
            arr = list(val)[:expected_len]
            if len(arr) < expected_len:
                arr += [0.0] * (expected_len - len(arr))
            return [float(x) for x in arr]
        # Scalar case
        return [float(val)] + [0.0] * (expected_len - 1)

    # Offsets for quote half
    q_off = 0
    # quote_sentiment (3)
    q_vals = _to_list(raw.get("quote_sentiment"), 3)
    quote_vec[q_off : q_off + 3] = torch.tensor(q_vals)
    q_off += 3
    # quote_deontic_count (1)
    quote_vec[q_off] = float(raw.get("quote_deontic_count", 0.0))
    q_off += 1
    # quote_pos (11)
    q_vals = _to_list(raw.get("quote_pos"), 11)
    quote_vec[q_off : q_off + 11] = torch.tensor(q_vals)
    q_off += 11
    # quote_ner (7)
    q_vals = _to_list(raw.get("quote_ner"), 7)
    quote_vec[q_off : q_off + 7] = torch.tensor(q_vals)
    q_off += 7
    # quote_deps (23) – counts + depth
    q_vals = _to_list(raw.get("quote_deps"), 23)
    quote_vec[q_off : q_off + 23] = torch.tensor(q_vals)
    q_off += 23
    # quote_wl (1)
    quote_vec[q_off] = float(raw.get("quote_wl", 0.0))

    # Offsets for context half
    c_off = 0
    # context_sentiment (3)
    c_vals = _to_list(raw.get("context_sentiment"), 3)
    context_vec[c_off : c_off + 3] = torch.tensor(c_vals)
    c_off += 3
    # context_deontic_count (1)
    context_vec[c_off] = float(raw.get("context_deontic_count", 0.0))
    c_off += 1
    # context_pos (11)
    c_vals = _to_list(raw.get("context_pos"), 11)
    context_vec[c_off : c_off + 11] = torch.tensor(c_vals)
    c_off += 11
    # context_ner (7)
    c_vals = _to_list(raw.get("context_ner"), 7)
    context_vec[c_off : c_off + 7] = torch.tensor(c_vals)
    c_off += 7
    # context_deps (23)
    c_vals = _to_list(raw.get("context_deps"), 23)
    context_vec[c_off : c_off + 23] = torch.tensor(c_vals)
    c_off += 23
    # context_wl (1)
    context_vec[c_off] = float(raw.get("context_wl", 0.0))

    return quote_vec, context_vec


def get_best_device() -> torch.device:
    """Get best available device with consistent priority: CUDA → CPU → MPS"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")  # Prefer CPU over MPS for stability


def _nx_to_pyg(g: nx.DiGraph, raw_features: Optional[Dict] = None) -> Data:
    """Convert a dependency NetworkX DiGraph to a PyG Data with fused features.

    Node features layout per node (no compression):
      - BASE_NODE_DIM (16): degree + POS one-hot (12) + relative position + bias
      - GLOBAL_FEATURE_DIM (92): concatenated [quote_vec_46 | context_vec_46]
      - NODE_TYPE_DIM (3): one-hot for [token, quote_node, context_node]

    Additionally, two special nodes are appended at the end of the graph:
      - QUOTE node carrying (quote_vec, zeros) and type=[0,1,0]
      - CONTEXT node carrying (zeros, context_vec) and type=[0,0,1]

    Both special nodes are connected with directed edges to all token nodes
    to broadcast global signals (star connections).
    """
    # Map original nodes to contiguous indices
    mapping = {n: i for i, n in enumerate(g.nodes())}

    # Build edge list for existing dependency edges
    if len(g.edges()) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            [[mapping[u] for u, v in g.edges()], [mapping[v] for u, v in g.edges()]],
            dtype=torch.long,
        )

    # Prepare global feature vectors (46 each) and concatenation (92)
    quote_vec46, context_vec46 = _extract_raw_features(raw_features)
    global_vec92 = torch.cat([quote_vec46, context_vec46], dim=0)  # [92]

    # Create token node features
    token_features: List[torch.Tensor] = []
    num_tokens = g.number_of_nodes()
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
    for node_id in range(num_tokens):
        node_data = g.nodes[node_id] if node_id in g.nodes else {"pos": "X", "text": ""}
        degree_val = int(g.degree(node_id)) if node_id in g.nodes else 0

        # Base features (16)
        base = torch.zeros(BASE_NODE_DIM)
        base[0] = min(degree_val / 10.0, 1.0)
        pos_idx = pos_map.get(node_data.get("pos", "X"), 11)
        base[1 + pos_idx] = 1.0
        base[14] = min(node_id / max(num_tokens, 1), 1.0)
        base[15] = 1.0

        # Node-type indicator for token
        node_type = torch.tensor([1.0, 0.0, 0.0])

        feat = torch.cat([base, global_vec92, node_type], dim=0)  # [GRAPH_INPUT_DIM]
        token_features.append(feat)

    # If graph is empty, create one dummy token node (all zeros except bias)
    if not token_features:
        base = torch.zeros(BASE_NODE_DIM)
        base[15] = 1.0
        node_type = torch.tensor([1.0, 0.0, 0.0])
        token_features = [torch.cat([base, global_vec92, node_type], dim=0)]
        num_tokens = 1

    # Create QUOTE and CONTEXT special nodes
    base_zero = torch.zeros(BASE_NODE_DIM)
    quote_type = torch.tensor([0.0, 1.0, 0.0])
    context_type = torch.tensor([0.0, 0.0, 1.0])
    quote_node_feat = torch.cat(
        [base_zero, torch.cat([quote_vec46, torch.zeros(46)], dim=0), quote_type]
    )
    context_node_feat = torch.cat(
        [base_zero, torch.cat([torch.zeros(46), context_vec46], dim=0), context_type]
    )

    # Stack features
    x = torch.stack(token_features + [quote_node_feat, context_node_feat])

    # Append star edges from special nodes to all token nodes
    if num_tokens > 0:
        quote_idx = num_tokens
        context_idx = num_tokens + 1
        extra_src = []
        extra_dst = []
        for t in range(num_tokens):
            extra_src += [quote_idx, context_idx]
            extra_dst += [t, t]
        if extra_src:
            extra_edges = torch.tensor([extra_src, extra_dst], dtype=torch.long)
            if edge_index.numel() == 0:
                edge_index = extra_edges
            else:
                edge_index = torch.cat([edge_index, extra_edges], dim=1)

    num_nodes = x.shape[0]
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
    use_amp: bool = False,
    patience: int = 8,
    lr: float = 5e-4,
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
    print(
        f"[GRAPHSAGE TRAINING] Optimizing link prediction (AP/AUROC) + DGI; recon as regularizer"
    )

    # Optimized for extended training - configurable LR, higher weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Determine hidden dimension dynamically from the model
    try:
        hidden_dim = model.convs[-1].out_channels  # type: ignore[attr-defined]
    except Exception:
        hidden_dim = 256

    # Persistent decoder for reconstruction loss - ensure on same device
    # Match output to current graph input dimensionality
    decoder = nn.Linear(hidden_dim, GRAPH_INPUT_DIM).to(device)
    decoder_opt = torch.optim.AdamW(decoder.parameters(), lr=lr)

    # Move graphs to device - fix device assignment
    train_graphs = [g.to(device) for g in train_graphs if g.num_nodes > 0]
    val_graphs = [g.to(device) for g in val_graphs if g.num_nodes > 0]

    # Compute per-feature mean/std for loss-only standardization
    if train_graphs:
        with torch.no_grad():
            all_x = torch.cat([g.x for g in train_graphs], dim=0)
            feat_mean = all_x.mean(dim=0)
            feat_std = all_x.std(dim=0, unbiased=False)
            # Clamp std to avoid exploding standardized magnitudes
            feat_std = feat_std.clamp_min(5e-2)
    else:
        feat_mean = torch.zeros(GRAPH_INPUT_DIM, device=device)
        feat_std = torch.ones(GRAPH_INPUT_DIM, device=device)

    # Exclude node-type (last 3) and bias (BASE 16th) from standardization by
    # forcing mean=0, std=1 for those indices
    # bias index within base block
    bias_idx = BASE_NODE_DIM - 1
    feat_mean[bias_idx] = 0.0
    feat_std[bias_idx] = 1.0
    # node-type indices
    type_start = BASE_NODE_DIM + GLOBAL_FEATURE_DIM
    feat_mean[type_start:] = 0.0
    feat_std[type_start:] = 1.0

    # Group weights: base(16)=1.0, global(92)=0.15, type(3)=0.05
    loss_weights = torch.ones(GRAPH_INPUT_DIM, device=device)
    loss_weights[:BASE_NODE_DIM] = 1.0
    loss_weights[BASE_NODE_DIM : BASE_NODE_DIM + GLOBAL_FEATURE_DIM] = 0.15
    loss_weights[BASE_NODE_DIM + GLOBAL_FEATURE_DIM :] = 0.05

    best_val_loss = float("inf")
    best_val_ap = -1.0
    best_val_dgi = float("inf")
    best_model_state = None
    best_decoder_state = None
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = patience  # from arguments

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        total_samples = 0

        # Process in batches using PyG Batch for better GPU utilization
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i : i + batch_size]
            if not batch_graphs:
                continue

            optimizer.zero_grad()
            decoder_opt.zero_grad()

            batch = Batch.from_data_list(batch_graphs)
            x = batch.x
            edge_index = batch.edge_index

            # Enable AMP only if requested and running on CUDA
            use_amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if (use_amp and device.type == "cuda")
                else nullcontext()
            )

            with use_amp_ctx:
                embeddings = model(x, edge_index)
                reconstructed = decoder(embeddings)

                # Standardized, weighted reconstruction loss
                x_std = (x - feat_mean) / feat_std
                rec_std = (reconstructed - feat_mean) / feat_std
                if loss_type == "hybrid":
                    mse = ((rec_std - x_std) ** 2) * loss_weights
                    mse_loss = mse.mean()
                    cosine_loss = compute_scaled_cosine_loss(rec_std, x_std)
                    recon_loss = 0.7 * mse_loss + 0.3 * cosine_loss
                elif loss_type == "cosine":
                    recon_loss = compute_scaled_cosine_loss(rec_std, x_std)
                else:
                    mse = ((rec_std - x_std) ** 2) * loss_weights
                    recon_loss = mse.mean()

                # DGI contrastive loss over node embeddings in the batch
                # Link prediction loss (binary cross-entropy on pos/neg edges)
                link_bce = compute_linkpred_bce(embeddings, edge_index)

                # DGI contrastive loss for structural consistency
                dgi_loss = (
                    compute_dgi_loss(embeddings, edge_index, num_negative)
                    if dgi_weight > 0
                    else torch.tensor(0.0, device=device)
                )

                # Total objective: 0.2*recon + 1.0*link + 0.1*DGI
                loss = 0.2 * recon_loss + 1.0 * link_bce + 0.1 * dgi_loss

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(decoder.parameters()), max_norm=1.0
            )
            optimizer.step()
            decoder_opt.step()

            total_loss += float(loss.detach().item())
            total_samples += 1

        scheduler.step()
        avg_train_loss = total_loss / max(total_samples, 1)
        train_losses.append(avg_train_loss)

        # Validation phase
        if val_graphs:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            # Link prediction metrics accumulation
            all_scores = []
            all_labels = []
            dgi_vals = []

            with torch.no_grad():
                for pyg_data in val_graphs:
                    if pyg_data.num_nodes == 0 or pyg_data.x is None:
                        continue

                    embeddings = model(pyg_data.x, pyg_data.edge_index)
                    reconstructed = decoder(embeddings)

                    # Validation mirrors standardized, weighted loss
                    x = pyg_data.x
                    x_std = (x - feat_mean) / feat_std
                    rec_std = (reconstructed - feat_mean) / feat_std
                    if loss_type == "hybrid":
                        mse = ((rec_std - x_std) ** 2) * loss_weights
                        mse_loss = mse.mean()
                        cosine_loss = compute_scaled_cosine_loss(rec_std, x_std)
                        vloss = 0.7 * mse_loss + 0.3 * cosine_loss
                    elif loss_type == "cosine":
                        vloss = compute_scaled_cosine_loss(rec_std, x_std)
                    else:
                        mse = ((rec_std - x_std) ** 2) * loss_weights
                        vloss = mse.mean()
                    val_loss += float(vloss.item())
                    val_samples += 1

                    # Link scores for AP/AUROC
                    scores, labels = compute_linkpred_scores(
                        embeddings, pyg_data.edge_index
                    )
                    if scores.numel() > 0:
                        all_scores.append(scores.cpu())
                        all_labels.append(labels.cpu())

                    # DGI on val graph
                    dgi_vals.append(
                        compute_dgi_loss(
                            embeddings, pyg_data.edge_index, num_negative
                        ).item()
                    )

            avg_val_loss = val_loss / max(val_samples, 1)
            val_losses.append(avg_val_loss)

            # Aggregate link metrics
            if all_scores:
                scores_cat = torch.cat(all_scores).numpy()
                labels_cat = torch.cat(all_labels).numpy()
                try:
                    val_ap = float(average_precision_score(labels_cat, scores_cat))
                except Exception:
                    val_ap = 0.0
                try:
                    val_auc = float(roc_auc_score(labels_cat, scores_cat))
                except Exception:
                    val_auc = 0.0
            else:
                val_ap = 0.0
                val_auc = 0.0

            avg_val_dgi = (
                float(sum(dgi_vals) / max(len(dgi_vals), 1)) if dgi_vals else 0.0
            )
        else:
            avg_val_loss = avg_train_loss  # Fallback if no validation set
            val_ap = 0.0
            val_auc = 0.0
            avg_val_dgi = 0.0

        # Progress logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[GRAPHSAGE TRAINING] Epoch {epoch:2d}: "
                f"train_loss = {avg_train_loss:.2e}, "
                f"val_MSE = {avg_val_loss:.2e}, "
                f"val_AP = {val_ap:.3f}, val_AUC = {val_auc:.3f}, "
                f"val_DGI = {avg_val_dgi:.3f}, "
                f"lr = {lr:.6f}"
            )

        # Early stopping on val AP (higher is better). Tie-breaker: lower val DGI.
        improved = val_ap > best_val_ap + 1e-4 or (
            abs(val_ap - best_val_ap) <= 1e-4 and avg_val_dgi < best_val_dgi
        )
        if improved:
            best_val_ap = val_ap
            best_val_dgi = avg_val_dgi
            best_val_loss = avg_val_loss
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_decoder_state = {
                k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch > 20:  # Allow longer training
            print(f"[GRAPHSAGE TRAINING] Early stopping at epoch {epoch}")
            break

    model.eval()

    # Restore best model/decoder if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)

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
            "target_metric": "val_AP",
            "best_val_ap": best_val_ap,
            "best_val_dgi": best_val_dgi,
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

            # Store pooled embeddings for downstream tasks (per-graph)
            graph_batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
            pooled = global_mean_pool(embeddings, graph_batch)
            val_embeddings.append(pooled.squeeze(0).cpu().numpy())

            # Reconstruction loss - ensure decoder is on same device
            if decoder is not None:
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
    in_channels: int = GRAPH_INPUT_DIM,
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
        num_heads: int = 8,
    ):
        super().__init__()
        # Set default dimensions - prioritize Legal-BERT compatibility
        graph_dim = graph_dim or text_dim
        fusion_dim = fusion_dim or max(
            text_dim, graph_dim
        )  # Use larger dimension as default (768 for Legal-BERT)

        # Ensure fusion dimension is appropriate for Legal-BERT (768D)
        if text_dim == 768:  # Legal-BERT case
            fusion_dim = 768  # Store final fused embeddings in 768D

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


class MomentumEncoder(nn.Module):
    """Momentum encoder with a FIFO queue of fused negatives (optional).

    - Maintains a queue of size `queue_size` in fused space for stronger negatives
    - Mirrors the query encoder weights via EMA updates
    """

    def __init__(
        self, model: CrossModalFusion, momentum: float = 0.999, queue_size: int = 4096
    ):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        # Mirror the fusion model (for future use if we encode keys separately)
        self.key_encoder = CrossModalFusion(
            text_dim=model.text_proj.in_features,
            graph_dim=model.graph_proj.in_features,
            fusion_dim=model.proj.out_features,
            dropout=model.final_dropout.p,
            num_heads=model.attn.num_heads,
        )
        self.key_encoder.load_state_dict(model.state_dict())
        self.key_encoder.eval()

        # Queue buffers (fused_dim x queue_size)
        fusion_dim = model.proj.out_features
        queue = torch.randn(fusion_dim, queue_size)
        queue_ptr = torch.zeros(1, dtype=torch.long)
        self.register_buffer("queue", F.normalize(queue, dim=0))
        self.register_buffer("queue_ptr", queue_ptr)

    @torch.no_grad()
    def update_key_encoder(self, query_encoder: CrossModalFusion) -> None:
        for param_q, param_k in zip(
            query_encoder.parameters(), self.key_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1 - self.momentum
            )

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        # keys: [B, fused_dim] (assumed normalized)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, : batch_size - remaining] = keys[remaining:].T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


def train_crossmodal_fusion(
    fusion_model: CrossModalFusion,
    text_embeddings: List[torch.Tensor],
    graph_embeddings: List[torch.Tensor],
    epochs: int = 15,
    batch_size: int = 256,
    temperature: float = 0.2,
    patience: int = 3,
    use_amp: bool = False,
    adaptive_temperature: bool = True,
    hard_negative_weight: float = 1.3,
    use_momentum_encoder: bool = False,
    queue_size: int = 4096,
    momentum: float = 0.999,
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

    # FIXED: Lower LR for numerical stability with AMP and contrastive learning
    optimizer = torch.optim.AdamW(
        fusion_model.parameters(), lr=0.0003, weight_decay=0.01  # 3e-4 instead of 1e-3
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

    # Optional momentum encoder and negatives queue
    momentum_enc: MomentumEncoder | None = None
    if use_momentum_encoder:
        momentum_enc = MomentumEncoder(
            fusion_model, momentum=momentum, queue_size=queue_size
        ).to(device)

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

                # FIXED: Pre-project inputs using model's registered projections
                proj_text = fusion_model.text_proj(batch_text)
                proj_graph = fusion_model.graph_proj(batch_graph)

                # Enhanced InfoNCE contrastive loss - legal text optimized
                momentum_queue = (
                    getattr(momentum_enc, "queue", None) if momentum_enc else None
                )
                batch_loss, batch_metrics = compute_infonce_loss(
                    fused,
                    proj_text,
                    proj_graph,
                    temperature=temperature,
                    use_nt_xent=use_amp,
                    adaptive_temp=adaptive_temperature,
                    hard_negative_weight=hard_negative_weight,
                    momentum_queue=momentum_queue,
                )

            batch_loss.backward()

            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update momentum encoder and enqueue fresh keys (normalized fused)
            if momentum_enc is not None:
                with torch.no_grad():
                    momentum_enc.update_key_encoder(fusion_model)
                    keys = F.normalize(fused, dim=1)
                    momentum_enc.enqueue(keys)

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

            # FIXED: Project both text and graph for symmetry, then normalize
            batch_fused_norm = F.normalize(batch_fused, dim=1, eps=1e-8)
            batch_text_proj = fusion_model.text_proj(batch_text)
            batch_text_norm = F.normalize(batch_text_proj, dim=1, eps=1e-8)

            # Project graph embeddings to fusion dimension then normalize
            batch_graph_proj = fusion_model.graph_proj(batch_graph)
            batch_graph_norm = F.normalize(batch_graph_proj, dim=1, eps=1e-8)

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
    temperature: float = 0.2,  # Higher temperature for stability (was 0.07)
    use_nt_xent: bool = False,
    adaptive_temp: bool = True,
    hard_negative_weight: float = 1.0,
    momentum_queue: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Enhanced InfoNCE contrastive loss for cross-modal alignment.
    Optimized for legal text dependency graph alignment with stability improvements.

    FIXED BUGS:
    - Hard negative weighting now preserves diagonal (positive) logits
    - Uses pre-projected inputs instead of creating new layers per batch

    Args:
        fused: Fused embeddings [B, fusion_dim]
        text: Pre-projected text embeddings [B, fusion_dim]
        graph: Pre-projected graph embeddings [B, fusion_dim]
        temperature: Base temperature for InfoNCE
        use_nt_xent: Use NT-Xent normalization for stability
        adaptive_temp: Dynamically adjust temperature based on alignment
        hard_negative_weight: Weight for hard negative mining

    Returns:
        Tuple of (loss, metrics_dict)
    """
    batch_size = fused.shape[0]
    device = fused.device

    # FIXED: Removed per-batch projection layers - use pre-projected inputs
    # The caller should project using model's registered projections before calling this

    # Guard against zero vectors and NaNs
    text = torch.where(torch.isnan(text), torch.zeros_like(text), text)
    graph = torch.where(torch.isnan(graph), torch.zeros_like(graph), graph)
    fused = torch.where(torch.isnan(fused), torch.zeros_like(fused), fused)

    # L2 normalize with eps for stability
    fused_norm = F.normalize(fused, dim=1, eps=1e-8)
    text_norm = F.normalize(text, dim=1, eps=1e-8)
    graph_norm = F.normalize(graph, dim=1, eps=1e-8)

    # Adaptive temperature based on current alignment (for legal quote complexity)
    if adaptive_temp:
        # Compute similarities in FP32 for numerical stability
        with torch.autocast(device_type=device.type, enabled=False):
            pos_text_sim = torch.diagonal(
                torch.matmul(text_norm.float(), fused_norm.float().T)
            )
            pos_graph_sim = torch.diagonal(
                torch.matmul(graph_norm.float(), fused_norm.float().T)
            )
            mean_pos_sim = (pos_text_sim.mean() + pos_graph_sim.mean()) / 2

        # Adaptive temperature: higher temp for stability (0.2-0.5 range)
        adaptive_temperature = temperature * (1.5 + 0.5 * mean_pos_sim.clamp(0.1, 0.9))
    else:
        adaptive_temperature = temperature

    # Compute similarity matrices in FP32 for numerical stability
    with torch.autocast(device_type=device.type, enabled=False):
        # Text-to-fused similarity matrix
        text_fused_sim = (
            torch.matmul(text_norm.float(), fused_norm.float().T) / adaptive_temperature
        )

        # Graph-to-fused similarity matrix
        graph_fused_sim = (
            torch.matmul(graph_norm.float(), fused_norm.float().T)
            / adaptive_temperature
        )

        # Optionally extend negatives pool with momentum queue (in fused space)
        # The queue is stored as [fused_dim, queue_size]. Normalize and append as columns.
        if momentum_queue is not None and momentum_queue.numel() > 0:
            mq = F.normalize(momentum_queue.T.float(), dim=1)
            # Concatenate as additional negatives columns for both directions
            text_fused_sim = torch.cat(
                [
                    text_fused_sim,
                    torch.matmul(text_norm.float(), mq.T) / adaptive_temperature,
                ],
                dim=1,
            )
            graph_fused_sim = torch.cat(
                [
                    graph_fused_sim,
                    torch.matmul(graph_norm.float(), mq.T) / adaptive_temperature,
                ],
                dim=1,
            )

    # Hard negative mining for legal quotes (emphasize challenging cases)
    # FIXED: Only scale off-diagonal (negative) elements, preserve diagonal (positives)
    if hard_negative_weight > 1.0:
        # Off-diagonal mask (negatives only)
        mask_off = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        scale = hard_negative_weight - 1.0

        # Scale only off-diagonal negatives, leave diagonal positives untouched
        text_fused_sim = text_fused_sim + scale * text_fused_sim * mask_off
        graph_fused_sim = graph_fused_sim + scale * graph_fused_sim * mask_off

    # Positive pairs are on the first `batch_size` columns (original in-batch items)
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


def compute_linkpred_scores(
    embeddings: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scores and labels for link prediction on a single graph:
      - positives: existing edges
      - negatives: sampled with equal count via negative_sampling
    Score = dot product of node embeddings for (u,v)
    Returns (scores, labels) on the same device as embeddings
    """
    device = embeddings.device
    num_nodes = embeddings.shape[0]
    if edge_index.numel() == 0 or num_nodes < 2:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    pos = edge_index
    # Sample negatives matching count of positives
    try:
        neg = negative_sampling(
            edge_index=pos,
            num_nodes=num_nodes,
            num_neg_samples=pos.shape[1],
            method="sparse",
        )
    except Exception:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    def edge_scores(ei: torch.Tensor) -> torch.Tensor:
        src, dst = ei[0], ei[1]
        return (embeddings[src] * embeddings[dst]).sum(dim=1)

    pos_scores = edge_scores(pos)
    neg_scores = edge_scores(neg)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_scores.size(0), device=device),
            torch.zeros(neg_scores.size(0), device=device),
        ]
    )
    return scores, labels


def compute_linkpred_bce(
    embeddings: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy loss for link prediction with negative sampling."""
    scores, labels = compute_linkpred_scores(embeddings, edge_index)
    if scores.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_graph_embedding(
    text: str,
    method: Literal["wl", "node2vec", "graphsage"],
    node2vec_model: Node2Vec | None = None,
    graphsage_model: nn.Module | None = None,
    fuse_model: nn.Module | None = None,
    raw_features: Optional[Dict] = None,
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

    # Build PyG graph with optional raw features for fusion
    pyg = _nx_to_pyg(g_nx, raw_features=raw_features)

    if method == "node2vec":
        if node2vec_model is None:
            raise ValueError("Node2Vec model not initialized")
        # run one forward pass
        return node2vec_model(pyg.edge_index)

    if method == "graphsage":
        # Get or create GraphSAGE model with fixed input dimensions
        if graphsage_model is None:
            graphsage_model = get_graphsage_embedder(
                in_channels=GRAPH_INPUT_DIM, hidden_channels=128, num_layers=2
            )

        # Handle empty graphs
        if pyg.num_nodes == 0 or pyg.x is None:
            # Match current model hidden size
            try:
                hdim = graphsage_model.convs[-1].out_channels  # type: ignore[attr-defined]
            except Exception:
                hdim = 128
            return torch.zeros(hdim)

        # Run GNN and mean-pool node embeddings (model should be trained by now)
        with torch.no_grad():
            node_embs = graphsage_model(pyg.x, pyg.edge_index)
            return node_embs.mean(dim=0)  # Mean pooling across nodes

    raise ValueError(f"Unknown graph embedding method: {method}")
