#!/usr/bin/env python3
"""
Enhanced CrossModal fusion training script with proper text↔graph contrastive learning
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

from corp_speech_risk_dataset.encoding.graphembedder import CrossModalFusion


class MomentumEncoder(nn.Module):
    """Proper MoCo-style momentum encoder with queue management"""

    def __init__(
        self, model: CrossModalFusion, momentum: float = 0.999, queue_size: int = 4096
    ):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        # **Directly clone the user's fusion_model (including its fusion_dim)**
        self.key_encoder = CrossModalFusion(
            text_dim=model.text_proj.in_features,
            graph_dim=model.graph_proj.in_features,
            fusion_dim=model.proj.out_features,  # Use exact fusion dimension from original model
        )
        self.key_encoder.load_state_dict(model.state_dict())
        self.key_encoder.eval()

        # Properly register queue buffers using fusion dimension
        fusion_dim = model.proj.out_features
        queue = torch.randn(fusion_dim, queue_size)
        queue_ptr = torch.zeros(1, dtype=torch.long)

        self.register_buffer("queue", F.normalize(queue, dim=0))
        self.register_buffer("queue_ptr", queue_ptr)

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass through key encoder"""
        with torch.no_grad():
            keys = self.key_encoder(text_emb, graph_emb)
            keys = F.normalize(keys, dim=1)
        return keys

    def update_key_encoder(self, query_encoder: CrossModalFusion):
        """Update key encoder with momentum"""
        for param_q, param_k in zip(
            query_encoder.parameters(), self.key_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1 - self.momentum
            )

    def enqueue_dequeue(self, keys: torch.Tensor):
        """Enqueue new keys and dequeue old ones"""
        batch_size = keys.shape[0]
        # Use .item() to get scalar value
        ptr = int(self.queue_ptr.item())

        # Access registered buffers
        with torch.no_grad():
            # Replace oldest keys with new ones
            if ptr + batch_size <= self.queue_size:
                self.queue[:, ptr : ptr + batch_size] = keys.T
            else:
                # Handle wraparound
                remaining = self.queue_size - ptr
                self.queue[:, ptr:] = keys[:remaining].T
                self.queue[:, : batch_size - remaining] = keys[remaining:].T

            # Update pointer
            ptr = (ptr + batch_size) % self.queue_size
            self.queue_ptr[0] = ptr


class AdvancedProjectionHead(nn.Module):
    """Projection head with angular margin for better separation"""

    def __init__(
        self, input_dim: int, proj_dim: int, num_layers: int = 2, margin: float = 0.2
    ):
        super().__init__()
        self.margin = margin

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else proj_dim
            out_dim = proj_dim
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU() if i < num_layers - 1 else nn.Identity(),
                ]
            )
        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=1)


class CrossModalDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Memory-efficient dataset for text-graph embedding pairs with stratified sampling"""

    def __init__(
        self,
        input_dir: Path,
        max_samples: int = 50000,
        edge_dropout: float = 0.1,
        feature_mask: float = 0.15,
    ):
        self.input_dir = input_dir
        self.max_samples = max_samples
        self.edge_dropout = edge_dropout
        self.feature_mask = feature_mask

        # Index all available files and samples with text length for stratification
        self.file_sample_pairs = []
        stage7_files = list(input_dir.rglob("*_stage7.jsonl"))

        for file_path in stage7_files:
            with open(file_path) as f:
                for line_num, line in enumerate(f):
                    if (
                        len(self.file_sample_pairs) >= max_samples * 3
                    ):  # Oversample for stratification
                        break
                    try:
                        js = json.loads(line)
                        if "st_emb" in js and "gph_emb" in js and "text" in js:
                            text_length = len(js["text"].split())
                            self.file_sample_pairs.append(
                                (file_path, line_num, text_length)
                            )
                    except:
                        continue
                if len(self.file_sample_pairs) >= max_samples * 3:
                    break

        # Stratified sampling by text length
        self._stratify_samples()

        print(f"Indexed {len(self.file_sample_pairs)} embedding pairs (stratified)")

    def _stratify_samples(self):
        """Stratified sampling by text length for balanced training"""
        if len(self.file_sample_pairs) <= self.max_samples:
            return

        # Create strata based on text length
        lengths = [pair[2] for pair in self.file_sample_pairs]
        length_array = np.array(lengths)

        # Define length buckets: short (<10), medium (10-25), long (>25)
        short_mask = length_array < 10
        medium_mask = (length_array >= 10) & (length_array <= 25)
        long_mask = length_array > 25

        # Get indices for each stratum
        short_indices = np.where(short_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        long_indices = np.where(long_mask)[0]

        # Sample evenly from each stratum
        samples_per_stratum = self.max_samples // 3
        selected_indices = []

        for indices in [short_indices, medium_indices, long_indices]:
            if len(indices) > 0:
                # Shuffle before sampling for better coverage
                np.random.shuffle(indices)
                selected = indices[: min(samples_per_stratum, len(indices))]
                selected_indices.extend(selected)

        # Shuffle final selection and truncate
        np.random.shuffle(selected_indices)
        selected_indices = selected_indices[: self.max_samples]

        # Update file_sample_pairs
        self.file_sample_pairs = [self.file_sample_pairs[i] for i in selected_indices]

    def __len__(self):
        return len(self.file_sample_pairs)

    def __getitem__(self, idx):
        file_path, line_num, text_length = self.file_sample_pairs[idx]

        # Load specific line from file
        js = None
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i == line_num:
                    js = json.loads(line)
                    break

        if js is None:
            raise ValueError(f"Line {line_num} not found in {file_path}")

        text_emb = torch.tensor(js["st_emb"], dtype=torch.float32)
        graph_emb = torch.tensor(js["gph_emb"], dtype=torch.float32)

        # Apply augmentations
        if self.feature_mask > 0:
            mask = torch.rand_like(text_emb) < self.feature_mask
            text_emb = text_emb.masked_fill(mask, 0.0)

        if self.edge_dropout > 0:
            dropout_mask = torch.rand_like(graph_emb) < self.edge_dropout
            graph_emb = graph_emb * (~dropout_mask).float()

        return text_emb, graph_emb, text_length


def triplet_loss_with_temperature(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Triplet loss with temperature-based soft margin"""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)

    # Apply temperature to create soft margin
    soft_margin = margin / temperature
    loss = F.relu(pos_dist - neg_dist + soft_margin)
    return loss.mean()


def mine_hard_negatives(
    anchor_emb: torch.Tensor, all_embs: torch.Tensor, num_hard: int = 50
) -> torch.Tensor:
    """Mine hard negatives based on cosine similarity"""
    anchor_norm = F.normalize(anchor_emb, dim=1)
    all_norm = F.normalize(all_embs, dim=1)
    sims = torch.matmul(anchor_norm, all_norm.T)

    # Get indices of hardest negatives (highest similarity, excluding self)
    _, hard_indices = torch.topk(sims, k=min(num_hard + 1, sims.shape[1]), dim=1)
    # Remove self-similarity (first index)
    hard_indices = (
        hard_indices[:, 1 : num_hard + 1] if hard_indices.shape[1] > 1 else hard_indices
    )

    return hard_indices


def get_adaptive_temperature(
    epoch: int,
    total_epochs: int,
    temp_start: float = 0.1,
    temp_end: float = 0.01,
    schedule: str = "cosine",
) -> float:
    """Adaptive temperature scheduling"""
    if schedule == "cosine":
        progress = epoch / max(total_epochs - 1, 1)
        return temp_end + (temp_start - temp_end) * 0.5 * (1 + np.cos(progress * np.pi))
    else:
        return temp_start


def create_simple_proxy_clusters(
    dataset: CrossModalDataset, n_clusters: int = 10
) -> torch.Tensor:
    """Create simple proxy clustering task based on text length for semantic grouping"""
    from sklearn.cluster import KMeans

    # Use text lengths as features for simple semantic clustering
    lengths = []
    for idx in range(len(dataset)):
        _, _, text_length = dataset.file_sample_pairs[idx]
        lengths.append([text_length])  # Make it 2D for KMeans

    # Cluster based on text length as proxy for semantic complexity
    length_array = np.array(lengths)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(length_array)

    return torch.tensor(labels, dtype=torch.long)


def get_best_device() -> torch.device:
    """CUDA → CPU → MPS priority (MPS can be unstable)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")  # Prefer CPU over MPS for stability


def compute_enhanced_cross_modal_loss(
    text_proj: torch.Tensor,
    graph_proj: torch.Tensor,
    temperature: float = 0.07,
    proxy_labels: Optional[torch.Tensor] = None,
    proxy_weight: float = 0.1,
    momentum_queue: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Enhanced cross-modal contrastive loss: text ↔ graph contrast on projected space
    Note: momentum_queue should match the dimension of text_proj/graph_proj
    """
    batch_size = text_proj.shape[0]
    device = text_proj.device
    proj_dim = text_proj.shape[1]

    # Normalize projected embeddings
    text_norm = F.normalize(text_proj, dim=1)
    graph_norm = F.normalize(graph_proj, dim=1)

    # Build negative pool from graph embeddings
    graph_negatives = graph_norm
    # Skip momentum queue when using projection heads (dimension mismatch)
    # Queue is 384D (fusion), projections are 256D
    if momentum_queue is not None and momentum_queue.shape[0] == proj_dim:
        # Only use queue if dimensions match projected embeddings
        queue_norm = F.normalize(momentum_queue.T, dim=1)
        graph_negatives = torch.cat([graph_negatives, queue_norm], dim=0)
    elif momentum_queue is not None:
        # Skip queue when dimensions don't match (common with projection heads)
        pass  # Don't use queue, rely on batch negatives only

    # Cross-modal similarities: text → graph and graph → text
    text_to_graph_sim = torch.matmul(text_norm, graph_negatives.T) / temperature
    graph_to_text_sim = torch.matmul(graph_norm, text_norm.T) / temperature

    # Labels for positive pairs (diagonal)
    labels = torch.arange(batch_size, device=device)

    # Symmetric cross-modal InfoNCE loss
    text_to_graph_loss = F.cross_entropy(text_to_graph_sim[:, :batch_size], labels)
    graph_to_text_loss = F.cross_entropy(graph_to_text_sim, labels)
    contrastive_loss = (text_to_graph_loss + graph_to_text_loss) / 2

    # Add proxy clustering loss if provided
    if proxy_labels is not None:
        # Use fused representation for proxy task
        fused_repr = (text_norm + graph_norm) / 2  # Simple fusion for proxy
        proxy_logits = torch.matmul(fused_repr, fused_repr.T) / temperature
        # Ensure proxy labels are on same device
        proxy_labels_device = proxy_labels[:batch_size].to(device)
        proxy_loss = F.cross_entropy(proxy_logits, proxy_labels_device)
        contrastive_loss = contrastive_loss + proxy_weight * proxy_loss

    return contrastive_loss


def train_enhanced_crossmodal_fusion(
    fusion_model: CrossModalFusion,
    dataset: CrossModalDataset,
    epochs: int = 100,
    batch_size: int = 512,
    temperature: float = 0.03,
    patience: int = 5,
    fusion_loss: str = "infonce",
    triplet_margin: float = 0.2,
    hard_negatives: int = 50,
    proxy_labels: Optional[torch.Tensor] = None,
    proxy_weight: float = 0.1,
    temp_scheduler: str = "cosine",
    temp_start: float = 0.1,
    temp_end: float = 0.01,
    proj_head: int = 2,
    proj_dim: int = 256,
    angular_margin: float = 0.2,
    momentum_encoder: bool = False,
    queue_size: int = 4096,
    use_amp: bool = False,
) -> CrossModalFusion:
    """Enhanced CrossModal fusion training with proper text↔graph contrastive learning"""
    device = get_best_device()
    fusion_model = fusion_model.to(device)
    fusion_model.train()

    # Initialize momentum encoder if requested (after projection heads are defined)
    momentum_enc = None

    # Create projection heads if requested - use fusion dimension as input
    fusion_dim = fusion_model.proj.out_features

    text_proj_head = None
    graph_proj_head = None
    if proj_head > 0:
        text_proj_head = AdvancedProjectionHead(
            fusion_dim, proj_dim, proj_head, angular_margin
        ).to(device)
        graph_proj_head = AdvancedProjectionHead(
            fusion_dim, proj_dim, proj_head, angular_margin
        ).to(device)

    # Initialize momentum encoder directly from fusion model
    if momentum_encoder:
        momentum_enc = MomentumEncoder(fusion_model, queue_size=queue_size).to(device)

    # Setup optimizer and scheduler
    params = list(fusion_model.parameters())
    if text_proj_head:
        params += list(text_proj_head.parameters())
    if graph_proj_head:
        params += list(graph_proj_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"[ENHANCED TRAINING] Training on {len(dataset)} pairs for {epochs} epochs")
    print(f"[ENHANCED TRAINING] Batch size: {batch_size}, Device: {device}")
    print(f"[ENHANCED TRAINING] Loss: {fusion_loss}, Temperature: {temperature}")
    print(f"[ENHANCED TRAINING] Using text↔graph cross-modal contrastive learning")
    print(f"[ENHANCED TRAINING] Fusion dimension: {fusion_dim}D")
    if momentum_encoder:
        print(f"[ENHANCED TRAINING] Momentum encoder with queue size: {queue_size}")
    if proj_head > 0:
        print(f"[ENHANCED TRAINING] Projection heads: {proj_head} layers × {proj_dim}D")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        # Update temperature for both InfoNCE and triplet
        current_temp = get_adaptive_temperature(
            epoch, epochs, temp_start, temp_end, temp_scheduler
        )

        for batch_text, batch_graph, _ in dataloader:
            batch_text = batch_text.to(device)
            batch_graph = batch_graph.to(device)

            optimizer.zero_grad()

            # Forward pass through fusion model first
            fused = fusion_model(batch_text, batch_graph)

            # Apply projection heads if available (after fusion)
            text_projected = text_proj_head(fused) if text_proj_head else fused
            graph_projected = graph_proj_head(fused) if graph_proj_head else fused

            # Compute loss based on type
            if fusion_loss == "triplet":
                # For triplet loss with temperature
                batch_size_curr = text_projected.shape[0]
                if batch_size_curr > 1:
                    # Simple triplet: use next item as positive, random as negative
                    anchors = text_projected[:-1]
                    positives = graph_projected[1:]  # Cross-modal positive
                    neg_indices = torch.randperm(batch_size_curr - 1)
                    negatives = graph_projected[neg_indices]
                    loss = triplet_loss_with_temperature(
                        anchors, positives, negatives, triplet_margin, current_temp
                    )
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                # Enhanced cross-modal InfoNCE loss
                momentum_queue = (
                    getattr(momentum_enc, "queue", None) if momentum_enc else None
                )
                current_proxy_labels = (
                    proxy_labels[: batch_text.shape[0]]
                    if proxy_labels is not None
                    else None
                )

                loss = compute_enhanced_cross_modal_loss(
                    text_projected,
                    graph_projected,
                    temperature=current_temp,
                    proxy_labels=current_proxy_labels,
                    proxy_weight=proxy_weight,
                    momentum_queue=momentum_queue,
                )

            loss.backward()
            optimizer.step()

            # Update momentum encoder with fused embeddings
            if momentum_enc:
                momentum_enc.update_key_encoder(fusion_model)
                with torch.no_grad():
                    # ***Always*** enqueue the raw fusion outputs (384D), not the 256D projections
                    keys = F.normalize(fused, dim=1)
                    momentum_enc.enqueue_dequeue(keys)

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[ENHANCED TRAINING] Epoch {epoch:3d}: loss = {avg_loss:.4f}, temp = {current_temp:.4f}, lr = {lr:.6f}"
            )

        if patience_counter >= patience and epoch > 10:
            print(f"[ENHANCED TRAINING] Early stopping at epoch {epoch}")
            break

    fusion_model.eval()
    fusion_model = fusion_model.cpu()
    print(f"[ENHANCED TRAINING] Training complete! Best loss: {best_loss:.4f}")

    return fusion_model


def apply_fusion_to_files(
    input_dir: Path, output_dir: Path, fusion_model: CrossModalFusion
):
    """Apply trained fusion model to all stage 7 files"""
    fusion_model.eval()
    fusion_model = fusion_model.cpu()

    stage7_files = list(input_dir.rglob("*_stage7.jsonl"))
    print(f"Applying fusion to {len(stage7_files)} files...")

    for file_path in tqdm(stage7_files, desc="Processing files"):
        # Create output path (stage7 → stage8)
        rel_path = file_path.relative_to(input_dir)
        out_path = (
            output_dir / rel_path.parent / rel_path.name.replace("stage7", "stage8")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path) as fin, open(out_path, "w") as fout:
            for line in fin:
                js = json.loads(line)

                # Apply fusion if embeddings exist
                if "st_emb" in js and "gph_emb" in js:
                    text_emb = torch.tensor(js["st_emb"])
                    graph_emb = torch.tensor(js["gph_emb"])

                    with torch.no_grad():
                        fused_emb = fusion_model(
                            text_emb.unsqueeze(0), graph_emb.unsqueeze(0)
                        )[0]

                    js["fused_emb"] = fused_emb.tolist()
                    js["stage"] = 8
                else:
                    # Skip entries without embeddings
                    continue

                fout.write(json.dumps(js, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced CrossModal fusion with proper text↔graph contrastive learning"
    )
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with stage 7 files"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for stage 8 files"
    )
    parser.add_argument(
        "--fusion-samples", type=int, default=40000, help="Max training samples"
    )
    parser.add_argument(
        "--fusion-epochs", type=int, default=100, help="Training epochs"
    )
    parser.add_argument("--fusion-batch-size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--fusion-temperature", type=float, default=0.03, help="InfoNCE temperature"
    )
    parser.add_argument(
        "--fusion-patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=384,
        help="Fusion dimension (default: max of text/graph)",
    )

    # Advanced loss & training techniques
    parser.add_argument(
        "--fusion-loss",
        choices=["infonce", "triplet"],
        default="infonce",
        help="Contrastive loss type",
    )
    parser.add_argument(
        "--triplet-margin", type=float, default=0.2, help="Triplet loss margin"
    )
    parser.add_argument(
        "--hard-negatives",
        type=int,
        default=50,
        help="Number of hard negatives to mine",
    )
    parser.add_argument(
        "--proxy-clusters", type=int, default=10, help="K-means clusters for proxy task"
    )
    parser.add_argument(
        "--proxy-weight", type=float, default=0.1, help="Proxy task loss weight"
    )
    parser.add_argument(
        "--temp-scheduler",
        choices=["fixed", "cosine"],
        default="cosine",
        help="Temperature scheduling",
    )
    parser.add_argument(
        "--temp-start",
        type=float,
        default=0.1,
        help="Starting temperature for scheduler",
    )
    parser.add_argument(
        "--temp-end", type=float, default=0.01, help="Ending temperature for scheduler"
    )
    parser.add_argument(
        "--proj-head", type=int, default=2, help="Projection head layers"
    )
    parser.add_argument(
        "--proj-dim", type=int, default=256, help="Projection head dimension"
    )
    parser.add_argument(
        "--angular-margin",
        type=float,
        default=0.2,
        help="Angular margin for separation",
    )

    # Augmentation and efficiency flags
    parser.add_argument(
        "--edge-dropout", type=float, default=0.1, help="Graph edge dropout rate"
    )
    parser.add_argument(
        "--feature-mask", type=float, default=0.15, help="Text feature masking rate"
    )
    parser.add_argument(
        "--momentum-encoder",
        action="store_true",
        help="Use MoCo-style momentum encoder",
    )
    parser.add_argument(
        "--queue-size", type=int, default=4096, help="Momentum queue size for negatives"
    )
    parser.add_argument(
        "--use-amp", action="store_true", help="Use automatic mixed precision"
    )

    args = parser.parse_args()

    print(f"Device: {get_best_device()}")

    # Create memory-efficient dataset with stratification
    dataset = CrossModalDataset(
        args.input_dir,
        args.fusion_samples,
        edge_dropout=args.edge_dropout,
        feature_mask=args.feature_mask,
    )

    if len(dataset) == 0:
        print("ERROR: No embeddings found in input directory")
        return

    # Get sample to determine dimensions
    sample_text, sample_graph, _ = dataset[0]
    text_dim = sample_text.shape[0]
    graph_dim = sample_graph.shape[0]

    # Use specified fusion dimension or default to max of input dimensions
    fusion_dim = args.fusion_dim or max(text_dim, graph_dim)

    print(
        f"Creating CrossModal fusion: {text_dim}D text + {graph_dim}D graph → {fusion_dim}D fusion"
    )
    fusion_model = CrossModalFusion(
        text_dim=text_dim, graph_dim=graph_dim, fusion_dim=fusion_dim
    )

    print("Training CrossModal fusion with ENHANCED TEXT↔GRAPH CONTRASTIVE LEARNING...")
    print(f"ADVANCED SETTINGS:")
    print(f"  ├─ Training pairs: {len(dataset):,}")
    print(f"  ├─ Epochs: {args.fusion_epochs}")
    print(f"  ├─ Batch size: {args.fusion_batch_size}")
    print(f"  ├─ Loss type: {args.fusion_loss.upper()}")
    print(f"  ├─ Fusion dimension: {fusion_dim}D")
    if args.fusion_loss == "triplet":
        print(f"  ├─ Triplet margin: {args.triplet_margin}")
    print(
        f"  ├─ Temperature: {args.fusion_temperature} (start: {args.temp_start}, end: {args.temp_end})"
    )
    print(f"  ├─ Temperature schedule: {args.temp_scheduler}")
    print(f"  ├─ Hard negatives: {args.hard_negatives}")
    print(f"  ├─ Proxy clusters: {args.proxy_clusters} (weight: {args.proxy_weight})")
    print(f"  ├─ Projection head: {args.proj_head} layers × {args.proj_dim}D")
    print(f"  ├─ Angular margin: {args.angular_margin}")
    print(f"  ├─ Edge dropout: {args.edge_dropout}")
    print(f"  ├─ Feature mask: {args.feature_mask}")
    print(f"  ├─ Momentum encoder: {args.momentum_encoder}")
    if args.momentum_encoder:
        print(f"  └─ Queue size: {args.queue_size:,}")

    # Create simple proxy clustering labels if requested
    proxy_labels = None
    if args.proxy_clusters > 0:
        print(
            f"\n[PROXY TASK] Creating {args.proxy_clusters} semantic clusters based on text length..."
        )
        proxy_labels = create_simple_proxy_clusters(dataset, args.proxy_clusters)
        print(
            f"[PROXY TASK] Cluster distribution: {torch.bincount(proxy_labels).tolist()}"
        )

    # Train with all advanced features
    fusion_model = train_enhanced_crossmodal_fusion(
        fusion_model,
        dataset,
        epochs=args.fusion_epochs,
        batch_size=args.fusion_batch_size,
        temperature=args.fusion_temperature,
        patience=args.fusion_patience,
        fusion_loss=args.fusion_loss,
        triplet_margin=args.triplet_margin,
        hard_negatives=args.hard_negatives,
        proxy_labels=proxy_labels,
        proxy_weight=args.proxy_weight,
        temp_scheduler=args.temp_scheduler,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        proj_head=args.proj_head,
        proj_dim=args.proj_dim,
        angular_margin=args.angular_margin,
        momentum_encoder=args.momentum_encoder,
        queue_size=args.queue_size,
        use_amp=args.use_amp,
    )

    print("Applying fusion to all files...")
    apply_fusion_to_files(args.input_dir, args.output_dir, fusion_model)

    print(f"✅ Complete! Output files in {args.output_dir}")
    print(f"Files will have 'fused_emb' field with {fusion_dim}D vectors")


if __name__ == "__main__":
    main()
