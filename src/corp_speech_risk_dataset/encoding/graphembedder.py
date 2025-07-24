from __future__ import annotations
from functools import lru_cache
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec, SAGEConv

from .parser import to_dependency_graph


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
    for node_id in range(len(g.nodes())):
        # Get node attributes
        node_data = g.nodes[node_id] if node_id in g.nodes else {"pos": "X", "text": ""}

        # Simple degree feature
        degree = g.degree(node_id) if node_id in g.nodes else 0

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
        feature[0] = min(degree / 10.0, 1.0)  # normalized degree
        feature[1 + pos_idx] = 1.0  # POS one-hot
        feature[14] = (
            min(node_id / len(g.nodes()), 1.0) if len(g.nodes()) > 0 else 0
        )  # relative position
        feature[15] = 1.0  # bias term

        node_features.append(feature)

    if not node_features:
        # Empty graph fallback
        node_features = [torch.zeros(16)]

    x = torch.stack(node_features)
    return Data(x=x, edge_index=edge_index, num_nodes=len(g))


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


@lru_cache(maxsize=None)
def get_graphsage_embedder(
    in_channels: int = 16, hidden_channels: int = 128, num_layers: int = 2
) -> nn.Module:
    """
    Returns a small inductive GraphSAGE model with fixed input dimensions.
    Default in_channels=16 matches our fixed node feature size.
    """

    class GraphSAGEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
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
                if i < len(self.convs) - 1:  # Apply ReLU to all but last layer
                    x = F.relu(x)
            return x

    return GraphSAGEModel()


class CrossModalFusion(nn.Module):
    """
    Simple two-stream cross-attention fusion:
      - text_embs: [B, Tdim]
      - graph_embs: [B, Gdim]
    returns fused [B, Fdim] where Fdim = Tdim
    """

    def __init__(self, text_dim: int, graph_dim: Optional[int] = None):
        super().__init__()
        # project text and graph embeddings into the same space
        graph_dim = graph_dim or text_dim
        self.text_proj = nn.Linear(text_dim, text_dim)
        self.graph_proj = nn.Linear(graph_dim, text_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=4, batch_first=True
        )
        self.proj = nn.Linear(text_dim * 2, text_dim)

    def forward(self, txt: torch.Tensor, gph: torch.Tensor) -> torch.Tensor:
        # Project into shared dimension
        txt_up = self.text_proj(txt)  # [B, Tdim]
        gph_up = self.graph_proj(gph)  # [B, Tdim]
        # Add a sequence dimension of 1 for attention
        txt_seq = txt_up.unsqueeze(1)  # [B,1,Tdim]
        gph_seq = gph_up.unsqueeze(1)  # [B,1,Tdim]
        # Cross-attend: text attends to graph
        o1, _ = self.attn(txt_seq, gph_seq, gph_seq)
        # Graph attends to text
        o2, _ = self.attn(gph_seq, txt_seq, txt_seq)
        # Remove sequence dim, concatenate, then project
        fused = torch.cat([o1.squeeze(1), o2.squeeze(1)], dim=-1)  # [B,2*Tdim]
        return self.proj(fused)  # [B,Tdim]


def train_graphsage_model(
    model: nn.Module, graphs: list[Data], epochs: int = 15
) -> nn.Module:
    """
    Train GraphSAGE model optimized for Mac M1 MPS and legal text dependency graphs.
    Uses contrastive learning + reconstruction for better embeddings.
    """
    # Enable MPS acceleration on Mac M1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Optimized for Mac M1 - higher LR, Adam with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Persistent decoder for reconstruction loss
    decoder = nn.Linear(128, 16).to(device)
    decoder_opt = torch.optim.AdamW(decoder.parameters(), lr=0.001)

    print(f"[GRAPHSAGE TRAINING] Training on {len(graphs)} graphs for {epochs} epochs")
    print(f"[GRAPHSAGE TRAINING] Device: {device}")
    print(f"[GRAPHSAGE TRAINING] Optimized for legal text dependency structures")

    # Move graphs to device
    graphs = [g.to(device) for g in graphs if g.num_nodes > 0]

    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0

        for pyg_data in graphs:
            if pyg_data.num_nodes == 0:
                continue

            optimizer.zero_grad()
            decoder_opt.zero_grad()

            # Forward pass
            embeddings = model(pyg_data.x, pyg_data.edge_index)

            # Reconstruction loss
            reconstructed = decoder(embeddings)
            recon_loss = F.mse_loss(reconstructed, pyg_data.x)

            # Additional contrastive loss for legal text patterns
            # Encourage similar POS patterns to have similar embeddings
            pos_mask = pyg_data.x[:, 1:13]  # POS one-hot features
            if embeddings.size(0) > 1:
                # Compute similarity matrix
                emb_sim = torch.mm(embeddings, embeddings.t())
                pos_sim = torch.mm(pos_mask, pos_mask.t())
                contrastive_loss = F.mse_loss(emb_sim, pos_sim * 0.1)  # Scale down

                total_loss_batch = recon_loss + 0.1 * contrastive_loss
            else:
                total_loss_batch = recon_loss

            total_loss_batch.backward()
            optimizer.step()
            decoder_opt.step()

            total_loss += total_loss_batch.item()
            total_samples += 1

        scheduler.step()
        avg_loss = total_loss / max(total_samples, 1)

        # Progress logging optimized for development
        if epoch % 3 == 0 or epoch == epochs - 1:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[GRAPHSAGE TRAINING] Epoch {epoch:2d}: loss = {avg_loss:.4f}, lr = {lr:.6f}"
            )

        # Early stopping for quick development
        if avg_loss < best_loss:
            best_loss = avg_loss
        elif epoch > 5 and avg_loss > best_loss * 1.1:
            print(f"[GRAPHSAGE TRAINING] Early stopping at epoch {epoch}")
            break

    model.eval()
    model = model.cpu()  # Move back to CPU for inference
    print(f"[GRAPHSAGE TRAINING] Training complete! Final loss: {best_loss:.4f}")
    return model


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
