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
      - graphsage: runs mini-batch through GNN (requires trained model)
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

        # Run GNN and mean-pool node embeddings
        with torch.no_grad():  # Ensure we're in eval mode for consistency
            node_embs = graphsage_model(pyg.x, pyg.edge_index)
            return node_embs.mean(dim=0)  # Mean pooling across nodes

    raise ValueError(f"Unknown graph embedding method: {method}")
