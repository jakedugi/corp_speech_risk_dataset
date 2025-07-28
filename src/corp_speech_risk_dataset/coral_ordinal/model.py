# ----------------------------------
# coral_ordinal/model.py
# ----------------------------------
from __future__ import annotations
import torch
import torch.nn as nn


class CORALHead(nn.Module):
    """Linear layer with shared weights for CORAL (one weight vector, K-1 biases).
    This enforces monotonicity as in the original paper.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_thresholds = num_classes - 1
        self.weight = nn.Parameter(torch.randn(in_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(self.num_thresholds))

    def forward(self, x):
        # x: (B, D) -> logits: (B, K-1)
        base = x @ self.weight  # (B,1)
        logits = base + self.bias  # broadcast (B, K-1)
        return logits.squeeze(1)


class CORALMLP(nn.Module):
    def __init__(
        self, in_dim: int, num_classes: int, hidden_dims=(512, 128), dropout=0.1
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = CORALHead(prev, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)  # (B, K-1)
        return logits

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        # class = number of thresholds passed
        return (probs > threshold).sum(dim=1).long()
