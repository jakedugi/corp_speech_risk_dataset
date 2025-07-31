# =============================
# pplm_ordinal/classifier_api.py
# =============================
"""Classifier side API.
- OrdinalClassifierBase: abstract interface the PPLM loop calls.
- SoftmaxOrdinalClassifier: simple MLP head over LM hidden states (multi-class CE).
- load_classifier: convenience to restore from disk.

You can wrap ANY upstream model (e.g., your CORAL head) by subclassing OrdinalClassifierBase
and only implementing forward() to return logits and loss for a given target class.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Any


class OrdinalClassifierBase(nn.Module):
    num_classes: int

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, K)."""
        raise NotImplementedError

    def loss_for_class(self, logits: torch.Tensor, target_class: int) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            torch.full(
                (logits.size(0),), target_class, dtype=torch.long, device=logits.device
            ),
        )


class SoftmaxOrdinalClassifier(OrdinalClassifierBase):
    def __init__(
        self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        return self.net(reps)


@dataclass
class ClassifierCheckpoint:
    state_dict: Dict[str, Any]
    meta: Dict[str, Any]


def load_classifier(
    path: str | None, device: torch.device
) -> OrdinalClassifierBase | None:
    if path is None:
        return None
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "meta" in ckpt:
        meta = ckpt["meta"]
        model = SoftmaxOrdinalClassifier(
            meta["in_dim"],
            meta["num_classes"],
            hidden=meta.get("hidden", 256),
            dropout=meta.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
    raise ValueError("Unsupported classifier checkpoint format.")
