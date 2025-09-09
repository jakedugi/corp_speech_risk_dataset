# ----------------------------------
# Binary Fusion Model with Pure MLP Head
# Adapts the CORAL ordinal architecture for binary classification
# Retains cross-modal fusion and residual backbone, swaps head for binary MLP
# ----------------------------------
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryMLPHead(nn.Module):
    """Pure MLP head for binary classification.

    Replaces CORAL ordinal head with simpler binary architecture:
    - Two-layer MLP with BatchNorm and Dropout
    - Sigmoid output for binary probability
    - No ordinal constraints or monotonic biases
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        # Two-layer MLP for binary decision
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary logit

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MLP with ReLU activation
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)  # Raw logits for BCE loss
        return logits.squeeze(1)


class BinaryFusionMLP(nn.Module):
    """Binary classification model with cross-modal fusion and residual MLP backbone.

    Architecture:
    - Input: Fused embeddings from CrossModalFusion (768D)
    - Backbone: 3-layer residual MLP (768 → 512 → 256)
    - Head: Binary MLP head (256 → 128 → 1)
    - Output: Binary probability via sigmoid

    Retains from CORAL architecture:
    - Residual connections for gradient flow
    - Progressive dropout for regularization
    - BatchNorm for training stability

    Removes:
    - CORAL ordinal head
    - Monotonic threshold constraints
    - Ordinal-specific loss functions
    """

    def __init__(
        self,
        in_dim: int = 768,  # Legal-BERT fusion dimension
        hidden_dims: tuple[int, ...] = (768, 512, 256),
        head_hidden: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()

        # Input normalization
        self.input_bn = nn.BatchNorm1d(in_dim)

        # Input projection for first residual connection
        self.input_proj = (
            nn.Linear(in_dim, hidden_dims[0])
            if in_dim != hidden_dims[0]
            else nn.Identity()
        )

        # Initialize input projection
        if in_dim != hidden_dims[0]:
            nn.init.kaiming_normal_(
                self.input_proj.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.constant_(self.input_proj.bias, 0)

        # Build residual blocks (same as CORALMLP)
        self.blocks = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        prev = hidden_dims[0]
        for i, h in enumerate(hidden_dims):
            # Create residual block: [Linear, BatchNorm, ReLU, Dropout]
            linear = nn.Linear(prev, h)
            nn.init.kaiming_normal_(linear.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(linear.bias, 0)

            block = nn.Sequential(
                linear,
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout * (i + 1)),  # Progressive dropout
            )
            self.blocks.append(block)

            # Projection for next residual connection if dimensions differ
            next_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else h
            if h != next_dim:
                res_proj = nn.Linear(h, next_dim)
                nn.init.kaiming_normal_(
                    res_proj.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(res_proj.bias, 0)
                self.res_projs.append(res_proj)
            else:
                self.res_projs.append(nn.Identity())

            prev = h

        # Binary MLP head instead of CORAL
        self.head = BinaryMLPHead(prev, head_hidden, head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization and projection
        x = self.input_bn(x)
        out = self.input_proj(x)

        # Apply residual blocks
        for i, block in enumerate(self.blocks):
            residual = block(out)

            # Apply residual connection
            if i == 0:
                # First block: direct residual
                out = out + residual
            else:
                # Subsequent blocks: project previous output if needed
                projected_out = self.res_projs[i - 1](out)
                out = projected_out + residual

        # Binary head
        logits = self.head(out)  # (B,) raw logits
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get binary probabilities."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions with configurable threshold."""
        probs = self.predict_proba(x)
        return (probs > threshold).long()


class BinaryFusionMLPWithFeatures(BinaryFusionMLP):
    """Extended binary model that can optionally concatenate interpretable features.

    Allows hybrid approaches:
    - Fusion embeddings only
    - Interpretable features only
    - Concatenated fusion + features
    """

    def __init__(
        self,
        in_dim: int = 768,
        feature_dim: int = 0,  # Additional interpretable features
        hidden_dims: tuple[int, ...] = (768, 512, 256),
        head_hidden: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        # Adjust input dimension if using features
        total_dim = in_dim + feature_dim
        super().__init__(
            in_dim=total_dim,
            hidden_dims=hidden_dims,
            head_hidden=head_hidden,
            dropout=dropout,
            head_dropout=head_dropout,
        )
        self.embed_dim = in_dim
        self.feature_dim = feature_dim

    def forward(
        self, x: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Optionally concatenate features
        if features is not None and self.feature_dim > 0:
            x = torch.cat([x, features], dim=1)

        return super().forward(x)


def get_binary_fusion_model(
    embed_dim: int = 768,
    feature_dim: int = 0,
    hidden_dims: tuple[int, ...] = (768, 512, 256),
    head_hidden: int = 128,
    dropout: float = 0.1,
    head_dropout: float = 0.3,
    use_features: bool = False,
) -> nn.Module:
    """Factory function to create binary fusion model.

    Args:
        embed_dim: Dimension of fused embeddings (768 for Legal-BERT)
        feature_dim: Dimension of additional features (0 if not used)
        hidden_dims: Hidden layer dimensions for residual backbone
        head_hidden: Hidden dimension for binary MLP head
        dropout: Base dropout rate (progressive in backbone)
        head_dropout: Dropout rate for binary head
        use_features: Whether to use concatenated features

    Returns:
        BinaryFusionMLP or BinaryFusionMLPWithFeatures model
    """
    if use_features and feature_dim > 0:
        return BinaryFusionMLPWithFeatures(
            in_dim=embed_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            head_hidden=head_hidden,
            dropout=dropout,
            head_dropout=head_dropout,
        )
    else:
        return BinaryFusionMLP(
            in_dim=embed_dim,
            hidden_dims=hidden_dims,
            head_hidden=head_hidden,
            dropout=dropout,
            head_dropout=head_dropout,
        )
