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
    """Multi-layer perceptron with CORAL ordinal head and residual connections."""

    def __init__(
        self, in_dim: int, num_classes: int, hidden_dims=(768, 512, 256), dropout=0.1
    ):
        super().__init__()

        # Optional input normalization
        self.input_bn = nn.BatchNorm1d(in_dim)

        # Input projection for first residual connection
        self.input_proj = (
            nn.Linear(in_dim, hidden_dims[0])
            if in_dim != hidden_dims[0]
            else nn.Identity()
        )

        # Initialize input projection properly
        if in_dim != hidden_dims[0]:
            nn.init.kaiming_normal_(
                self.input_proj.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.constant_(self.input_proj.bias, 0)

        # Build residual blocks
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

        self.head = CORALHead(prev, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
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

        # Final CORAL head
        logits = self.head(out)  # (B, K-1)
        return logits

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        # class = number of thresholds passed
        return (probs > threshold).sum(dim=1).long()


class HybridOrdinalMLP(nn.Module):
    """Hybrid ordinal regression model with classification + regression heads."""

    def __init__(
        self, in_dim: int, num_classes: int, hidden_dims=(768, 512, 256), dropout=0.1
    ):
        super().__init__()
        self.num_classes = num_classes

        # Reuse the residual backbone from CORALMLP
        # Optional input normalization
        self.input_bn = nn.BatchNorm1d(in_dim)

        # Input projection for first residual connection
        self.input_proj = (
            nn.Linear(in_dim, hidden_dims[0])
            if in_dim != hidden_dims[0]
            else nn.Identity()
        )

        # Initialize input projection properly
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

        # Dual heads for hybrid loss
        self.class_head = nn.Linear(prev, num_classes)  # CE classification
        self.reg_head = nn.Linear(prev, 1)  # MSE regression

        # Initialize heads
        nn.init.kaiming_normal_(
            self.class_head.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.class_head.bias, 0)
        nn.init.kaiming_normal_(
            self.reg_head.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.reg_head.bias, 0)

    def forward(self, x):
        # Same residual backbone as CORALMLP
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

        # Dual outputs
        class_logits = self.class_head(out)  # (B, num_classes)
        reg_output = self.reg_head(out)  # (B, 1)

        return class_logits, reg_output

    @torch.no_grad()
    def predict(self, x):
        class_logits, reg_output = self.forward(x)

        # Classification prediction
        class_pred = torch.argmax(class_logits, dim=1)

        # Regression prediction (clamp to valid range)
        reg_pred = torch.clamp(
            reg_output.squeeze(1).round(), 0, self.num_classes - 1
        ).long()

        return class_pred, reg_pred, reg_output.squeeze(1)
