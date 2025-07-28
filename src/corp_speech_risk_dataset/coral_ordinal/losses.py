# ----------------------------------
# coral_ordinal/losses.py
# ----------------------------------
import torch
import torch.nn.functional as F


def coral_targets(y: torch.Tensor, num_classes: int):
    """Convert class indices (0..K-1) to CORAL targets shape (B, K-1) of 0/1 indicating y>k"""
    # y: (B,)
    B = y.size(0)
    Km1 = num_classes - 1
    # each column k is 1 if y > k else 0
    arange = torch.arange(Km1, device=y.device).unsqueeze(0).expand(B, -1)
    return (y.unsqueeze(1) > arange).float()


def coral_loss(
    logits: torch.Tensor, y: torch.Tensor, num_classes: int, reduction="mean"
):
    targets = coral_targets(y, num_classes)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    return loss
