# ----------------------------------
# coral_ordinal/losses.py
# ----------------------------------
import torch
import torch.nn.functional as F


def coral_targets(y: torch.Tensor, num_classes: int, eps: float = 0.0):
    """Convert class indices (0..K-1) to CORAL targets shape (B, K-1) of 0/1 indicating y>k"""
    # y: (B,)
    B = y.size(0)
    Km1 = num_classes - 1
    # each column k is 1 if y > k else 0
    arange = torch.arange(Km1, device=y.device).unsqueeze(0).expand(B, -1)
    hard = (y.unsqueeze(1) > arange).float()

    if eps <= 0:
        return hard
    # Label smoothing: symmetric smoothing in [eps/2, 1 - eps/2]
    return hard * (1 - eps) + 0.5 * eps


def coral_threshold_prevalences(labels: list, num_classes: int):
    """Compute prevalence P(y>k) for each threshold k"""
    import numpy as np

    y = np.asarray(labels)
    km1 = num_classes - 1
    prev = []
    for k in range(km1):
        prev.append(float((y > k).mean()))
    return np.array(prev)


def coral_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    reduction="mean",
    lambda_k=None,
    label_smoothing=0.0,
):
    """CORAL loss with optional threshold reweighting and label smoothing"""
    targets = coral_targets(y, num_classes, eps=label_smoothing)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )  # [B, K-1]

    if lambda_k is not None:
        loss = loss * lambda_k  # Reweight per threshold

    return loss.mean() if reduction == "mean" else loss.sum()
