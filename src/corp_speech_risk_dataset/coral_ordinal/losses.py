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


def hybrid_loss(
    class_logits: torch.Tensor,
    reg_output: torch.Tensor,
    labels: torch.Tensor,
    lambda_cls: float = 0.7,
    lambda_reg: float = 0.3,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Hybrid loss combining classification and regression for ordinal data.

    Args:
        class_logits: Classification logits (B, num_classes)
        reg_output: Regression output (B, 1)
        labels: True class indices (B,)
        lambda_cls: Weight for classification loss
        lambda_reg: Weight for regression loss
        label_smoothing: Label smoothing factor
    """
    # Classification loss (cross-entropy)
    if label_smoothing > 0:
        # Smoothed cross-entropy
        num_classes = class_logits.size(1)
        confidence = 1.0 - label_smoothing
        smooth_positives = confidence
        smooth_negatives = label_smoothing / (num_classes - 1)

        log_probs = F.log_softmax(class_logits, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, labels.unsqueeze(1), 1
        )
        targets_smooth = (
            targets_one_hot * smooth_positives
            + (1 - targets_one_hot) * smooth_negatives
        )

        ce_loss = -(targets_smooth * log_probs).sum(dim=1).mean()
    else:
        ce_loss = F.cross_entropy(class_logits, labels)

    # Regression loss (MSE on ordinal labels)
    reg_targets = labels.float().unsqueeze(1)
    mse_loss = F.mse_loss(reg_output, reg_targets)

    # Combined loss
    total_loss = lambda_cls * ce_loss + lambda_reg * mse_loss

    return total_loss
