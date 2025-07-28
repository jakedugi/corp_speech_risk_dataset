# ----------------------------------
# tests/unit/test_coral_loss.py
# ----------------------------------
import torch
from coral_ordinal.losses import coral_targets, coral_loss


def test_coral_targets():
    y = torch.tensor([0, 1, 2])
    t = coral_targets(y, num_classes=3)
    assert t.shape == (3, 2)
    assert torch.all(t[0] == torch.tensor([0.0, 0.0]))
    assert torch.all(t[2] == torch.tensor([1.0, 1.0]))


def test_coral_loss_runs():
    logits = torch.randn(4, 2)
    y = torch.tensor([0, 1, 2, 1])
    loss = coral_loss(logits, y, 3)
    assert loss.requires_grad
