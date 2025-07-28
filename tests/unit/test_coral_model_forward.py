# ----------------------------------
# tests/unit/test_coral_model_forward.py
# ----------------------------------
import torch
from coral_ordinal.model import CORALMLP


def test_forward():
    model = CORALMLP(in_dim=10, num_classes=3)
    x = torch.randn(5, 10)
    logits = model(x)
    assert logits.shape == (5, 2)
