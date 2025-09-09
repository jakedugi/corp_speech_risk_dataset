# =============================
# tests/test_pplm_classifier.py
# =============================
import torch
from pplm_ordinal.classifier_api import SoftmaxOrdinalClassifier


def test_softmax_classifier_shapes():
    m = SoftmaxOrdinalClassifier(in_dim=10, num_classes=4)
    x = torch.randn(2, 10)
    logits = m(x)
    assert logits.shape == (2, 4)
