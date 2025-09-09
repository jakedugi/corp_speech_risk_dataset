# =============================
# tests/test_pplm_metrics_gate.py
# =============================
import numpy as np
from pplm_ordinal.metrics import compute_metrics, go_no_go_gate


def test_gate():
    y = np.array([0, 1, 2, 2, 1])
    p = np.array([0, 1, 1, 2, 1])
    m = compute_metrics(y, p)
    assert isinstance(go_no_go_gate(m), bool)
