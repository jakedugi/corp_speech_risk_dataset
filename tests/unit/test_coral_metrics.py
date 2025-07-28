# ----------------------------------
# tests/unit/test_coral_metrics.py
# ----------------------------------
import numpy as np
from coral_ordinal.metrics import compute_metrics


def test_metrics():
    y = np.array([0, 1, 2, 2])
    p = np.array([0, 1, 1, 2])
    m = compute_metrics(y, p)
    assert 0 <= m["exact"].value <= 1
    assert 0 <= m["off_by_one"].value <= 1
