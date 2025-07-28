# ----------------------------------
# tests/unit/test_coral_buckets.py
# ----------------------------------
import numpy as np
from coral_ordinal.buckets import make_buckets, bucketize


def test_bucketize():
    vals = np.arange(0, 100)
    edges, labels = make_buckets(
        vals, labels=["Low", "Med", "High"], qcuts=[0, 0.33, 0.66, 1]
    )
    assert bucketize(5, edges, labels) in labels
