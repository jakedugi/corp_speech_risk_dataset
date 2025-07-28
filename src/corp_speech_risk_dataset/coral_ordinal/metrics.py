# ----------------------------------
# coral_ordinal/metrics.py
# ----------------------------------
from __future__ import annotations
import torch
import numpy as np
from scipy.stats import spearmanr
from dataclasses import dataclass


@dataclass
class ExactMatch:
    value: float


@dataclass
class OffByOne:
    value: float


@dataclass
class SpearmanR:
    value: float
    pvalue: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    exact = (y_true == y_pred).mean()
    off1 = (np.abs(y_true - y_pred) <= 1).mean()
    rho, p = spearmanr(y_true, y_pred)
    return {
        "exact": ExactMatch(exact),
        "off_by_one": OffByOne(off1),
        "spearman_r": SpearmanR(rho, p),
    }
