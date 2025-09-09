# =============================
# pplm_ordinal/metrics.py
# =============================
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from dataclasses import dataclass


@dataclass
class Exact:
    value: float


@dataclass
class OffByOne:
    value: float


@dataclass
class Spearman:
    value: float
    pvalue: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    exact = (y_true == y_pred).mean()
    off1 = (np.abs(y_true - y_pred) <= 1).mean()
    rho, p = spearmanr(y_true, y_pred)
    return {
        "exact": Exact(exact),
        "off_by_one": OffByOne(off1),
        "spearman_r": Spearman(rho, p),
    }


def go_no_go_gate(metrics: dict, exact_thr=0.40, off1_thr=0.80, rho_thr=0.5) -> bool:
    return (
        metrics["exact"].value >= exact_thr
        and metrics["off_by_one"].value >= off1_thr
        and metrics["spearman_r"].value >= rho_thr
    )
