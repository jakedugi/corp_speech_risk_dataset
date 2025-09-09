"""Temporal DEV split for tiny-data friendly CV.

This module implements the temporal-safe DEV policy that handles small validation sets
by intelligently combining train/val data while maintaining strict temporal ordering.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


@dataclass
class TemporalCVConfig:
    # DEV construction (tiny-data friendly)
    dev_tail_frac: float = 0.20  # start tail size; will expand if minima unmet
    min_dev_cases: int = 3  # was 10; too strict for your folds
    min_dev_quotes: int = 150  # fallback to 100 if needed
    require_all_classes: bool = False  # prefer all 3; accept >=2
    # Embargo (purge window BEFORE dev_start)
    embargo_days: int = 90
    # Whether we may include early VAL cases (strictly earlier than TEST)
    include_early_val_in_dev: bool = True


def _class_set(ids: List[str], y_map: Dict[str, int]) -> set:
    return {y_map[cid] for cid in ids if cid in y_map}


def _ensure_dev_tail(
    candidate_ids: List[str],
    df_cases: pd.DataFrame,
    y_map: Dict[str, int],
    cfg: TemporalCVConfig,
) -> Tuple[List[str], List[str]]:
    """
    candidate_ids: outer TRAIN (and optionally early VAL) restricted to < test_start.
    Returns (train_core_ids, dev_ids) with DEV a time-contiguous tail.
    """
    if not candidate_ids:
        return [], []
    sub = df_cases.loc[candidate_ids].sort_values("case_time")
    n = len(sub)
    for frac in [cfg.dev_tail_frac, 0.25, 0.30, 0.35, 0.40, 0.50]:
        k = max(cfg.min_dev_cases, int(np.ceil(n * frac)))
        k = min(k, n)
        dev = sub.iloc[-k:]
        trn = sub.iloc[:-k]
        if trn.empty:
            continue
        dev_cases = len(dev)
        dev_quotes = int(dev["n_quotes"].sum())
        classes = _class_set(dev.index.tolist(), y_map)
        have_enough_classes = len(classes) >= (3 if cfg.require_all_classes else 2)
        have_enough_quotes = (dev_quotes >= cfg.min_dev_quotes) or (dev_quotes >= 100)
        if (
            dev_cases >= cfg.min_dev_cases
            and have_enough_quotes
            and have_enough_classes
        ):
            return trn.index.tolist(), dev.index.tolist()
    # Fallback: smallest feasible tail (may be class-sparse)
    k = max(cfg.min_dev_cases, int(np.ceil(n * cfg.dev_tail_frac)))
    k = min(k, n)
    return sub.iloc[:-k].index.tolist(), sub.iloc[-k:].index.tolist()


def _apply_embargo_before_dev(
    train_core_ids: List[str],
    dev_ids: List[str],
    df_cases: pd.DataFrame,
    embargo_days: int,
) -> List[str]:
    """
    Purge TRAIN cases that lie in the embargo interval immediately BEFORE DEV.
    Keep only training cases with time <= (dev_start - embargo_days).
    """
    if embargo_days <= 0 or not train_core_ids or not dev_ids:
        return train_core_ids
    dev_start = df_cases.loc[dev_ids, "case_time"].min()
    lower_bound = dev_start - pd.Timedelta(days=embargo_days)
    keep = [
        cid for cid in train_core_ids if df_cases.loc[cid, "case_time"] <= lower_bound
    ]
    return keep


def build_temporal_inner_splits(
    outer_train: List[str],
    outer_test: List[str],
    df_cases: pd.DataFrame,
    y_map: Dict[str, int],
    cfg: Optional[TemporalCVConfig] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (train_core_ids, dev_ids, test_ids) for a single outer fold.
    df_cases index must be case_id; columns: case_time (datetime64), n_quotes (int).
    outer_test is unchanged; we never touch TEST.
    """
    if cfg is None:
        cfg = TemporalCVConfig()

    # TEST starts at the earliest test case time
    test_start = df_cases.loc[outer_test, "case_time"].min()

    # Candidate DEV pool: all OUTER TRAIN strictly earlier than TEST
    pool = [cid for cid in outer_train if df_cases.loc[cid, "case_time"] < test_start]

    # Build DEV as a tail of the (time-ordered) pool
    pool = sorted(pool, key=lambda cid: df_cases.loc[cid, "case_time"])
    train_core_ids, dev_ids = _ensure_dev_tail(pool, df_cases, y_map, cfg)

    # Apply embargo BEFORE DEV to TRAIN_CORE
    train_core_ids = _apply_embargo_before_dev(
        train_core_ids, dev_ids, df_cases, cfg.embargo_days
    )

    return train_core_ids, dev_ids, outer_test
