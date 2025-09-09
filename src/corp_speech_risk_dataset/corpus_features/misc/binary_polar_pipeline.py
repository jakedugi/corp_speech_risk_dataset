"""Binary POLAR (Proportional Odds Logistic Regression) pipeline with full CV protocol.

This module implements the complete paper-quality CV and final run protocol
for binary POLAR models, including:
- Binary column governance (exactly 10 features)
- Per-fold binary cutpoints
- Alpha-normalized combined weights
- Cumulative isotonic calibration
- Comprehensive evaluation metrics
- 4-fold CV + fold_4 final training structure
"""

from __future__ import annotations

# Temporarily disable orjson to avoid encoding issues
import json

FAST_JSON = False
print("📊 Using standard json (orjson disabled for debugging)")

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import time
import re
from datetime import datetime
from numpy.typing import NDArray
from sklearn.metrics import (
    f1_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler
from loguru import logger
import joblib

from .models import POLR, MLR, LR_L2, LR_L1, LR_ElasticNet, SVM_Linear
from .binary_column_governance import (
    validate_binary_columns,
    get_binary_feature_columns,
    validate_binary_target,
    validate_binary_weights,
    pyify,
)

# Exact 10 features we need (from feature importance analysis)
REQUIRED_BINARY_FEATURES = {
    "feat_new_attribution_verb_density",
    "feat_new2_deception_cluster_density",
    "interpretable_lex_deception_count",
    "interpretable_seq_trans_neutral_to_neutral",
    "feat_new3_neutral_edge_coverage",
    "feat_new2_attribution_verb_clustering_score",
    "feat_new5_deesc_half_life",
    "feat_new2_attr_verb_near_neutral_transition",
    "feat_new2_neutral_run_mean",
    "feat_new2_neutral_to_deception_transition_rate",
}

# Essential fields for training
REQUIRED_TRAINING_FIELDS = {
    "outcome_bin",  # Labels
    "sample_weight",  # Weights
    "case_id",  # For validation/grouping
}

ALL_REQUIRED_FIELDS = REQUIRED_BINARY_FEATURES | REQUIRED_TRAINING_FIELDS

# ============================================================================
# CPI AND TEMPORAL UTILITIES FOR INFLATION ADJUSTMENT
# ============================================================================


def load_cpi_data() -> pd.DataFrame:
    """Load US CPI-U seasonally adjusted monthly data (2000-2025).

    Source: BLS series CUSR0000SA0 (All items, U.S. city average, seasonally adjusted).
    Values embedded for Jan 2000 through Jul 2025.
    Base year for derived index: 2020 average = 100.
    Returns DataFrame with columns: year, month, cpi_sa, base_2020
    """
    from io import StringIO

    # Embedded monthly CPI-U (SA) values from BLS (CUSR0000SA0)
    csv_data = StringIO(
        """
# date,cpi_sa
2000-01,169.3
2000-02,170.0
2000-03,171.0
2000-04,170.9
2000-05,171.2
2000-06,172.2
2000-07,172.7
2000-08,172.7
2000-09,173.6
2000-10,173.9
2000-11,174.2
2000-12,174.6
2001-01,175.6
2001-02,176.0
2001-03,176.1
2001-04,176.4
2001-05,177.3
2001-06,177.7
2001-07,177.4
2001-08,177.4
2001-09,178.1
2001-10,177.6
2001-11,177.5
2001-12,177.4
2002-01,177.7
2002-02,178.0
2002-03,178.5
2002-04,179.3
2002-05,179.5
2002-06,179.6
2002-07,180.0
2002-08,180.5
2002-09,180.8
2002-10,181.2
2002-11,181.5
2002-12,181.8
2003-01,182.6
2003-02,183.6
2003-03,183.9
2003-04,183.2
2003-05,182.9
2003-06,183.1
2003-07,183.7
2003-08,184.5
2003-09,185.1
2003-10,184.9
2003-11,185.0
2003-12,185.5
2004-01,186.3
2004-02,186.7
2004-03,187.1
2004-04,187.4
2004-05,188.2
2004-06,188.9
2004-07,189.1
2004-08,189.2
2004-09,189.8
2004-10,190.8
2004-11,191.7
2004-12,191.7
2005-01,191.6
2005-02,192.4
2005-03,193.1
2005-04,193.7
2005-05,193.6
2005-06,193.7
2005-07,194.9
2005-08,196.1
2005-09,198.8
2005-10,199.1
2005-11,198.1
2005-12,198.1
2006-01,199.3
2006-02,199.4
2006-03,199.7
2006-04,200.7
2006-05,201.3
2006-06,201.8
2006-07,202.9
2006-08,203.8
2006-09,202.8
2006-10,201.9
2006-11,202.0
2006-12,203.1
2007-01,203.437
2007-02,204.226
2007-03,205.288
2007-04,205.904
2007-05,206.755
2007-06,207.234
2007-07,207.603
2007-08,207.667
2007-09,208.547
2007-10,209.190
2007-11,210.834
2007-12,211.445
2008-01,212.174
2008-02,212.687
2008-03,213.448
2008-04,213.942
2008-05,215.208
2008-06,217.463
2008-07,219.016
2008-08,218.690
2008-09,218.877
2008-10,216.995
2008-11,213.153
2008-12,211.398
2009-01,211.933
2009-02,212.705
2009-03,212.495
2009-04,212.709
2009-05,213.022
2009-06,214.790
2009-07,214.726
2009-08,215.445
2009-09,215.861
2009-10,216.509
2009-11,217.234
2009-12,217.347
2010-01,217.488
2010-02,217.281
2010-03,217.353
2010-04,217.403
2010-05,217.290
2010-06,217.199
2010-07,217.605
2010-08,217.923
2010-09,218.275
2010-10,219.035
2010-11,219.590
2010-12,220.472
2011-01,221.187
2011-02,221.898
2011-03,223.046
2011-04,224.093
2011-05,224.806
2011-06,224.806
2011-07,225.395
2011-08,226.106
2011-09,226.597
2011-10,226.750
2011-11,227.169
2011-12,227.223
2012-01,227.842
2012-02,228.329
2012-03,228.807
2012-04,229.187
2012-05,228.713
2012-06,228.524
2012-07,228.590
2012-08,229.918
2012-09,231.015
2012-10,231.638
2012-11,231.249
2012-12,231.221
2013-01,231.679
2013-02,232.937
2013-03,232.282
2013-04,231.797
2013-05,231.893
2013-06,232.445
2013-07,232.900
2013-08,233.456
2013-09,233.544
2013-10,233.669
2013-11,234.100
2013-12,234.719
2014-01,235.288
2014-02,235.547
2014-03,236.028
2014-04,236.468
2014-05,236.918
2014-06,237.231
2014-07,237.498
2014-08,237.460
2014-09,237.477
2014-10,237.430
2014-11,236.983
2014-12,236.252
2015-01,234.747
2015-02,235.342
2015-03,235.976
2015-04,236.222
2015-05,237.001
2015-06,237.657
2015-07,238.034
2015-08,238.033
2015-09,237.498
2015-10,237.733
2015-11,238.017
2015-12,237.761
2016-01,237.652
2016-02,237.336
2016-03,238.080
2016-04,238.992
2016-05,239.557
2016-06,240.222
2016-07,240.101
2016-08,240.545
2016-09,241.176
2016-10,241.741
2016-11,242.026
2016-12,242.637
2017-01,243.618
2017-02,244.006
2017-03,243.892
2017-04,244.193
2017-05,244.004
2017-06,244.163
2017-07,244.243
2017-08,245.183
2017-09,246.435
2017-10,246.626
2017-11,247.284
2017-12,247.805
2018-01,248.859
2018-02,249.529
2018-03,249.577
2018-04,250.227
2018-05,250.792
2018-06,251.018
2018-07,251.214
2018-08,251.663
2018-09,252.182
2018-10,252.772
2018-11,252.594
2018-12,252.767
2019-01,252.561
2019-02,253.319
2019-03,254.277
2019-04,255.233
2019-05,255.296
2019-06,255.213
2019-07,255.802
2019-08,256.036
2019-09,256.430
2019-10,257.155
2019-11,257.879
2019-12,258.630
2020-01,259.127
2020-02,259.250
2020-03,258.076
2020-04,256.032
2020-05,255.802
2020-06,257.042
2020-07,258.352
2020-08,259.316
2020-09,259.997
2020-10,260.319
2020-11,260.911
2020-12,262.045
2021-01,262.639
2021-02,263.573
2021-03,264.847
2021-04,266.625
2021-05,268.404
2021-06,270.710
2021-07,271.965
2021-08,272.752
2021-09,273.942
2021-10,276.528
2021-11,278.824
2021-12,280.806
2022-01,282.542
2022-02,284.525
2022-03,287.467
2022-04,288.582
2022-05,291.299
2022-06,295.072
2022-07,294.940
2022-08,295.162
2022-09,296.421
2022-10,297.979
2022-11,298.708
2022-12,298.808
2023-01,300.456
2023-02,301.476
2023-03,301.643
2023-04,302.858
2023-05,303.316
2023-06,304.099
2023-07,304.615
2023-08,306.138
2023-09,307.374
2023-10,307.653
2023-11,308.087
2023-12,308.735
2024-01,309.794
2024-02,311.022
2024-03,312.107
2024-04,313.016
2024-05,313.140
2024-06,313.131
2024-07,313.566
2024-08,314.131
2024-09,314.851
2024-10,315.564
2024-11,316.449
2024-12,317.603
2025-01,319.086
2025-02,319.775
2025-03,319.615
2025-04,320.321
2025-05,320.580
2025-06,321.500
2025-07,322.132
        """
    )

    # Parse the CSV into a DataFrame
    raw = pd.read_csv(csv_data, comment="#", header=None, names=["date", "cpi_sa"])
    # Split date into year and month
    raw[["year", "month"]] = raw["date"].str.split("-", expand=True).astype(int)
    raw.rename(columns={"cpi_sa": "cpi_sa"}, inplace=True)

    # Keep only requested span just in case
    cpi_df = raw[(raw["year"] >= 2000) & (raw["year"] <= 2025)].copy()

    # Compute 2020 average and 2020=100 index
    cpi_2020_avg = cpi_df[cpi_df["year"] == 2020]["cpi_sa"].mean()
    if pd.isna(cpi_2020_avg) or cpi_2020_avg == 0:
        raise ValueError("CPI 2020 average could not be computed from embedded data.")
    cpi_df["base_2020"] = (cpi_df["cpi_sa"] / cpi_2020_avg) * 100.0

    # Reorder/select columns
    cpi_df = cpi_df[["year", "month", "cpi_sa", "base_2020"]].reset_index(drop=True)

    logger.info(
        f"📊 Loaded CPI data: {len(cpi_df)} months (2000-2025 through Jul 2025), base year 2020 avg = {cpi_2020_avg:.3f}"
    )
    return cpi_df


def extract_case_date_from_id(case_id: str) -> Optional[Tuple[int, int]]:
    """Extract year and month from case_id naming conventions.

    Common patterns:
    - "court_year_month_id" -> extract year, month
    - "court_YYYY_case_num" -> extract year, default month=6
    - "YYYY-jurisdiction-number" -> extract year, default month=6

    Returns:
        (year, month) tuple or None if not extractable
    """

    # Pattern 1: YYYY-MM in case_id
    match = re.search(r"(\d{4})[_-](\d{1,2})", case_id)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 2000 <= year <= 2025 and 1 <= month <= 12:
            return year, month

    # Pattern 2: Just YYYY (default to mid-year)
    match = re.search(r"(\d{4})", case_id)
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2025:
            return year, 6  # Default to June

    return None


def deflate_to_real_dollars(
    nominal_amount: float,
    judgment_year: int,
    judgment_month: int,
    cpi_df: pd.DataFrame,
    base_year: int = 2020,
) -> float:
    """Convert nominal dollars to real dollars using CPI deflation.

    Args:
        nominal_amount: Dollar amount in judgment-time currency
        judgment_year: Year of judgment/settlement
        judgment_month: Month of judgment/settlement
        cpi_df: CPI data with year, month, base_2020 columns
        base_year: Base year for real dollars (default 2020)

    Returns:
        Real dollar amount in base_year terms
    """

    # Find CPI for judgment date
    judgment_cpi = cpi_df[
        (cpi_df["year"] == judgment_year) & (cpi_df["month"] == judgment_month)
    ]

    if judgment_cpi.empty:
        # Fallback: use annual average CPI for that year
        year_cpi = cpi_df[cpi_df["year"] == judgment_year]
        if year_cpi.empty:
            logger.warning(
                f"No CPI data for {judgment_year}-{judgment_month}, using 2020 baseline"
            )
            return nominal_amount  # Return nominal if no deflation possible

        judgment_cpi_value = year_cpi["base_2020"].mean()
        logger.debug(
            f"Using annual average CPI for {judgment_year}: {judgment_cpi_value:.1f}"
        )
    else:
        judgment_cpi_value = float(pd.Series(judgment_cpi["base_2020"]).iloc[0])

    # Get base year CPI (should be 100 for base_2020)
    base_cpi = 100.0  # By definition for base_2020

    # Convert: Real = Nominal * (CPI_base / CPI_judgment)
    real_amount = nominal_amount * (base_cpi / judgment_cpi_value)

    return real_amount


def create_temporal_split_with_dates(
    df: pd.DataFrame,
    inner_train_ratio: float = 0.8,
    cpi_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create inner temporal split using proper case dates for calibration.

    This ensures strict temporal ordering by extracting actual case dates,
    not just assumed ordering from case_id strings.
    """

    df_with_dates = df.copy()

    # Extract temporal information for sorting
    temporal_keys = []

    for _, row in df.iterrows():
        case_id_val = row.get("case_id", "")
        case_id = str(case_id_val) if case_id_val is not None else ""

        # Try to extract date from case_id
        date_tuple = extract_case_date_from_id(case_id)

        if date_tuple:
            year, month = date_tuple
            temporal_key = year * 100 + month  # YYYYMM format for sorting
        else:
            # Fallback: use case_id alphabetical order as proxy
            temporal_key = hash(case_id) % 1000000  # Deterministic but arbitrary

        temporal_keys.append(temporal_key)

    df_with_dates["temporal_key"] = temporal_keys

    # Sort by temporal order
    df_sorted = df_with_dates.sort_values(["temporal_key", "case_id"]).copy()

    # Split by time
    n_inner_train = int(len(df_sorted) * inner_train_ratio)

    inner_train_df = df_sorted.iloc[:n_inner_train].copy()
    inner_calib_df = df_sorted.iloc[n_inner_train:].copy()

    # Log temporal ranges
    train_years = [
        extract_case_date_from_id(str(cid)) for cid in inner_train_df["case_id"]
    ]
    calib_years = [
        extract_case_date_from_id(str(cid)) for cid in inner_calib_df["case_id"]
    ]

    train_year_range = [y[0] for y in train_years if y is not None]
    calib_year_range = [y[0] for y in calib_years if y is not None]

    if train_year_range and calib_year_range:
        logger.info(
            f"📅 Inner temporal split: Train {min(train_year_range)}-{max(train_year_range)}, "
            f"Calib {min(calib_year_range)}-{max(calib_year_range)}"
        )
    else:
        logger.info(
            f"📅 Inner temporal split: {len(inner_train_df)} inner-train, {len(inner_calib_df)} inner-calib"
        )

    return inner_train_df, inner_calib_df


def fast_load_jsonl_selective(file_path: Path) -> List[Dict[str, Any]]:
    """Load only the 10 required features + labels + weights from JSONL.

    This avoids loading hundreds of unused features, dramatically reducing memory usage.
    """
    start_time = time.time()
    data = []

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i % 5000 == 0 and i > 0:
                logger.info(f"📥 Loaded {i:,} lines from {file_path.name}...")

            # Parse full record (simplified)
            full_record = json.loads(line.strip())

            # Extract only required fields
            selective_record = {}
            for field in ALL_REQUIRED_FIELDS:
                if field in full_record:
                    selective_record[field] = full_record[field]
                # Handle missing fields gracefully
                elif field in REQUIRED_BINARY_FEATURES:
                    selective_record[field] = 0.0  # Default for missing features
                else:
                    # Missing essential fields - this is an error
                    if field in REQUIRED_TRAINING_FIELDS:
                        raise ValueError(
                            f"Missing required field '{field}' in {file_path.name}"
                        )

            data.append(selective_record)

    load_time = time.time() - start_time
    total_fields = len(data) * len(ALL_REQUIRED_FIELDS) if data else 0
    logger.info(
        f"✅ Selectively loaded {len(data):,} records ({total_fields:,} fields) from {file_path.name} in {load_time:.2f}s"
    )
    logger.info(f"💾 Memory savings: ~{(len(data) * 700):,} unused fields NOT loaded")
    return data


# Fallback function for compatibility
def fast_load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL with orjson acceleration if available."""
    return fast_load_jsonl_selective(file_path)


def fast_load_fold_data(
    fold_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test data for a fold with 3-5x faster JSON processing."""
    start_time = time.time()
    logger.info(f"🚀 Fast loading fold data from {fold_dir}...")

    # Load with fast JSON
    train_data = fast_load_jsonl(fold_dir / "train.jsonl")
    val_data = fast_load_jsonl(fold_dir / "val.jsonl")
    test_data = fast_load_jsonl(fold_dir / "test.jsonl")

    # Convert to DataFrames (this is still fast)
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    total_time = time.time() - start_time
    logger.info(
        f"✅ Fast loaded fold data in {total_time:.2f}s - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}"
    )

    return train_df, val_df, test_df


def debug_dataframe_nans(df: pd.DataFrame, name: str) -> None:
    """Debug NaN values in dataframe with detailed logging."""
    nan_info = df.isnull().sum()
    total_nans = nan_info.sum()

    if total_nans > 0:
        logger.warning(f"🚨 {name} has {total_nans:,} NaN values:")
        for col, count in nan_info[nan_info > 0].items():
            logger.warning(f"  - {col}: {count:,} NaNs ({count/len(df)*100:.1f}%)")
    else:
        logger.info(f"✅ {name} has no NaN values")

    # Check for infinite values in numeric columns
    numeric_data = df.select_dtypes(include=[np.number]).to_numpy()
    try:
        if np.isinf(numeric_data).any():
            logger.warning(f"🚨 {name} contains infinite values!")
    except Exception:
        pass

    logger.info(f"📊 {name} shape: {df.shape}")
    logger.info(
        f"📊 {name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )


# ============================================================================
# CALIBRATION AND THRESHOLD OPTIMIZATION UTILITIES
# ============================================================================


class FoldDecisionPolicy:
    """Stores calibrator and threshold for a specific fold in temporal CV with inflation adjustment.

    This class encapsulates the decision policy learned within a fold's training window:
    - Probability calibrator (Platt scaling or isotonic regression)
    - Optimal probability threshold for binary decisions
    - Fold-specific metadata including deflation info
    - Temporal bounds for leakage verification
    """

    def __init__(
        self,
        fold_id: Union[int, str],
        calibrator: Optional[Union[LogisticRegression, IsotonicRegression]] = None,
        threshold: float = 0.5,
        calibration_method: str = "none",
        optimization_metric: str = "f1",
    ):
        self.fold_id = fold_id
        self.calibrator = calibrator
        self.threshold = threshold
        self.calibration_method = calibration_method
        self.optimization_metric = optimization_metric
        self.fitted = False

        # Temporal and deflation metadata
        self.base_year = 2020
        self.deflation_series = "CPI-U SA"
        self.temporal_bounds = {}  # Store train/calib date ranges

    def fit(
        self,
        probabilities: NDArray[np.floating],
        labels: NDArray[np.integer],
        calibration_method: str = "platt",
        optimization_metric: str = "f1",
    ) -> None:
        """Fit calibrator and find optimal threshold on inner calibration set."""

        self.calibration_method = calibration_method
        self.optimization_metric = optimization_metric

        # 0) Orientation check (flip if ROC-AUC < 0.5)
        try:
            # Determine if input looks like probabilities ([0,1]) or scores (unbounded)
            pmin = float(np.min(probabilities))
            pmax = float(np.max(probabilities))
            auc = roc_auc_score(labels, probabilities)
            if auc < 0.5:
                if 0.0 - 1e-6 <= pmin and pmax <= 1.0 + 1e-6:
                    probabilities = 1.0 - probabilities
                else:
                    probabilities = -probabilities
        except Exception:
            pass

        # 1. Fit probability calibrator
        if calibration_method == "platt":
            self.calibrator = LogisticRegression(solver="lbfgs", random_state=42)
            self.calibrator.fit(probabilities.reshape(-1, 1), labels)
        elif calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(probabilities, labels)
        else:
            self.calibrator = None  # No calibration

        # 2. Apply calibration
        calibrated_probs = self.apply_calibration(probabilities)

        # 3. Find optimal threshold
        self.threshold = self._optimize_threshold(
            calibrated_probs, labels, optimization_metric
        )

        self.fitted = True
        logger.info(
            f"🎯 Fold {self.fold_id}: Fitted {calibration_method} calibrator, optimal threshold = {self.threshold:.3f}"
        )

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Export fold policy as metadata dictionary for storage."""
        return {
            "calibration": self.calibration_method,
            "theta": float(self.threshold),
            "optimization_metric": self.optimization_metric,
            "deflation_base_year": self.base_year,
            "deflation_series": self.deflation_series,
            "temporal_bounds": self.temporal_bounds,
            "fitted": self.fitted,
        }

    def apply_calibration(
        self, probabilities: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply calibration to raw probabilities."""
        if self.calibrator is None:
            return probabilities
        if self.calibration_method == "platt" and isinstance(
            self.calibrator, LogisticRegression
        ):
            return self.calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        if self.calibration_method == "isotonic" and isinstance(
            self.calibrator, IsotonicRegression
        ):
            return self.calibrator.predict(probabilities)
        return probabilities

    def predict(self, probabilities: NDArray[np.floating]) -> NDArray[np.int_]:
        """Apply calibration and threshold to get binary predictions."""
        calibrated_probs = self.apply_calibration(probabilities)
        return (calibrated_probs >= self.threshold).astype(int)

    def _optimize_threshold(
        self,
        probabilities: NDArray[np.floating],
        labels: NDArray[np.integer],
        metric: str = "f1",
    ) -> float:
        """Find optimal probability threshold on calibration set."""

        # Build threshold candidates from midpoints between sorted unique probabilities
        ps = np.unique(probabilities.astype(float))
        if ps.size <= 1:
            logger.warning(
                "Threshold optimization: only one unique probability; falling back to 0.5"
            )
            return 0.5

        candidates = (ps[:-1] + ps[1:]) / 2.0

        best_score = -1.0
        best_threshold = float(np.median(ps))
        best_tiebreak = (-1.0, -1.0)  # (balanced_accuracy, f1)

        n = labels.shape[0]
        for threshold in candidates:
            predictions = (probabilities >= threshold).astype(int)

            # Skip degenerate thresholds that predict a single class
            pos = int(predictions.sum())
            if pos == 0 or pos == n:
                continue

            try:
                if metric == "mcc":
                    score_primary = matthews_corrcoef(labels, predictions)
                    score_ba = balanced_accuracy_score(labels, predictions)
                    score_f1 = f1_score(labels, predictions, zero_division="warn")
                    # Primary: MCC; tie-break: Balanced Accuracy, then F1
                    if (score_primary > best_score) or (
                        np.isclose(score_primary, best_score)
                        and (score_ba, score_f1) > best_tiebreak
                    ):
                        best_score = score_primary
                        best_tiebreak = (score_ba, score_f1)
                        best_threshold = float(threshold)
                else:
                    if metric == "f1":
                        score = f1_score(labels, predictions, zero_division="warn")
                    elif metric == "balanced_accuracy":
                        score = balanced_accuracy_score(labels, predictions)
                    else:
                        raise ValueError(f"Unknown optimization metric: {metric}")

                    if score > best_score:
                        best_score = score
                        best_threshold = float(threshold)

            except Exception as e:
                logger.warning(
                    f"Error computing {metric} for threshold {threshold}: {e}"
                )
                continue

        logger.info(
            f"🎯 Threshold optimization: {metric}={best_score:.3f} at threshold={best_threshold:.3f}"
        )
        return best_threshold


# =============================================================================
# LABEL SHARPENING (OPTIONAL BOUNDARY NOISE REDUCTION)
# =============================================================================


def apply_label_sharpening(
    df: pd.DataFrame, fold_edges: Dict[str, float], epsilon: float = 0.03
) -> pd.DataFrame:
    """
    Apply label sharpening by removing data points too close to fold boundaries.

    Args:
        df: DataFrame with 'actual_amount_2020' column
        fold_edges: Dict with 'e1' and 'e2' boundary values
        epsilon: Relative margin (default 3%)

    Returns:
        Filtered DataFrame
    """
    if "actual_amount_2020" not in df.columns:
        logger.warning("actual_amount_2020 column missing - skipping label sharpening")
        return df

    e1, e2 = fold_edges.get("e1", 0), fold_edges.get("e2", float("inf"))

    # Calculate margins
    e1_margin = e1 * epsilon
    e2_margin = e2 * epsilon

    # Keep points that are sufficiently far from boundaries
    amounts = df["actual_amount_2020"].values
    keep_mask = (
        (amounts < e1 - e1_margin)  # Well below e1
        | (
            (amounts > e1 + e1_margin) & (amounts < e2 - e2_margin)
        )  # Well between e1 and e2
        | (amounts > e2 + e2_margin)  # Well above e2
    )

    sharpened_df = df[keep_mask].copy()
    removed_count = len(df) - len(sharpened_df)

    if removed_count > 0:
        logger.info(
            f"📐 Label sharpening: removed {removed_count}/{len(df)} boundary points (ε={epsilon:.1%})"
        )

    return sharpened_df


# Decide calibration method based on inner-calib size and positive rate
def choose_calibration_method(calib_labels: np.ndarray) -> str:
    """Return 'platt' for small/sparse sets, else 'isotonic'."""
    try:
        n = int(calib_labels.shape[0])
        pos = int(np.sum(calib_labels))
        pos_rate = (pos / max(n, 1)) if n > 0 else 0.0
    except Exception:
        n, pos, pos_rate = 0, 0, 0.0
    # Heuristic per spec
    if n < 4000 or pos < 1000 or pos_rate < 0.10:
        return "platt"
    return "isotonic"


# Legacy function - use create_temporal_split_with_dates for new code
def create_inner_temporal_split(
    df: pd.DataFrame, inner_train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create inner temporal split for calibration within a fold's training data.

    DEPRECATED: Use create_temporal_split_with_dates for proper date extraction.
    """
    logger.warning(
        "Using deprecated create_inner_temporal_split - switch to create_temporal_split_with_dates"
    )
    return create_temporal_split_with_dates(df, inner_train_ratio)


def get_binary_probabilities(
    model, X: np.ndarray, model_type: str = "polr"
) -> np.ndarray:
    """Extract binary probabilities from model predictions.

    For binary POLR/MLR, gets P(class=1) probabilities.
    For ordinal POLR, can extract specific class probabilities.
    """

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Handle 1D/2D robustly
            if isinstance(proba, np.ndarray):
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                if proba.ndim == 2 and proba.shape[1] == 1:
                    return proba[:, 0]
                if proba.ndim == 1:
                    return proba
            # Fallback
            return np.full((X.shape[0],), 0.5, dtype=float)
        # SVM path: use decision_function scores (to be calibrated later)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # Ensure 1D
            if isinstance(scores, np.ndarray) and scores.ndim > 1:
                scores = np.ravel(scores)
            # Convert SVM decision function scores to [0,1] probabilities via sigmoid
            # This prevents negative probability errors in calibration
            from scipy.special import expit

            return expit(scores)
        elif hasattr(model, "get_cumulative_probs"):
            # For POLR models with cumulative probabilities
            cum_probs = model.get_cumulative_probs(X)
            if cum_probs.shape[1] == 1:
                # Binary case: cumulative prob is P(high)
                return cum_probs[:, 0]
            else:
                # Multi-class: convert cumulative to class probabilities
                class_probs = np.diff(
                    np.column_stack(
                        [
                            np.zeros(cum_probs.shape[0]),
                            cum_probs,
                            np.ones(cum_probs.shape[0]),
                        ]
                    ),
                    axis=1,
                )
                return (
                    class_probs[:, -1]
                    if class_probs.shape[1] > 1
                    else class_probs[:, 0]
                )
        else:
            logger.warning("Model has no probability prediction method")
            return np.full((X.shape[0],), 0.5, dtype=float)

    except Exception as e:
        logger.warning(f"Error extracting probabilities: {e}")
        return np.full((X.shape[0],), 0.5, dtype=float)


# Constants for binary classification
EMIT_PREFIX = "polr"
BINARY_BUCKET_NAMES = ["low", "high"]  # Binary classification
BINARY_CLASS_MAPPING = {0: "low", 1: "high"}


# Pickle-safe transformation functions
def _binarize_transform(X):
    """Binarize transformation: convert values >0 to 1.0, <=0 to 0.0"""
    return (X > 0).astype(float)


def _log1p_transform(X):
    """Log1p transformation: apply log(1+x)"""
    return np.log1p(X)


def _winsor99(X):
    """Winsorize at 99th percentile"""
    hi = np.nanpercentile(X, 99.0, axis=0)
    return np.minimum(X, hi)


# =============================================================================
# TRAIN-ONLY WINSORIZATION HELPERS (p99.5 caps)
# =============================================================================

HEAVY_TAILED_FEATURES = [
    "feat_new2_neutral_run_mean",
    "feat_new_attribution_verb_density",
    "interpretable_lex_deception_count",  # optional per spec
]


def compute_winsor_caps(
    train_X_df: pd.DataFrame, features: List[str], quantile: float = 0.995
) -> Dict[str, float]:
    """Compute train-only winsorization caps for specified features.
    Returns mapping feature -> cap value (NaN caps are skipped by apply step).
    """
    present = [c for c in features if c in train_X_df.columns]
    if not present:
        return {}
    caps_series = train_X_df[present].quantile(quantile)
    caps: Dict[str, float] = {}
    for c in present:
        try:
            val = float(caps_series.get(c, np.nan))
        except Exception:
            val = float("nan")
        if not np.isnan(val):
            caps[c] = val
    return caps


def apply_winsor_caps(X_df: pd.DataFrame, caps: Dict[str, float]) -> pd.DataFrame:
    """Apply winsor caps (clip upper tail) to dataframe columns.
    Returns a new DataFrame with caps applied.
    """
    if not caps:
        return X_df
    Xc = X_df.copy()
    for c, cap in caps.items():
        if c in Xc.columns:
            Xc[c] = np.minimum(Xc[c].astype(float), float(cap))
    return Xc


@dataclass
class BinaryPOLARConfig:
    """Configuration for binary POLAR pipeline."""

    # Data paths
    kfold_dir: str = "data/final_stratified_kfold_splits_binary_quote_balanced"
    output_dir: str = "runs/binary_polr_experiment"

    # Model parameters
    model_type: str = "polr"  # "polr" or "mlr"
    hyperparameter_grid: Optional[Dict[str, List[Any]]] = None
    scoring_priority: Optional[List[str]] = None

    # Training parameters
    n_inner_cv: int = 4  # 4-fold CV (fold_0, fold_1, fold_2, fold_3)
    calibration_method: str = "isotonic_cumulative"
    calibration_split: float = 0.15

    # Binary classification parameters
    continuous_target_field: str = "final_judgement_real"  # For reference only
    binary_target_field: str = "outcome_bin"  # Binary target (0/1)
    seed: int = 42
    n_jobs: int = -1

    # Inflation adjustment parameters
    deflation_base_year: int = 2020  # Base year for real dollars
    deflation_series: str = "CPI-U SA"  # CPI series name
    use_cpi_deflation: bool = True  # Enable/disable inflation adjustment

    # Temporal DEV policy parameters
    dev_tail_frac: float = 0.20
    min_dev_cases: int = 3
    min_dev_quotes: int = 150
    require_all_classes: bool = False  # Accept ≥2 classes by default
    embargo_days: int = 90
    safe_qwk: bool = True
    min_cal_n: int = 500  # Minimum samples for direct isotonic
    iso_bins: int = 30  # Quantile bins for small-sample isotonic
    max_categories: int = 50  # Top-K + __OTHER__ threshold

    def __post_init__(self):
        if self.hyperparameter_grid is None:
            if self.model_type == "mlr":
                self.hyperparameter_grid = {
                    "C": [0.01, 1, 100],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                    "tol": [1e-4],
                    "class_weight": ["balanced", None],
                }
            else:
                self.hyperparameter_grid = {
                    "C": [0.01, 1, 100],
                    "solver": ["lbfgs"],
                    "max_iter": [200],
                }

        if self.scoring_priority is None:
            # Binary classification metrics - prioritize MCC per spec
            self.scoring_priority = [
                "mcc",
                "avg_precision",
                "roc_auc",
                "balanced_accuracy",
                "f1",
                "accuracy",
                "precision",
                "recall",
            ]


class BinaryProgressReporter:
    """Progress reporter for binary POLAR training."""

    def __init__(self, n_folds: int):
        self.n_folds = n_folds
        self.current_fold = 0

    def report(self, stage: str, message: str):
        """Report progress with binary-specific formatting."""
        logger.info(
            f"[BINARY-{stage}] Fold {self.current_fold}/{self.n_folds}: {message}"
        )

    def next_fold(self):
        """Move to next fold."""
        self.current_fold += 1


def load_binary_fold_metadata(kfold_dir: Path) -> Dict[str, Any]:
    """Load binary fold metadata with validation."""
    metadata_path = kfold_dir / "per_fold_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Validate binary structure
    if metadata.get("binning", {}).get("classification_type") != "binary":
        logger.warning(
            "Metadata indicates non-binary classification, but using binary pipeline"
        )

    # Ensure we have 5 folds (fold_0, fold_1, fold_2, fold_3, fold_4)
    fold_edges = metadata.get("binning", {}).get("fold_edges", {})
    expected_folds = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

    missing_folds = [f for f in expected_folds if f not in fold_edges]
    if missing_folds:
        raise ValueError(f"Missing fold edges for: {missing_folds}")

    logger.info(f"Loaded binary metadata for {len(fold_edges)} folds")
    return metadata


def prepare_binary_features(
    df: pd.DataFrame,
    preprocessor: Optional[Any] = None,
    fit_preprocessor: bool = False,
    max_categories: int = 50,
) -> Tuple[pd.DataFrame, Optional[Any]]:
    """Prepare binary features - RAW FEATURES ONLY.

    These 10 features are already properly encoded, normalized, and have zero nulls.
    No transformations needed - use them directly for maximum interpretability.
    """
    start_time = time.time()

    # Validate columns
    validation_result = validate_binary_columns(df.columns.tolist())

    if not validation_result["valid"]:
        raise ValueError("No valid binary features found")

    # Get only the 10 allowed features
    feature_columns = get_binary_feature_columns(df.columns.tolist())

    if len(feature_columns) != 10:
        raise ValueError(
            f"Expected exactly 10 binary features, got {len(feature_columns)}"
        )

    logger.info(
        f"✨ Using {len(feature_columns)} RAW binary features (no transformations): {feature_columns}"
    )

    # Extract raw features
    X = df[feature_columns].copy()
    logger.info(f"📊 Raw features shape: {X.shape}")

    # Quick data quality check (should be clean)
    null_count = X.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"⚠️ Found {null_count} unexpected nulls - filling with 0.0")
        X = X.fillna(0.0)
    else:
        logger.info("✅ Perfect data quality - zero nulls as expected")

    # Ensure proper data types (should already be float)
    X = X.astype(float)

    # Note: Train-only winsorization will be applied in training routines (inner/outer) where needed

    # Note: RobustScaler will be fit in training routines (inner/outer) where needed

    prep_time = time.time() - start_time
    logger.info(
        f"⚡ Raw feature extraction completed in {prep_time:.3f}s (no preprocessing overhead)"
    )

    # Return raw features and None for preprocessor since we don't need one
    return X, None


def compute_binary_scores(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute comprehensive binary classification scores."""

    scores = {}

    # Basic classification metrics
    scores["accuracy"] = accuracy_score(y_true, y_pred)
    scores["precision"] = precision_score(y_true, y_pred, average="binary")
    scores["recall"] = recall_score(y_true, y_pred, average="binary")
    scores["f1"] = f1_score(y_true, y_pred, average="binary")
    # Robust, skew-insensitive metrics
    scores["mcc"] = matthews_corrcoef(y_true, y_pred)
    scores["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Probabilistic metrics (if probabilities provided)
    if y_proba is not None:
        if y_proba.shape[1] >= 2:
            # Use positive class probabilities
            pos_proba = y_proba[:, 1]
            scores["roc_auc"] = roc_auc_score(y_true, pos_proba)
            scores["avg_precision"] = average_precision_score(y_true, pos_proba)
            scores["brier_score"] = brier_score_loss(y_true, pos_proba)
        else:
            # Single probability column
            scores["roc_auc"] = roc_auc_score(y_true, y_proba[:, 0])
            scores["avg_precision"] = average_precision_score(y_true, y_proba[:, 0])
            scores["brier_score"] = brier_score_loss(y_true, y_proba[:, 0])

    return scores


def binary_hyperparameter_search_with_calibration(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    weights_dev: np.ndarray,
    param_grid: Dict[str, List[Any]],
    scoring_priority: List[str],
    reporter: BinaryProgressReporter,
    fold: int,
    dev_metadata: Dict[str, Any],
    model_type: str = "polr",
    calibration_method: str = "platt",
    optimization_metric: str = "f1",
) -> Tuple[Dict[str, Any], Dict[str, Any], FoldDecisionPolicy]:
    """Binary hyperparameter search with calibrated decision boundaries (temporal CV).

    Implements proper fold-specific threshold optimization using:
    1. Inner temporal split within training data
    2. Probability calibration (Platt or isotonic)
    3. Threshold optimization on inner calibration set
    4. Leakage-safe evaluation on outer validation set
    """

    reporter.report(
        "HYPERPARAM",
        f"Starting calibrated hyperparameter search with {len(list(ParameterGrid(param_grid)))} configurations",
    )

    # Load CPI data for inflation adjustment (if needed)
    try:
        cpi_df = load_cpi_data()
    except Exception as e:
        logger.warning(
            f"Could not load CPI data: {e}, proceeding without inflation adjustment"
        )
        cpi_df = None

    # 1. Create inner temporal split for calibration using proper date extraction
    inner_train_df, inner_calib_df = create_temporal_split_with_dates(
        train_df, inner_train_ratio=0.8, cpi_df=cpi_df
    )

    # Prepare features for inner split
    X_inner_train, _ = prepare_binary_features(inner_train_df)
    X_inner_calib, _ = prepare_binary_features(inner_calib_df)

    # Extract labels and weights for inner split
    y_inner_train = inner_train_df["outcome_bin"].values
    y_inner_calib = inner_calib_df["outcome_bin"].values
    weights_inner_train = inner_train_df["sample_weight"].values
    weights_inner_calib = inner_calib_df["sample_weight"].values

    # Train-only winsorization caps on inner-train, apply to calib and dev (leakage-safe)
    winsor_caps = compute_winsor_caps(
        X_inner_train, HEAVY_TAILED_FEATURES, quantile=0.995
    )
    if winsor_caps:
        X_inner_train = apply_winsor_caps(X_inner_train, winsor_caps)
        X_inner_calib = apply_winsor_caps(X_inner_calib, winsor_caps)

    # Log class distributions to debug calibration failures
    train_classes = np.bincount(y_inner_train)
    calib_classes = np.bincount(y_inner_calib)

    logger.info(
        f"📅 Inner split: {len(inner_train_df)} inner-train, {len(inner_calib_df)} inner-calib"
    )
    logger.info(f"🎯 Inner-train class distribution: {train_classes}")
    logger.info(f"🎯 Inner-calib class distribution: {calib_classes}")

    # Check for class imbalance in calibration set and fix if needed (temporal-safe)
    calibration_available = True
    if len(calib_classes) < 2 or calib_classes.min() == 0:
        logger.warning(
            f"⚠️  Inner-calib has insufficient class diversity: {calib_classes}"
        )
        logger.info(
            "🔧 Adjusting temporal cutoff to ensure class diversity in inner-calib (temporal-safe)"
        )

        # Reconstruct temporal ordering over the full outer-train and pick a later cutoff
        df_with_dates = train_df.copy()
        temporal_keys: List[int] = []
        for _, row in train_df.iterrows():
            case_id_val = row.get("case_id", "")
            case_id = str(case_id_val) if case_id_val is not None else ""
            date_tuple = extract_case_date_from_id(case_id)
            if date_tuple:
                year, month = date_tuple
                temporal_key = year * 100 + month
            else:
                temporal_key = hash(case_id) % 1000000
            temporal_keys.append(int(temporal_key))
        df_with_dates["temporal_key"] = temporal_keys
        df_sorted = df_with_dates.sort_values(["temporal_key", "case_id"]).copy()

        n_total = len(df_sorted)
        # Search cutoffs from 70% .. 95% to find a tail (inner-calib) with both classes
        candidate_fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        found = False
        for frac in candidate_fracs:
            cutoff = int(n_total * frac)
            inner_train_df = df_sorted.iloc[:cutoff].copy()
            inner_calib_df = df_sorted.iloc[cutoff:].copy()
            if len(inner_calib_df) < 50:  # require a minimal calibration size
                continue
            y_ct = np.bincount(inner_calib_df["outcome_bin"].values)
            if len(y_ct) >= 2 and y_ct.min() > 0:
                found = True
                logger.info(
                    f"✅ Temporal cutoff adjusted to {frac:.2f} → calib size {len(inner_calib_df)} with classes {y_ct}"
                )
                break

        if not found:
            # As last resort, take a fixed-size temporal tail that contains at least one positive and one negative
            logger.warning(
                "⚠️  Could not ensure class diversity via fractional cutoff; using fixed-size temporal tail sweep"
            )
            min_tail = min(200, max(50, n_total // 20))
            for tail in [
                min_tail,
                max(min_tail, n_total // 10),
                max(min_tail, n_total // 8),
            ]:
                inner_train_df = df_sorted.iloc[:-tail].copy()
                inner_calib_df = df_sorted.iloc[-tail:].copy()
                y_ct = np.bincount(inner_calib_df["outcome_bin"].values)
                if len(y_ct) >= 2 and y_ct.min() > 0:
                    found = True
                    logger.info(f"✅ Temporal tail size {tail} → calib classes {y_ct}")
                    break

        if not found:
            # Temporally safe fallback: proceed WITHOUT calibration/threshold tuning
            # to avoid leakage. We'll use direct model.predict on the dev set.
            calibration_available = False
            logger.error(
                "❌ Unable to create temporally valid inner calibration set with both classes — proceeding without calibration (temporal-safe)"
            )

        # Re-extract features and labels
        X_inner_train, _ = prepare_binary_features(inner_train_df)
        X_inner_calib, _ = prepare_binary_features(inner_calib_df)
        y_inner_train = inner_train_df["outcome_bin"].values
        y_inner_calib = inner_calib_df["outcome_bin"].values
        weights_inner_train = inner_train_df["sample_weight"].values
        weights_inner_calib = inner_calib_df["sample_weight"].values

        # Log new distributions
        train_classes = np.bincount(y_inner_train)
        calib_classes = np.bincount(y_inner_calib)
        logger.info(
            f"🔧 Temporal-adjusted split: {len(inner_train_df)} inner-train, {len(inner_calib_df)} inner-calib"
        )
        logger.info(f"✅ Fixed inner-train class distribution: {train_classes}")
        logger.info(f"✅ Fixed inner-calib class distribution: {calib_classes}")

    best_params = None
    best_score = -np.inf
    best_scores = None
    best_fold_policy = None
    all_results = []

    # Convert to numpy for model fitting
    X_inner_train_np = X_inner_train.values
    X_inner_calib_np = X_inner_calib.values
    X_dev_np = X_dev.values if hasattr(X_dev, "values") else X_dev

    # Robust scaling (fit on inner-train only) for solver conditioning in EN/SVM
    scaler = RobustScaler()
    try:
        scaler.fit(X_inner_train_np)
        X_inner_train_np = scaler.transform(X_inner_train_np)
        X_inner_calib_np = scaler.transform(X_inner_calib_np)
        X_dev_np = scaler.transform(X_dev_np)
    except Exception:
        pass

    for param_set in ParameterGrid(param_grid):
        try:
            # 2. Train model on inner-train only
            if model_type == "mlr":
                model = MLR(**param_set)
            elif model_type == "lr_l2":
                model = LR_L2(**param_set)
            elif model_type == "lr_l1":
                model = LR_L1(**param_set)
            elif model_type == "lr_elasticnet":
                model = LR_ElasticNet(**param_set)
            elif model_type == "svm_linear":
                model = SVM_Linear(**param_set)
            else:
                model = POLR(**param_set)

            model.fit(
                X_inner_train_np, y_inner_train, sample_weight=weights_inner_train
            )

            if calibration_available:
                # 3. Get probabilities on inner-calib set
                inner_calib_probs = get_binary_probabilities(
                    model, X_inner_calib_np, model_type
                )
                if inner_calib_probs is None:
                    logger.warning(
                        f"Could not extract probabilities for param set: {param_set}"
                    )
                    continue

                # 4. Fit fold decision policy (calibrator + threshold) on inner-calib
                fold_policy = FoldDecisionPolicy(fold_id=fold)
                selected_calibration = choose_calibration_method(y_inner_calib)
                fold_policy.fit(
                    probabilities=inner_calib_probs,
                    labels=y_inner_calib,
                    calibration_method=selected_calibration,
                    optimization_metric="mcc",
                )

                # 5. Evaluate on outer dev set using calibrated predictions
                dev_probs = get_binary_probabilities(model, X_dev_np, model_type)
                if dev_probs is None:
                    logger.warning(
                        f"Could not extract dev probabilities for param set: {param_set}"
                    )
                    continue

                # Apply calibration and threshold for predictions
                y_pred_calibrated = fold_policy.predict(dev_probs)

                # Get calibrated probabilities for probabilistic metrics
                dev_probs_calibrated = fold_policy.apply_calibration(dev_probs)
                y_proba_calibrated = np.column_stack(
                    [1 - dev_probs_calibrated, dev_probs_calibrated]
                )
            else:
                # No calibration fallback: use direct model predictions on dev (temporal-safe)
                fold_policy = FoldDecisionPolicy(fold_id=fold)
                fold_policy.calibration_method = "none"
                fold_policy.threshold = 0.5
                fold_policy.fitted = True
                y_pred_calibrated = model.predict(X_dev_np)
                y_proba_calibrated = None

            # 6. Compute scores using calibrated predictions
            scores = compute_binary_scores(y_dev, y_pred_calibrated, y_proba_calibrated)

            # Add calibration-specific metrics
            scores["calibrated_threshold"] = float(fold_policy.threshold)
            scores["calibration_method"] = (
                fold_policy.calibration_method
                if hasattr(fold_policy, "calibration_method")
                else calibration_method
            )
            scores["optimization_metric"] = optimization_metric

            # Determine primary score (first in priority list)
            primary_score = scores.get(scoring_priority[0], 0.0)

            result = {
                "params": param_set,
                "scores": scores,
                "primary_score": primary_score,
                # store metadata only for serialization
                "fold_policy": fold_policy.to_metadata_dict(),
            }
            all_results.append(result)

            # Track best based on PRIMARY metric (threshold-free)
            if scoring_priority[0] in [
                "roc_auc",
                "avg_precision",
            ]:  # Threshold-free metrics
                comparison_score = primary_score
            else:  # Threshold-dependent metrics (F1, etc.)
                comparison_score = primary_score

            if comparison_score > best_score:
                best_score = comparison_score
                best_params = param_set
                best_scores = scores
                best_fold_policy = fold_policy

            logger.info(
                f"✅ Params {param_set}: {scoring_priority[0]}={primary_score:.3f}, threshold={fold_policy.threshold:.3f}"
            )

        except Exception as e:
            logger.warning(
                f"Hyperparameter configuration failed: {param_set}, error: {e}"
            )
            continue

    if best_params is None:
        raise ValueError("No hyperparameter configuration succeeded")

    reporter.report(
        "HYPERPARAM",
        f"Best {scoring_priority[0]}: {best_score:.4f} (calibrated threshold: {best_fold_policy.threshold:.3f})",
    )
    logger.info(f"🎯 Best parameters: {best_params}")
    logger.info(f"📊 Best scores: {best_scores}")
    logger.info(
        f"🔬 Calibration method: {calibration_method}, optimization metric: {optimization_metric}"
    )

    hyperparameter_results = {
        "best_params": best_params,
        "best_scores": best_scores,
        "all_results": all_results,
        "dev_metadata": dev_metadata,
        # keep only metadata here, not the object
        "fold_policy_metadata": (
            best_fold_policy.to_metadata_dict() if best_fold_policy else None
        ),
        "calibration_method": (
            best_fold_policy.calibration_method
            if best_fold_policy
            else calibration_method
        ),
        "optimization_metric": optimization_metric,
        "cpi_deflation_enabled": cpi_df is not None,
        "winsor_caps": winsor_caps,
    }

    return best_params, hyperparameter_results, best_fold_policy


def train_binary_polar_cv(config: BinaryPOLARConfig) -> Dict[str, Any]:
    """Run complete binary POLAR training with cross-validation protocol.

    Implements the full paper-quality protocol for binary classification including:
    - Binary column governance (exactly 10 features)
    - Per-fold binary cutpoints
    - Alpha-normalized combined weights
    - Hyperparameter search on 4 folds (0,1,2,3)
    - Final training on fold_4
    - Comprehensive binary evaluation

    Returns:
        Dictionary with all results, models, and evaluation metrics
    """
    # Setup
    kfold_dir = Path(config.kfold_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = load_binary_fold_metadata(kfold_dir)

    # Initialize results storage
    results = {
        "config": asdict(config),
        "metadata": metadata,
        "folds": {},
        "oof_predictions": [],
        "final_model": None,
        "timestamp": datetime.now().isoformat(),
        "model_type": "binary_" + config.model_type,
    }

    # Count folds - use folds 0,1,2,3 for hyperparameter search
    fold_dirs = sorted(
        [d for d in kfold_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]
    )
    cv_fold_dirs = [
        d for d in fold_dirs if d.name in ["fold_0", "fold_1", "fold_2", "fold_3"]
    ]
    n_cv_folds = len(cv_fold_dirs)

    reporter = BinaryProgressReporter(n_cv_folds)

    logger.info(
        f"Starting Binary POLR CV training with {n_cv_folds} folds (0,1,2,3) for hyperparameter search"
    )
    logger.info(f"Final training will use fold_4")
    logger.info(f"Output directory: {output_dir}")

    # Process each CV fold (0,1,2,3 only)
    cv_results = []

    for fold_dir in cv_fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[1])
        reporter.next_fold()

        logger.info(f"Processing fold {fold_idx}")

        # Load fold data
        train_path = fold_dir / "train.jsonl"
        val_path = fold_dir / "val.jsonl"
        test_path = fold_dir / "test.jsonl"

        if not all(p.exists() for p in [train_path, val_path, test_path]):
            logger.error(f"Missing data files in fold {fold_idx}")
            continue

        # Read data with fast JSON loading (3-5x faster than pd.read_json)
        train_df, val_df, test_df = fast_load_fold_data(fold_dir)

        # Validate binary structure
        for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if not validate_binary_target(df.columns.tolist()):
                raise ValueError(
                    f"Missing binary target 'outcome_bin' in {df_name} data for fold {fold_idx}"
                )

            weight_validation = validate_binary_weights(df.columns.tolist())
            missing_weights = [k for k, v in weight_validation.items() if not v]
            if missing_weights:
                logger.warning(
                    f"Missing weight columns {missing_weights} in {df_name} data for fold {fold_idx}"
                )

        # Extract targets and weights
        y_train = train_df["outcome_bin"].values
        y_val = val_df["outcome_bin"].values
        y_test = test_df["outcome_bin"].values

        weights_train = train_df["sample_weight"].values
        weights_val = val_df["sample_weight"].values
        weights_test = test_df["sample_weight"].values

        # Get fold-specific metadata
        fold_metadata = metadata["weights"].get(f"fold_{fold_idx}", {})
        fold_class_weights = fold_metadata.get("class_weights", {})

        logger.info(
            f"Fold {fold_idx} class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}"
        )
        logger.info(f"Using fold class weights: {fold_class_weights}")

        # Prepare raw features (no preprocessing needed)
        reporter.report("FEATURES", "Extracting raw binary features")
        X_train, _ = prepare_binary_features(train_df)
        X_val, _ = prepare_binary_features(val_df)
        X_test, _ = prepare_binary_features(test_df)

        reporter.report(
            "FEATURES", f"Binary feature dimensions: {X_train.shape[1]} features"
        )

        # Hyperparameter search on VAL
        dev_metadata = {
            "fold": fold_idx,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "class_weights": fold_class_weights,
        }
        # Apply train-only winsor caps and RobustScaler in a leakage-safe way
        X_train_raw = X_train
        X_val_raw = X_val
        winsor_caps_outer = compute_winsor_caps(
            X_train_raw, HEAVY_TAILED_FEATURES, quantile=0.995
        )
        if winsor_caps_outer:
            X_train_raw = apply_winsor_caps(X_train_raw, winsor_caps_outer)
            X_val_raw = apply_winsor_caps(X_val_raw, winsor_caps_outer)

        scaler_outer = RobustScaler()
        try:
            scaler_outer.fit(X_train_raw.values)
            X_train_scaled = scaler_outer.transform(X_train_raw.values)
            X_val_scaled = scaler_outer.transform(X_val_raw.values)
        except Exception:
            X_train_scaled = X_train_raw.values
            X_val_scaled = X_val_raw.values

        # Replace X_train/X_val fed to HP search with scaled versions
        X_train = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns)
        X_val = pd.DataFrame(X_val_scaled, columns=X_val_raw.columns)

        best_params, hp_results, fold_policy = (
            binary_hyperparameter_search_with_calibration(
                pd.concat(
                    [X_train, train_df[["outcome_bin", "sample_weight", "case_id"]]],
                    axis=1,
                ),  # restricted fields
                y_train,
                weights_train,
                X_val,
                y_val,
                weights_val,
                config.hyperparameter_grid,
                config.scoring_priority,
                reporter,
                fold_idx,
                dev_metadata,
                model_type=config.model_type,
                calibration_method="platt",  # initial; function will choose based on data
                optimization_metric="mcc",  # per spec
            )
        )

        # Train final model for this fold with best params
        if config.model_type == "mlr":
            model_name = "MLR"
            model = MLR(**best_params)
        elif config.model_type == "lr_l2":
            model_name = "LR_L2"
            model = LR_L2(**best_params)
        elif config.model_type == "lr_l1":
            model_name = "LR_L1"
            model = LR_L1(**best_params)
        elif config.model_type == "lr_elasticnet":
            model_name = "LR_ElasticNet"
            model = LR_ElasticNet(**best_params)
        elif config.model_type == "svm_linear":
            model_name = "SVM_Linear"
            model = SVM_Linear(**best_params)
        else:
            model_name = "POLR"
            model = POLR(**best_params)

        reporter.report(
            "TRAINING", f"Training binary {model_name} with best hyperparameters"
        )

        model.fit(X_train.values, y_train, sample_weight=weights_train)

        # Evaluate on test set using calibrated predictions
        reporter.report(
            "EVALUATION",
            f"Evaluating with calibrated threshold: {fold_policy.threshold:.3f}",
        )

        # Get raw probabilities
        test_probs = get_binary_probabilities(model, X_test.values, config.model_type)
        if test_probs is None:
            logger.warning(
                "Could not extract test probabilities, falling back to direct prediction"
            )
            y_pred_test = model.predict(X_test.values)
            y_proba_test = None
        else:
            # Apply calibrated prediction using fold policy
            y_pred_test = fold_policy.predict(test_probs)

            # Get calibrated probabilities for evaluation metrics
            test_probs_calibrated = fold_policy.apply_calibration(test_probs)
            y_proba_test = np.column_stack(
                [1 - test_probs_calibrated, test_probs_calibrated]
            )

        test_scores = compute_binary_scores(y_test, y_pred_test, y_proba_test)

        # Store fold results
        fold_results = {
            "fold": fold_idx,
            "best_params": best_params,
            "hyperparameter_results": hp_results,
            "test_scores": test_scores,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "class_distribution": {
                "train": np.bincount(y_train).tolist(),
                "val": np.bincount(y_val).tolist(),
                "test": np.bincount(y_test).tolist(),
            },
        }

        results["folds"][f"fold_{fold_idx}"] = fold_results
        cv_results.append(fold_results)

        logger.info(f"Fold {fold_idx} test scores: {test_scores}")

    # Aggregate CV results
    if cv_results:
        cv_summary = {}
        for metric in config.scoring_priority:
            scores = [fold["test_scores"].get(metric, 0.0) for fold in cv_results]
            cv_summary[f"{metric}_mean"] = np.mean(scores)
            cv_summary[f"{metric}_std"] = np.std(scores)

        # Aggregate fold policies for temporal/deflation metadata
        fold_policies = {}
        threshold_values = []

        for fold_result in cv_results:
            hp_results = fold_result.get("hyperparameter_results", {})
            fold_policy_meta = hp_results.get("fold_policy_metadata")
            if fold_policy_meta:
                fold_id = fold_result["fold"]
                fold_policies[f"fold_{fold_id}"] = fold_policy_meta
                threshold_values.append(fold_policy_meta.get("theta", 0.5))

        if threshold_values:
            cv_summary["threshold_mean"] = np.mean(threshold_values)
            cv_summary["threshold_std"] = np.std(threshold_values)
            cv_summary["threshold_range"] = [
                min(threshold_values),
                max(threshold_values),
            ]

        results["cv_summary"] = cv_summary
        results["fold_policies"] = fold_policies
        results["deflation"] = {
            "series": config.deflation_series,
            "base_year": config.deflation_base_year,
            "enabled": config.use_cpi_deflation,
        }

        logger.info(f"CV Summary: {cv_summary}")
        if threshold_values:
            logger.info(
                f"Threshold distribution: {cv_summary['threshold_mean']:.3f} ± {cv_summary['threshold_std']:.3f}"
            )

    # Save CV results
    cv_results_path = output_dir / "cv_results.json"
    with open(cv_results_path, "w") as f:
        json.dump(pyify(results), f, indent=2)

    logger.info(f"Binary POLR CV training completed. Results saved to {output_dir}")

    return results


def train_final_binary_polar_model(
    config: BinaryPOLARConfig, cv_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Train final binary model on fold_4 and evaluate on OOF test set."""

    kfold_dir = Path(config.kfold_dir)
    output_dir = Path(config.output_dir)

    logger.info("Starting final binary POLR model training on fold_4")

    # Load fold_4 data (final training fold)
    fold_4_dir = kfold_dir / "fold_4"
    if not fold_4_dir.exists():
        raise FileNotFoundError(f"fold_4 directory not found: {fold_4_dir}")

    train_path = fold_4_dir / "train.jsonl"
    dev_path = fold_4_dir / "dev.jsonl"

    if not all(p.exists() for p in [train_path, dev_path]):
        raise FileNotFoundError(f"Missing train/dev files in fold_4")

    # Load OOF test data
    oof_test_dir = kfold_dir / "oof_test"
    oof_test_path = oof_test_dir / "test.jsonl"

    if not oof_test_path.exists():
        raise FileNotFoundError(f"OOF test file not found: {oof_test_path}")

    # Read data with fast JSON loading (3-5x faster than pd.read_json)
    logger.info("🚀 Fast loading final training data...")
    train_data = fast_load_jsonl(train_path)
    dev_data = fast_load_jsonl(dev_path)
    oof_test_data = fast_load_jsonl(oof_test_path)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    oof_test_df = pd.DataFrame(oof_test_data)

    logger.info(
        f"Final training data: {len(train_df)} train, {len(dev_df)} dev, {len(oof_test_df)} OOF test"
    )

    # Extract targets and weights
    y_train = train_df["outcome_bin"].values
    y_dev = dev_df["outcome_bin"].values
    y_oof = oof_test_df["outcome_bin"].values

    weights_train = train_df["sample_weight"].values
    weights_dev = dev_df["sample_weight"].values
    weights_oof = oof_test_df["sample_weight"].values

    # Prepare raw features (no preprocessing needed)
    logger.info("⚡ Extracting raw features for final training...")

    X_train, _ = prepare_binary_features(train_df)
    X_dev, _ = prepare_binary_features(dev_df)
    X_oof, _ = prepare_binary_features(oof_test_df)

    logger.info(
        f"📊 Raw feature extraction complete - Train: {X_train.shape}, Dev: {X_dev.shape}, OOF: {X_oof.shape}"
    )

    # Helper: default params per model
    def _default_params(model_type: str) -> Dict[str, Any]:
        if model_type == "mlr":
            return {"C": 1.0, "solver": "lbfgs", "max_iter": 200}
        if model_type == "lr_l2":
            return {"C": 1.0, "solver": "lbfgs", "max_iter": 200}
        if model_type == "lr_l1":
            return {"C": 1.0, "solver": "liblinear", "max_iter": 200}
        if model_type == "lr_elasticnet":
            return {"C": 1.0, "l1_ratio": 0.5, "solver": "saga", "max_iter": 500}
        # polr default
        return {"C": 1.0, "solver": "lbfgs", "max_iter": 200}

    # Get best hyperparameters from CV results if available
    best_params: Dict[str, Any]
    folds_dict = cv_results.get("folds", {}) if isinstance(cv_results, dict) else {}
    if folds_dict:
        best_fold_results = None
        best_primary_score = -np.inf
        for _, fold_results in folds_dict.items():
            primary_metric = (config.scoring_priority or ["accuracy"])[0]
            primary_score = fold_results.get("test_scores", {}).get(primary_metric, 0.0)
            if primary_score > best_primary_score:
                best_primary_score = primary_score
                best_fold_results = fold_results
        if best_fold_results is not None:
            best_params = best_fold_results.get(
                "best_params", _default_params(config.model_type)
            )
            logger.info(f"Using best hyperparameters from CV: {best_params}")
        else:
            best_params = _default_params(config.model_type)
            logger.warning(
                "No valid fold results found in CV; falling back to default hyperparameters for final training"
            )
    else:
        best_params = _default_params(config.model_type)
        logger.warning(
            "CV results not provided; using default hyperparameters for final training"
        )

    # For final training, create inner split for calibration within final training data
    logger.info("🎯 Creating inner temporal split for final model calibration...")

    # Load CPI data for proper temporal processing
    try:
        cpi_df = load_cpi_data()
    except Exception as e:
        logger.warning(
            f"Could not load CPI data: {e}, proceeding without inflation adjustment"
        )
        cpi_df = None

    final_combined_df = pd.concat([train_df, dev_df])
    final_inner_train_df, final_inner_calib_df = create_temporal_split_with_dates(
        final_combined_df, inner_train_ratio=0.8, cpi_df=cpi_df
    )

    # Prepare features for final inner split
    X_final_inner_train, _ = prepare_binary_features(final_inner_train_df)
    X_final_inner_calib, _ = prepare_binary_features(final_inner_calib_df)

    # Extract labels and weights for final inner split
    y_final_inner_train = final_inner_train_df["outcome_bin"].values
    y_final_inner_calib = final_inner_calib_df["outcome_bin"].values
    weights_final_inner_train = final_inner_train_df["sample_weight"].values
    weights_final_inner_calib = final_inner_calib_df["sample_weight"].values

    logger.info(
        f"📊 Final inner split: {len(final_inner_train_df)} inner-train, {len(final_inner_calib_df)} inner-calib"
    )

    # Train final model on inner training data
    if config.model_type == "mlr":
        final_model = MLR(**best_params)
    elif config.model_type == "lr_l2":
        final_model = LR_L2(**best_params)
    elif config.model_type == "lr_l1":
        final_model = LR_L1(**best_params)
    elif config.model_type == "lr_elasticnet":
        final_model = LR_ElasticNet(**best_params)
    elif config.model_type == "svm_linear":
        final_model = SVM_Linear(**best_params)
    else:
        final_model = POLR(**best_params)

    # Apply RobustScaler fit on final inner-train only
    try:
        final_scaler = RobustScaler()
        final_scaler.fit(X_final_inner_train.values)
        X_fit = final_scaler.transform(X_final_inner_train.values)
        X_cal = final_scaler.transform(X_final_inner_calib.values)
        X_oof_scaled = final_scaler.transform(X_oof.values)
    except Exception:
        X_fit = X_final_inner_train.values
        X_cal = X_final_inner_calib.values
        X_oof_scaled = X_oof.values

    final_model.fit(X_fit, y_final_inner_train, sample_weight=weights_final_inner_train)

    # Create final decision policy using inner calibration set
    # If calibration set is degenerate (single class), use temporal-safe fallback: no calibration
    try:
        y_final_classes = np.bincount(y_final_inner_calib)
    except Exception:
        y_final_classes = np.array([])

    if y_final_classes.size < 2 or (
        y_final_classes.size >= 2 and y_final_classes.min() == 0
    ):
        logger.error(
            "❌ Final inner calibration lacks both classes — proceeding without calibration (temporal-safe)"
        )
        final_policy = FoldDecisionPolicy(fold_id="final")
        final_policy.calibration_method = "none"
        final_policy.threshold = 0.5
        final_policy.fitted = True
    else:
        final_inner_calib_probs = get_binary_probabilities(
            final_model, X_cal, config.model_type
        )
        if final_inner_calib_probs is None:
            raise ValueError(
                "Could not extract probabilities for final model calibration"
            )
        final_policy = FoldDecisionPolicy(fold_id="final")
        final_policy.fit(
            probabilities=final_inner_calib_probs,
            labels=y_final_inner_calib,
            calibration_method="platt",
            optimization_metric="f1",
        )

    logger.info(f"🎯 Final model calibrated threshold: {final_policy.threshold:.3f}")

    # Evaluate on OOF test set using calibrated predictions
    oof_probs = get_binary_probabilities(final_model, X_oof_scaled, config.model_type)
    if oof_probs is None:
        logger.warning(
            "Could not extract OOF probabilities, falling back to direct prediction"
        )
        y_pred_oof = final_model.predict(X_oof.values)
        y_proba_oof = None
    else:
        # Apply calibrated prediction using final policy
        y_pred_oof = final_policy.predict(oof_probs)

        # Get calibrated probabilities for evaluation metrics
        oof_probs_calibrated = final_policy.apply_calibration(oof_probs)
        y_proba_oof = np.column_stack([1 - oof_probs_calibrated, oof_probs_calibrated])

    oof_scores = compute_binary_scores(y_oof, y_pred_oof, y_proba_oof)

    # Create OOF predictions with polr_ prefix
    oof_predictions = []
    for i, (idx, row) in enumerate(oof_test_df.iterrows()):
        pred_dict = {
            "case_id": row["case_id"],
            "fold": "oof_test",
            "split": "test",
            "model": "binary_" + config.model_type,
            # Binary predictions with polr_ prefix
            f"{EMIT_PREFIX}_pred_class": int(y_pred_oof[i]),
            f"{EMIT_PREFIX}_pred_bucket": BINARY_CLASS_MAPPING[y_pred_oof[i]],
            f"{EMIT_PREFIX}_true_class": int(y_oof[i]),
            f"{EMIT_PREFIX}_true_bucket": BINARY_CLASS_MAPPING[y_oof[i]],
        }

        if y_proba_oof is not None:
            pred_dict[f"{EMIT_PREFIX}_prob_low"] = float(y_proba_oof[i, 0])
            pred_dict[f"{EMIT_PREFIX}_prob_high"] = float(y_proba_oof[i, 1])
            pred_dict[f"{EMIT_PREFIX}_confidence"] = float(np.max(y_proba_oof[i]))
            pred_dict[f"{EMIT_PREFIX}_class_probs"] = {
                "low": float(y_proba_oof[i, 0]),
                "high": float(y_proba_oof[i, 1]),
            }

        # Add metadata
        pred_dict.update(
            {
                "sample_weight": float(weights_oof[i]),
                "hyperparams": best_params,
            }
        )

        oof_predictions.append(pred_dict)

    # Save models and results
    final_model_path = output_dir / f"final_binary_{config.model_type}_model.joblib"
    preprocessor_path = output_dir / "final_binary_preprocessor.joblib"
    oof_predictions_path = output_dir / f"final_oof_predictions.jsonl"
    oof_metrics_path = output_dir / "final_oof_metrics.json"

    joblib.dump(final_model, final_model_path)
    # No preprocessor needed for raw features
    # joblib.dump(preprocessor, preprocessor_path)

    with open(oof_predictions_path, "w") as f:
        for pred in oof_predictions:
            f.write(json.dumps(pyify(pred)) + "\n")

    with open(oof_metrics_path, "w") as f:
        json.dump(pyify(oof_scores), f, indent=2)

    final_results = {
        "oof_scores": oof_scores,
        "oof_predictions": oof_predictions,
        "best_hyperparams": best_params,
        "model_path": str(final_model_path),
        "preprocessor_path": str(preprocessor_path),
        "n_final_train": len(final_inner_train_df),
        "n_oof_test": len(oof_test_df),
        "class_distribution": {
            "final_train": np.bincount(y_final_inner_train).tolist(),
            "oof_test": np.bincount(y_oof).tolist(),
        },
        "final_fold_policy": final_policy.to_metadata_dict(),
        "deflation": {
            "series": config.deflation_series,
            "base_year": config.deflation_base_year,
            "enabled": config.use_cpi_deflation,
        },
        "temporal_cv": {
            "inner_split_ratio": 0.8,
            "date_extraction_method": "case_id_patterns",
            "cpi_deflation_applied": cpi_df is not None,
        },
    }

    logger.info(f"Final binary model training completed!")
    logger.info(f"OOF test scores: {oof_scores}")
    logger.info(f"Results saved to {output_dir}")

    return final_results
