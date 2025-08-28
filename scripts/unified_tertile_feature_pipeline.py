#!/usr/bin/env python3
"""
Unified Tertile Feature Validation Pipeline

This script is adapted from the binary feature pipeline to work with tertile classification.
It validates features for 3-class (low/medium/high) risk prediction using the same
comprehensive validation framework.

KEY DIFFERENCES FROM BINARY:
1. Works with 3 classes (0, 1, 2) instead of 2
2. Uses Quadratic Weighted Kappa (QWK) as primary metric instead of AUC
3. Adapted thresholds for 3-class discrimination
4. Multi-class leakage detection
5. Per-class coverage requirements

Usage:
    python scripts/unified_tertile_feature_pipeline.py \
        --data-dir data/final_stratified_kfold_splits_authoritative_complete \
        --output-dir results/tertile_comprehensive_validation \
        --fold 3 \
        --sample-size 20000 \
        --iterations 1
"""

import argparse
import json
import orjson
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, kruskal
from sklearn.metrics import (
    mutual_info_score,
    cohen_kappa_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import datetime
import hashlib
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)
from corp_speech_risk_dataset.fully_interpretable.column_governance import (
    validate_columns,
)

warnings.filterwarnings("ignore")

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=3):
    """Calculate Quadratic Weighted Kappa for ordinal classification."""
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2

    weights = weights / weights.max()

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Calculate expected matrix
    n = len(y_true)
    sum_rows = cm.sum(axis=1)
    sum_cols = cm.sum(axis=0)
    expected = np.outer(sum_rows, sum_cols) / n

    # Calculate weighted sums
    po = np.sum(weights * cm) / n
    pe = np.sum(weights * expected) / n

    # Calculate kappa
    kappa = 1 - (po / pe) if pe != 0 else 0

    return kappa


class TertileFeaturePipeline:
    """Unified pipeline for tertile feature development and validation."""

    @staticmethod
    def _convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                k: TertileFeaturePipeline._convert_numpy_types(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [TertileFeaturePipeline._convert_numpy_types(v) for v in obj]
        return obj

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        fold: int = 3,
        sample_size: int = 50000,
        auto_update_governance: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fold = fold
        self.sample_size = sample_size
        self.auto_update_governance = auto_update_governance

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Initialize tracking
        self.iteration_results = []
        self.governance_updates = []

        # Load DNT columns
        self.dnt_columns = self._load_dnt_columns()

        # Tertile-specific thresholds
        self.TERTILE_THRESHOLDS = {
            "quality": {
                "sparsity": 0.99,  # Max 99% sparse
                "missing": 0.30,  # Max 30% missing
                "per_class_coverage": 0.005,  # Min 0.5% per class
            },
            "discriminative": {
                "qwk": 0.05,  # Min QWK (positive)
                "qwk_ci_lower": 0.0,  # CI must include positive values
                "cv_std": 0.10,  # Max CV std
                "min_fold_qwk": -0.05,  # Min per-fold QWK
            },
            "leakage": {
                "outcome": 0.20,  # Max outcome leakage (3 classes)
                "groups": 0.25,  # Max group leakage
                "case_bias": 0.30,  # Max case bias
                "quote_bias": 0.30,  # Max quote bias
            },
            "temporal": {"era_variance": 0.10},  # Max variance across eras
            "causality": {
                "ablation_delta": 0.02,  # Min delta QWK
                "residual_qwk": 0.03,  # Min residual QWK
            },
        }

    def _load_dnt_columns(self) -> Set[str]:
        """Load do-not-train columns from manifest."""
        dnt_path = self.data_dir / "dnt_manifest.json"
        if dnt_path.exists():
            with open(dnt_path, "rb") as f:
                manifest = orjson.loads(f.read())
                return set(manifest.get("do_not_train", []))
        return set()

    def _clean_feature_name(self, feature_name: str) -> str:
        """Clean feature name for display by removing prefixes."""
        return (
            feature_name.replace("interpretable_", "")
            .replace("feat_new_", "new_")
            .replace("feat_new2_", "new2_")
            .replace("feat_new3_", "new3_")
            .replace("feat_new4_", "new4_")
            .replace("feat_new5_", "new5_")
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test data for specified fold."""
        logger.info(f"Loading data from fold {self.fold}")

        fold_dir = self.data_dir / f"fold_{self.fold}"

        # Load data using orjson for speed
        def load_jsonl_fast(path):
            """Load JSONL file using orjson for faster parsing."""
            records = []
            with open(path, "rb") as f:
                for line in f:
                    if line.strip():
                        records.append(orjson.loads(line))
            return pd.DataFrame(records)

        train_df = load_jsonl_fast(fold_dir / "train.jsonl")
        val_df = load_jsonl_fast(
            fold_dir / "dev.jsonl"
        )  # Note: file is named dev.jsonl not val.jsonl
        test_df = (
            load_jsonl_fast(fold_dir / "test.jsonl")
            if (fold_dir / "test.jsonl").exists()
            else val_df
        )

        # Verify tertile labels
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            unique_labels = df["outcome_bin"].unique()
            if not set(unique_labels).issubset({0, 1, 2}):
                raise ValueError(f"{name} has non-tertile labels: {unique_labels}")
            logger.info(
                f"{name} label distribution: {df['outcome_bin'].value_counts().sort_index().to_dict()}"
            )

        # Sample if needed
        if self.sample_size and len(train_df) > self.sample_size:
            # Stratified sampling to maintain class distribution
            from sklearn.model_selection import train_test_split

            _, train_df = train_test_split(
                train_df,
                test_size=self.sample_size / len(train_df),
                stratify=train_df["outcome_bin"],
                random_state=42,
            )
            logger.info(f"Sampled training data to {len(train_df)} rows")

        return train_df, val_df, test_df

    def get_feature_candidates(self, df: pd.DataFrame) -> List[str]:
        """Get all interpretable features as candidates."""
        all_interpretable = [
            col
            for col in df.columns
            if (
                col.startswith("interpretable_")
                or col.startswith("feat_new_")
                or col.startswith("feat_new2_")
                or col.startswith("feat_new3_")
                or col.startswith("feat_new4_")
                or col.startswith("feat_new5_")
            )
            and col not in self.dnt_columns
        ]

        logger.info(f"Found {len(all_interpretable)} feature candidates")
        return all_interpretable

    def validate_feature_quality(
        self, train_df: pd.DataFrame, feature: str
    ) -> Dict[str, Any]:
        """Validate basic quality metrics for a feature."""
        series = train_df[feature]

        # Basic statistics
        stats = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "unique_values": int(series.nunique()),
            "missing_rate": float(series.isna().mean()),
            "zero_rate": float((series == 0).mean()),
            "sparsity": float((series == 0).mean()),
        }

        # Per-class coverage (3 classes)
        per_class_coverage = {}
        for class_val in [0, 1, 2]:
            class_data = series[train_df["outcome_bin"] == class_val]
            coverage = (class_data > 0).mean() if len(class_data) > 0 else 0
            per_class_coverage[f"class_{class_val}_coverage"] = float(coverage)

        stats.update(per_class_coverage)

        # Quality checks
        passed = (
            stats["sparsity"] <= self.TERTILE_THRESHOLDS["quality"]["sparsity"]
            and stats["missing_rate"] <= self.TERTILE_THRESHOLDS["quality"]["missing"]
            and all(
                cov >= self.TERTILE_THRESHOLDS["quality"]["per_class_coverage"]
                for cov in per_class_coverage.values()
            )
        )

        return {
            "stats": stats,
            "passed": passed,
            "reason": None if passed else "Failed quality thresholds",
        }

    def validate_discriminative_power(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature: str
    ) -> Dict[str, Any]:
        """Validate discriminative power using QWK and multi-class metrics."""
        # Prepare data
        X_train = train_df[[feature]].values
        y_train = train_df["outcome_bin"].values
        X_val = val_df[[feature]].values
        y_val = val_df["outcome_bin"].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train multi-class logistic regression
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )

        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            # Calculate QWK
            qwk = quadratic_weighted_kappa(y_val, y_pred)

            # Calculate per-class F1
            f1_scores = f1_score(y_val, y_pred, average=None)
            macro_f1 = f1_score(y_val, y_pred, average="macro")

            # Cross-validation for stability
            cv_scores = []
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_cv_train = X_train_scaled[train_idx]
                y_cv_train = y_train[train_idx]
                X_cv_val = X_train_scaled[val_idx]
                y_cv_val = y_train[val_idx]

                cv_model = LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                )
                cv_model.fit(X_cv_train, y_cv_train)
                cv_pred = cv_model.predict(X_cv_val)
                cv_qwk = quadratic_weighted_kappa(y_cv_val, cv_pred)
                cv_scores.append(cv_qwk)

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Bootstrap CI for QWK
            n_bootstrap = 1000
            bootstrap_qwks = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(y_val), len(y_val), replace=True)
                boot_qwk = quadratic_weighted_kappa(y_val[idx], y_pred[idx])
                bootstrap_qwks.append(boot_qwk)

            ci_lower = np.percentile(bootstrap_qwks, 2.5)
            ci_upper = np.percentile(bootstrap_qwks, 97.5)

            # Effect size (Kruskal-Wallis for 3 groups)
            groups = [
                train_df[train_df["outcome_bin"] == i][feature].values
                for i in [0, 1, 2]
            ]
            h_stat, p_value = kruskal(*groups)
            effect_size = h_stat / (len(train_df) - 1)  # Normalized H statistic

            # Mutual information
            mi_score = mutual_info_score(
                y_train, pd.qcut(X_train.flatten(), q=10, duplicates="drop")
            )

            result = {
                "qwk": float(qwk),
                "qwk_ci": [float(ci_lower), float(ci_upper)],
                "cv_mean": float(cv_mean),
                "cv_std": float(cv_std),
                "cv_scores": [float(s) for s in cv_scores],
                "macro_f1": float(macro_f1),
                "per_class_f1": [float(f) for f in f1_scores],
                "effect_size": float(effect_size),
                "p_value": float(p_value),
                "mutual_info": float(mi_score),
            }

            # Check if passed
            passed = (
                qwk >= self.TERTILE_THRESHOLDS["discriminative"]["qwk"]
                and ci_lower
                >= self.TERTILE_THRESHOLDS["discriminative"]["qwk_ci_lower"]
                and cv_std <= self.TERTILE_THRESHOLDS["discriminative"]["cv_std"]
                and min(cv_scores)
                >= self.TERTILE_THRESHOLDS["discriminative"]["min_fold_qwk"]
            )

            result["passed"] = passed
            result["reason"] = None if passed else "Failed discriminative thresholds"

        except Exception as e:
            result = {
                "qwk": 0.0,
                "passed": False,
                "reason": f"Model fitting failed: {str(e)}",
            }

        return result

    def validate_leakage(self, train_df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """Validate for various types of leakage in tertile setting."""
        results = {}

        # 1. Direct outcome leakage (adapted for 3 classes)
        if "outcome_bin" in train_df.columns:
            # Use Kruskal-Wallis test for 3 groups
            groups = [
                train_df[train_df["outcome_bin"] == i][feature].values
                for i in [0, 1, 2]
            ]
            h_stat, p_value = kruskal(*groups)
            effect_size = h_stat / (len(train_df) - 1)

            results["outcome_leakage"] = {
                "h_statistic": float(h_stat),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "passed": effect_size <= self.TERTILE_THRESHOLDS["leakage"]["outcome"],
            }

        # 2. Group leakage (speaker, court, etc.)
        group_cols = ["speaker", "court", "circuit", "district"]
        for group_col in group_cols:
            if group_col in train_df.columns:
                # Check if feature varies significantly by group
                groups = train_df.groupby(group_col)[feature].agg(["mean", "count"])
                groups = groups[groups["count"] >= 10]  # Min group size

                if len(groups) > 1:
                    # ANOVA for multiple groups
                    group_data = [
                        train_df[train_df[group_col] == g][feature].values
                        for g in groups.index
                    ]
                    f_stat, p_value = stats.f_oneway(*group_data)

                    results[f"{group_col}_leakage"] = {
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "n_groups": len(groups),
                        "passed": p_value > 0.05,  # Not significant
                    }

        # 3. Case/Quote bias
        if "case_id" in train_df.columns:
            case_features = train_df.groupby("case_id")[feature].mean()
            case_outcomes = train_df.groupby("case_id")["outcome_bin"].first()

            # Correlation between case-level feature and outcome
            corr, p_value = spearmanr(case_features, case_outcomes)

            results["case_bias"] = {
                "correlation": float(corr),
                "p_value": float(p_value),
                "n_cases": len(case_features),
                "passed": abs(corr) <= self.TERTILE_THRESHOLDS["leakage"]["case_bias"],
            }

        # Overall leakage assessment
        passed = all(test.get("passed", True) for test in results.values())

        return {
            "tests": results,
            "passed": passed,
            "reason": None if passed else "Potential leakage detected",
        }

    def validate_temporal_stability(
        self, train_df: pd.DataFrame, feature: str
    ) -> Dict[str, Any]:
        """Validate temporal stability across different time periods."""
        if "decision_date" not in train_df.columns:
            return {"passed": True, "reason": "No temporal data available"}

        # Convert to datetime
        train_df["year"] = pd.to_datetime(train_df["decision_date"]).dt.year

        # Split into eras
        eras = {
            "early": train_df[train_df["year"] < 2010],
            "middle": train_df[(train_df["year"] >= 2010) & (train_df["year"] < 2018)],
            "recent": train_df[train_df["year"] >= 2018],
        }

        era_qwks = {}
        for era_name, era_df in eras.items():
            if len(era_df) < 100:  # Skip small eras
                continue

            # Simple QWK calculation for era
            X = era_df[[feature]].values
            y = era_df["outcome_bin"].values

            if len(np.unique(y)) < 3:  # Need all 3 classes
                continue

            # Fit model and predict
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )

            try:
                # Use cross-validation within era
                kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                era_scores = []

                for train_idx, val_idx in kf.split(X_scaled, y):
                    model.fit(X_scaled[train_idx], y[train_idx])
                    y_pred = model.predict(X_scaled[val_idx])
                    qwk = quadratic_weighted_kappa(y[val_idx], y_pred)
                    era_scores.append(qwk)

                era_qwks[era_name] = np.mean(era_scores)

            except:
                continue

        if len(era_qwks) < 2:
            return {"passed": True, "reason": "Insufficient temporal data"}

        # Calculate variance
        qwk_values = list(era_qwks.values())
        variance = np.var(qwk_values)

        result = {
            "era_qwks": era_qwks,
            "variance": float(variance),
            "passed": variance <= self.TERTILE_THRESHOLDS["temporal"]["era_variance"],
        }

        if not result["passed"]:
            result["reason"] = "High temporal variance"

        return result

    def validate_causality(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature: str
    ) -> Dict[str, Any]:
        """Validate causal relationships through ablation."""
        # Baseline model with feature
        X_train = train_df[[feature]].values
        y_train = train_df["outcome_bin"].values
        X_val = val_df[[feature]].values
        y_val = val_df["outcome_bin"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42
        )

        try:
            # Full model QWK
            model.fit(X_train_scaled, y_train)
            y_pred_full = model.predict(X_val_scaled)
            qwk_full = quadratic_weighted_kappa(y_val, y_pred_full)

            # Ablated model (shuffle feature)
            X_train_ablated = X_train_scaled.copy()
            np.random.shuffle(X_train_ablated)

            model_ablated = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            model_ablated.fit(X_train_ablated, y_train)

            # Predict with shuffled validation features
            X_val_ablated = X_val_scaled.copy()
            np.random.shuffle(X_val_ablated)
            y_pred_ablated = model_ablated.predict(X_val_ablated)
            qwk_ablated = quadratic_weighted_kappa(y_val, y_pred_ablated)

            # Residual test
            residuals = X_val_scaled - model.predict_proba(X_val_scaled)[:, 1:2]

            # Can residuals predict outcome?
            model_residual = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )

            try:
                model_residual.fit(
                    residuals[: int(len(residuals) * 0.8)],
                    y_val[: int(len(y_val) * 0.8)],
                )
                y_pred_residual = model_residual.predict(
                    residuals[int(len(residuals) * 0.8) :]
                )
                qwk_residual = quadratic_weighted_kappa(
                    y_val[int(len(y_val) * 0.8) :], y_pred_residual
                )
            except:
                qwk_residual = 0.0

            result = {
                "qwk_full": float(qwk_full),
                "qwk_ablated": float(qwk_ablated),
                "ablation_delta": float(qwk_full - qwk_ablated),
                "qwk_residual": float(qwk_residual),
            }

            # Check thresholds
            passed = (
                result["ablation_delta"]
                >= self.TERTILE_THRESHOLDS["causality"]["ablation_delta"]
                or result["qwk_residual"]
                <= self.TERTILE_THRESHOLDS["causality"]["residual_qwk"]
            )

            result["passed"] = passed
            result["reason"] = None if passed else "Weak causal relationship"

        except Exception as e:
            result = {"passed": False, "reason": f"Causality test failed: {str(e)}"}

        return result

    def run_comprehensive_validation(
        self,
        feature: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run all validation tests on a single feature."""
        logger.info(f"Validating feature: {feature}")

        results = {"feature": feature, "tests": {}}

        # Phase 1: Quality
        quality_result = self.validate_feature_quality(train_df, feature)
        results["tests"]["quality"] = quality_result

        if not quality_result["passed"]:
            results["overall_passed"] = False
            results["rejection_phase"] = "quality"
            return results

        # Phase 2: Discriminative Power
        disc_result = self.validate_discriminative_power(train_df, val_df, feature)
        results["tests"]["discriminative"] = disc_result

        if not disc_result["passed"]:
            results["overall_passed"] = False
            results["rejection_phase"] = "discriminative"
            return results

        # Phase 3: Leakage
        leakage_result = self.validate_leakage(train_df, feature)
        results["tests"]["leakage"] = leakage_result

        if not leakage_result["passed"]:
            results["overall_passed"] = False
            results["rejection_phase"] = "leakage"
            return results

        # Phase 4: Temporal Stability
        temporal_result = self.validate_temporal_stability(train_df, feature)
        results["tests"]["temporal"] = temporal_result

        if not temporal_result["passed"]:
            results["overall_passed"] = False
            results["rejection_phase"] = "temporal"
            return results

        # Phase 5: Causality
        causal_result = self.validate_causality(train_df, val_df, feature)
        results["tests"]["causality"] = causal_result

        if not causal_result["passed"]:
            results["overall_passed"] = False
            results["rejection_phase"] = "causality"
            return results

        # All tests passed
        results["overall_passed"] = True
        results["rejection_phase"] = None

        return results

    def generate_summary_report(
        self, all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary report of all validation results."""
        summary = {
            "total_features": len(all_results),
            "passed_features": sum(
                1 for r in all_results if r.get("overall_passed", False)
            ),
            "rejection_breakdown": {},
            "top_performers": [],
            "feature_details": [],
        }

        # Count rejections by phase
        for result in all_results:
            if not result.get("overall_passed", False):
                phase = result.get("rejection_phase", "unknown")
                summary["rejection_breakdown"][phase] = (
                    summary["rejection_breakdown"].get(phase, 0) + 1
                )

        # Get top performers by QWK
        passed_features = [r for r in all_results if r.get("overall_passed", False)]
        if passed_features:
            passed_features.sort(
                key=lambda x: x["tests"]["discriminative"].get("qwk", 0), reverse=True
            )
            summary["top_performers"] = [
                {
                    "feature": f["feature"],
                    "qwk": f["tests"]["discriminative"]["qwk"],
                    "macro_f1": f["tests"]["discriminative"]["macro_f1"],
                }
                for f in passed_features[:10]
            ]

        # Detailed results for all features
        for result in all_results:
            detail = {
                "feature": result["feature"],
                "passed": result.get("overall_passed", False),
                "rejection_phase": result.get("rejection_phase"),
                "qwk": None,
                "macro_f1": None,
            }

            if "discriminative" in result["tests"]:
                detail["qwk"] = result["tests"]["discriminative"].get("qwk")
                detail["macro_f1"] = result["tests"]["discriminative"].get("macro_f1")

            summary["feature_details"].append(detail)

        return summary

    def visualize_results(self, all_results: List[Dict[str, Any]], iteration: int):
        """Create visualizations of validation results."""
        # Filter for features that passed quality
        features_with_metrics = [
            r
            for r in all_results
            if "discriminative" in r["tests"]
            and r["tests"]["discriminative"].get("qwk") is not None
        ]

        if not features_with_metrics:
            logger.warning("No features with discriminative metrics to visualize")
            return

        # 1. Feature Performance Scatter
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        qwks = [r["tests"]["discriminative"]["qwk"] for r in features_with_metrics]
        f1s = [r["tests"]["discriminative"]["macro_f1"] for r in features_with_metrics]
        passed = [r.get("overall_passed", False) for r in features_with_metrics]

        colors = ["green" if p else "red" for p in passed]
        ax.scatter(qwks, f1s, c=colors, alpha=0.6, s=100)

        # Add labels for top features
        for r in features_with_metrics[:5]:
            ax.annotate(
                self._clean_feature_name(r["feature"]),
                (
                    r["tests"]["discriminative"]["qwk"],
                    r["tests"]["discriminative"]["macro_f1"],
                ),
                fontsize=8,
                rotation=45,
            )

        ax.set_xlabel("Quadratic Weighted Kappa")
        ax.set_ylabel("Macro F1 Score")
        ax.set_title("Tertile Feature Performance")
        ax.axhline(y=0.33, color="gray", linestyle="--", alpha=0.5, label="Random F1")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="Zero QWK")
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / f"feature_performance_iter_{iteration}.png", dpi=300
        )
        plt.close()

        # 2. Rejection Funnel
        phases = ["quality", "discriminative", "leakage", "temporal", "causality"]
        remaining = [len(all_results)]

        for phase in phases:
            rejected_in_phase = sum(
                1 for r in all_results if r.get("rejection_phase") == phase
            )
            remaining.append(remaining[-1] - rejected_in_phase)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        x = range(len(phases) + 1)
        labels = ["Start"] + [p.capitalize() for p in phases]

        ax.bar(x, remaining, color="skyblue", edgecolor="navy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Features Remaining")
        ax.set_title("Tertile Feature Validation Funnel")

        # Add percentage labels
        for i, v in enumerate(remaining):
            pct = (v / len(all_results)) * 100
            ax.text(i, v + 0.5, f"{v}\n({pct:.1f}%)", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / f"validation_funnel_iter_{iteration}.png", dpi=300
        )
        plt.close()

        # 3. QWK Distribution
        if features_with_metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            all_qwks = [
                r["tests"]["discriminative"]["qwk"] for r in features_with_metrics
            ]
            passed_qwks = [
                r["tests"]["discriminative"]["qwk"]
                for r in features_with_metrics
                if r.get("overall_passed", False)
            ]

            ax.hist(all_qwks, bins=30, alpha=0.5, label="All Features", color="blue")
            if passed_qwks:
                ax.hist(
                    passed_qwks,
                    bins=30,
                    alpha=0.7,
                    label="Passed Features",
                    color="green",
                )

            ax.axvline(x=0, color="red", linestyle="--", label="Zero QWK")
            ax.axvline(
                x=self.TERTILE_THRESHOLDS["discriminative"]["qwk"],
                color="orange",
                linestyle="--",
                label="Min Threshold",
            )

            ax.set_xlabel("Quadratic Weighted Kappa")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Feature QWK Scores")
            ax.legend()

            plt.tight_layout()
            plt.savefig(
                self.figures_dir / f"qwk_distribution_iter_{iteration}.png", dpi=300
            )
            plt.close()

    def save_results(
        self, all_results: List[Dict[str, Any]], summary: Dict[str, Any], iteration: int
    ):
        """Save all results to disk."""
        # Save detailed results using orjson
        results_path = self.output_dir / f"validation_results_iter_{iteration}.json"
        with open(results_path, "wb") as f:
            # Convert numpy types before serialization
            data_to_save = {
                "timestamp": datetime.datetime.now().isoformat(),
                "fold": self.fold,
                "sample_size": self.sample_size,
                "thresholds": self.TERTILE_THRESHOLDS,
                "results": self._convert_numpy_types(all_results),
                "summary": self._convert_numpy_types(summary),
            }
            f.write(orjson.dumps(data_to_save, option=orjson.OPT_INDENT_2))

        # Save summary report using orjson
        report_path = self.reports_dir / f"summary_report_iter_{iteration}.json"
        with open(report_path, "wb") as f:
            f.write(
                orjson.dumps(
                    self._convert_numpy_types(summary), option=orjson.OPT_INDENT_2
                )
            )

        # Create markdown report
        md_path = self.reports_dir / f"validation_report_iter_{iteration}.md"
        with open(md_path, "w") as f:
            f.write(f"# Tertile Feature Validation Report - Iteration {iteration}\n\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total Features Tested: {summary['total_features']}\n")
            f.write(f"- Features Passed: {summary['passed_features']} ")
            f.write(
                f"({summary['passed_features']/summary['total_features']*100:.1f}%)\n\n"
            )

            f.write("## Rejection Breakdown\n\n")
            for phase, count in summary["rejection_breakdown"].items():
                f.write(f"- {phase.capitalize()}: {count} features\n")
            f.write("\n")

            if summary["top_performers"]:
                f.write("## Top Performing Features\n\n")
                f.write("| Feature | QWK | Macro F1 |\n")
                f.write("|---------|-----|----------|\n")
                for feat in summary["top_performers"]:
                    f.write(f"| {self._clean_feature_name(feat['feature'])} | ")
                    f.write(f"{feat['qwk']:.3f} | {feat['macro_f1']:.3f} |\n")
                f.write("\n")

            f.write("## Detailed Results\n\n")
            f.write("| Feature | Passed | QWK | Rejection Phase |\n")
            f.write("|---------|--------|-----|----------------|\n")

            # Sort by QWK for better readability
            sorted_details = sorted(
                summary["feature_details"],
                key=lambda x: x.get("qwk", -999) if x.get("qwk") is not None else -999,
                reverse=True,
            )

            for detail in sorted_details[:50]:  # Top 50
                f.write(f"| {self._clean_feature_name(detail['feature'])} | ")
                f.write(f"{'✓' if detail['passed'] else '✗'} | ")
                f.write(
                    f"{detail['qwk']:.3f} | " if detail["qwk"] is not None else "N/A | "
                )
                f.write(f"{detail['rejection_phase'] or 'N/A'} |\n")

    def run_full_pipeline(self, iterations: int = 1):
        """Run the complete validation pipeline for specified iterations."""
        logger.info(
            f"Starting tertile feature validation pipeline with {iterations} iterations"
        )

        for iteration in range(iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting iteration {iteration + 1}/{iterations}")
            logger.info(f"{'='*50}\n")

            # Load data
            train_df, val_df, test_df = self.load_data()

            # Get feature candidates
            features = self.get_feature_candidates(train_df)

            # Validate all features
            all_results = []

            for feature in tqdm(
                features, desc=f"Validating features (iter {iteration + 1})"
            ):
                result = self.run_comprehensive_validation(
                    feature, train_df, val_df, test_df
                )
                all_results.append(result)

            # Generate summary
            summary = self.generate_summary_report(all_results)

            # Visualize results
            self.visualize_results(all_results, iteration + 1)

            # Save results
            self.save_results(all_results, summary, iteration + 1)

            # Store for tracking
            self.iteration_results.append(
                {"iteration": iteration + 1, "summary": summary, "results": all_results}
            )

            logger.info(f"\nIteration {iteration + 1} Complete!")
            logger.info(
                f"Passed features: {summary['passed_features']}/{summary['total_features']}"
            )

        # Final summary across all iterations
        self.generate_final_summary()

    def generate_final_summary(self):
        """Generate final summary across all iterations."""
        if not self.iteration_results:
            return

        logger.info("\n" + "=" * 50)
        logger.info("FINAL SUMMARY - ALL ITERATIONS")
        logger.info("=" * 50)

        # Aggregate results
        all_features = set()
        consistent_passes = set()

        for i, iter_result in enumerate(self.iteration_results):
            passed_features = {
                r["feature"]
                for r in iter_result["results"]
                if r.get("overall_passed", False)
            }

            if i == 0:
                consistent_passes = passed_features
            else:
                consistent_passes = consistent_passes.intersection(passed_features)

            all_features.update(r["feature"] for r in iter_result["results"])

        logger.info(f"Total unique features tested: {len(all_features)}")
        logger.info(f"Consistently passing features: {len(consistent_passes)}")

        # Save final summary
        final_summary = {
            "total_iterations": len(self.iteration_results),
            "total_features": len(all_features),
            "consistent_passes": list(consistent_passes),
            "per_iteration_summary": [
                {
                    "iteration": r["iteration"],
                    "passed": r["summary"]["passed_features"],
                    "total": r["summary"]["total_features"],
                }
                for r in self.iteration_results
            ],
        }

        with open(self.output_dir / "final_summary.json", "wb") as f:
            f.write(
                orjson.dumps(
                    self._convert_numpy_types(final_summary), option=orjson.OPT_INDENT_2
                )
            )

        # Create final report
        with open(self.output_dir / "TERTILE_VALIDATION_FINAL.md", "w") as f:
            f.write("# Tertile Feature Validation - Final Report\n\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")
            f.write(f"## Overview\n\n")
            f.write(f"- Dataset: {self.data_dir}\n")
            f.write(f"- Fold: {self.fold}\n")
            f.write(f"- Sample Size: {self.sample_size}\n")
            f.write(f"- Iterations: {len(self.iteration_results)}\n\n")

            f.write("## Results Summary\n\n")
            f.write(f"- Total Features Tested: {len(all_features)}\n")
            f.write(f"- Consistently Passing: {len(consistent_passes)} ")
            f.write(f"({len(consistent_passes)/len(all_features)*100:.1f}%)\n\n")

            if consistent_passes:
                f.write("## Consistently Passing Features\n\n")
                for feat in sorted(consistent_passes):
                    f.write(f"- {self._clean_feature_name(feat)}\n")
                f.write("\n")

            f.write("## Per-Iteration Results\n\n")
            f.write("| Iteration | Passed | Total | Pass Rate |\n")
            f.write("|-----------|--------|-------|----------|\n")

            for r in self.iteration_results:
                summary = r["summary"]
                pass_rate = summary["passed_features"] / summary["total_features"] * 100
                f.write(f"| {r['iteration']} | {summary['passed_features']} | ")
                f.write(f"{summary['total_features']} | {pass_rate:.1f}% |\n")

        logger.info(f"\nAll results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Tertile Feature Validation Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/final_stratified_kfold_splits_authoritative_complete",
        help="Path to tertile dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tertile_comprehensive_validation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--fold", type=int, default=3, help="Fold number to use (default: 3)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Sample size for training data (default: 20000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of validation iterations (default: 1)",
    )
    parser.add_argument(
        "--auto-update-governance",
        action="store_true",
        help="Automatically update column governance",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = TertileFeaturePipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        fold=args.fold,
        sample_size=args.sample_size,
        auto_update_governance=args.auto_update_governance,
    )

    # Run validation
    pipeline.run_full_pipeline(iterations=args.iterations)


if __name__ == "__main__":
    main()
