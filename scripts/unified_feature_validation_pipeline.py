#!/usr/bin/env python3
"""
Unified Feature Validation Pipeline - Optimized for Mac M1

This script combines all validation logic into an efficient pipeline:
1. Tier 1: Ultra-fast basic screening (eliminate 70-80% quickly)
2. Tier 2: Discriminative power assessment (eliminate another 10-15%)
3. Tier 3: Comprehensive validation (only final 5-10%)
4. Auto-blacklist: Update governance immediately
5. Report: Generate comprehensive JSON performance report

Optimized for Mac M1 with minimal I/O and progressive filtering.

Usage:
    python scripts/unified_feature_validation_pipeline.py \
        --fold-dir data/final_stratified_kfold_splits_authoritative \
        --fold 3 \
        --sample-size 8000 \
        --output-dir docs/unified_validation_results \
        --auto-update-governance
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


def sanitize_for_json(obj):
    """Recursively sanitize data structure for JSON serialization by converting NaN to null."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj):
            return None
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj


class UnifiedFeatureValidator:
    """Unified feature validation pipeline optimized for Mac M1."""

    def __init__(self, output_dir: Path, auto_update_governance: bool = False):
        self.output_dir = output_dir
        self.auto_update_governance = auto_update_governance
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress = {
            "total_features": 0,
            "tier1_survivors": 0,
            "tier2_survivors": 0,
            "tier3_tested": 0,
            "blacklisted_features": [],
            "validation_summary": {},
            "step_timings": {},
            "start_time": time.time(),
        }

        # Efficient thresholds for progressive filtering
        self.thresholds = {
            # Tier 1: Ultra-fast elimination
            "tier1_max_sparsity": 95.0,  # Max 95% zeros
            "tier1_min_unique": 3,  # Minimum unique values
            "tier1_max_size_bias": 0.8,  # Very lenient size bias
            "tier1_max_leakage": 0.3,  # Very lenient leakage
            # Tier 2: Discriminative power
            "tier2_min_class0_auc": 0.52,  # Minimum discrimination
            "tier2_max_kw_pvalue": 0.1,  # Statistical significance
            "tier2_min_mutual_info": 0.001,  # Information content
            # Tier 3: Comprehensive validation
            "tier3_min_cv_stability": 0.7,  # Cross-validation stability
            "tier3_max_vif": 10.0,  # Multicollinearity
            "tier3_min_perm_importance": 0.0001,
            # Blacklisting criteria (strict)
            "blacklist_max_sparsity": 99.0,  # Extremely sparse
            "blacklist_max_size_bias": 0.5,  # Size bias
            "blacklist_max_leakage": 0.1,  # Leakage
            "blacklist_min_class0_auc": 0.51,  # Weak discrimination
        }

        # Results storage
        self.results = {
            "tier1_basic_screening": {},
            "tier2_discriminative_power": {},
            "tier3_advanced_validation": {},
            "blacklisted_features": {},
            "feature_performance_summary": {},
            "validation_summary": {},
        }

    def emit_progress(self, step: str, details: Dict[str, Any]):
        """Emit JSON progress report."""
        self.progress["step_timings"][step] = time.time() - self.progress["start_time"]

        progress_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "step": step,
            "progress": self.progress.copy(),
            "details": details,
        }

        # Save progress report
        progress_file = (
            self.output_dir / f"progress_{step.lower().replace(' ', '_')}.json"
        )
        with open(progress_file, "w") as f:
            json.dump(sanitize_for_json(progress_report), f, indent=2, default=str)

        print(f"📊 {step}: {details.get('summary', 'Processing...')}")

    def run_validation_pipeline(
        self, df: pd.DataFrame, target_col: str = "outcome_bin"
    ) -> Dict[str, Any]:
        """Run the complete validation pipeline with progressive filtering."""

        # Get all interpretable features
        feature_cols = [col for col in df.columns if col.startswith("interpretable_")]
        self.progress["total_features"] = len(feature_cols)

        self.emit_progress(
            "Pipeline Start",
            {
                "summary": f"Starting validation of {len(feature_cols)} features",
                "total_features": len(feature_cols),
                "sample_size": len(df),
            },
        )

        # TIER 1: Ultra-fast basic screening
        tier1_survivors = self._run_tier1_ultra_fast_screening(df, feature_cols)

        # TIER 2: Discriminative power assessment
        tier2_survivors = self._run_tier2_discriminative_assessment(
            df, tier1_survivors, target_col
        )

        # TIER 3: Comprehensive validation
        self._run_tier3_comprehensive_validation(df, tier2_survivors, target_col)

        # Auto-blacklist failing features
        blacklisted = self._auto_blacklist_failed_features()

        # Generate comprehensive report
        final_results = self._generate_comprehensive_report()

        # Update governance if requested
        if self.auto_update_governance:
            self._update_column_governance(blacklisted)

        self.emit_progress(
            "Pipeline Complete",
            {
                "summary": f"Validation complete: {len(tier2_survivors)} high-quality features identified",
                "tier1_survivors": len(tier1_survivors),
                "tier2_survivors": len(tier2_survivors),
                "tier3_tested": self.progress["tier3_tested"],
                "blacklisted": len(blacklisted),
                "total_runtime": time.time() - self.progress["start_time"],
            },
        )

        return final_results

    def _run_tier1_ultra_fast_screening(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> List[str]:
        """Tier 1: Ultra-fast basic screening to eliminate obvious failures."""
        start_time = time.time()

        # Use vectorized operations for maximum efficiency
        surviving_features = []
        failed_features = {}

        # Pre-compute size bias and leakage columns if available
        size_col = None
        target_col = None
        if "case_size" in df.columns:
            size_col = df["case_size"]
        if "outcome_bin" in df.columns:
            target_col = df["outcome_bin"]

        # Batch process features for efficiency
        batch_size = 50  # Optimal for Mac M1
        for i in range(0, len(feature_cols), batch_size):
            batch = feature_cols[i : i + batch_size]

            for feature in batch:
                if feature not in df.columns:
                    continue

                try:
                    # Vectorized basic statistics
                    feature_series = df[feature].dropna()
                    if len(feature_series) == 0:
                        failed_features[feature] = "empty_feature"
                        continue

                    # Fast sparsity check
                    zero_pct = (feature_series == 0).mean() * 100
                    unique_count = len(feature_series.unique())

                    # Basic elimination criteria
                    if zero_pct > self.thresholds["tier1_max_sparsity"]:
                        failed_features[feature] = "too_sparse"
                        continue

                    if unique_count < self.thresholds["tier1_min_unique"]:
                        failed_features[feature] = "insufficient_variation"
                        continue

                    # Fast bias checks (if data available)
                    size_bias_corr = 0.0
                    if size_col is not None and len(feature_series) > 10:
                        # Use only non-null intersection for speed
                        valid_idx = feature_series.notna() & size_col.notna()
                        if valid_idx.sum() > 10:
                            size_bias_corr = feature_series[valid_idx].corr(
                                size_col[valid_idx], method="spearman"
                            )
                            # Handle NaN values
                            if pd.isna(size_bias_corr):
                                size_bias_corr = 0.0
                            elif (
                                abs(size_bias_corr)
                                > self.thresholds["tier1_max_size_bias"]
                            ):
                                failed_features[feature] = "size_bias"
                                continue

                    leakage_corr = 0.0
                    if target_col is not None and len(feature_series) > 10:
                        valid_idx = feature_series.notna() & target_col.notna()
                        if valid_idx.sum() > 10:
                            leakage_corr = feature_series[valid_idx].corr(
                                target_col[valid_idx], method="spearman"
                            )
                            # Handle NaN values
                            if pd.isna(leakage_corr):
                                leakage_corr = 0.0
                            elif (
                                abs(leakage_corr) > self.thresholds["tier1_max_leakage"]
                            ):
                                failed_features[feature] = "leakage"
                                continue

                    # Feature survives Tier 1
                    surviving_features.append(feature)

                    # Store results efficiently
                    self.results["tier1_basic_screening"][feature] = {
                        "zero_pct": zero_pct,
                        "unique_count": unique_count,
                        "size_bias_corr": size_bias_corr,
                        "leakage_corr": leakage_corr,
                        "passes_tier1": True,
                    }

                except Exception as e:
                    failed_features[feature] = f"error_{str(e)[:20]}"
                    continue

        # Store failed features
        for feature, reason in failed_features.items():
            self.results["tier1_basic_screening"][feature] = {
                "passes_tier1": False,
                "failure_reason": reason,
            }

        elapsed = time.time() - start_time
        self.progress["tier1_survivors"] = len(surviving_features)

        self.emit_progress(
            "Tier 1 Complete",
            {
                "summary": f"{len(surviving_features)}/{len(feature_cols)} features passed basic screening",
                "survivors": len(surviving_features),
                "failed": len(failed_features),
                "elapsed_seconds": elapsed,
                "efficiency": f"{len(feature_cols)/elapsed:.0f} features/second",
            },
        )

        return surviving_features

    def _run_tier2_discriminative_assessment(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str
    ) -> List[str]:
        """Tier 2: Discriminative power assessment for Tier 1 survivors."""
        start_time = time.time()

        if target_col not in df.columns:
            print(f"Warning: Target column {target_col} not found, skipping Tier 2")
            return feature_cols

        surviving_features = []

        # Pre-compute target statistics for efficiency
        target_data = df[target_col].dropna()
        target_unique = target_data.unique()

        for feature in feature_cols:
            if feature not in df.columns:
                continue

            try:
                # Get clean data intersection
                clean_data = df[[feature, target_col]].dropna()
                if len(clean_data) < 20:
                    continue

                X = clean_data[feature].values
                y = clean_data[target_col].values

                # Fast mutual information approximation
                try:
                    # Use quantile binning for speed
                    X_binned = pd.qcut(
                        X, q=min(5, len(np.unique(X))), duplicates="drop"
                    )
                    mutual_info = mutual_info_score(y, X_binned.cat.codes)
                except:
                    mutual_info = 0.0

                # Fast Kruskal-Wallis test
                try:
                    groups = [X[y == k] for k in target_unique if (y == k).sum() > 0]
                    if len(groups) >= 2:
                        _, kw_pvalue = kruskal(*groups)
                    else:
                        kw_pvalue = 1.0
                except:
                    kw_pvalue = 1.0

                # Fast Class 0 discrimination
                try:
                    if 0 in target_unique and len(target_unique) > 1:
                        y_binary = (y == 0).astype(int)
                        if len(np.unique(y_binary)) == 2 and X.std() > 0:
                            class0_auc = roc_auc_score(y_binary, X)
                            class0_auc = max(class0_auc, 1 - class0_auc)
                        else:
                            class0_auc = 0.5
                    else:
                        class0_auc = 0.5
                except:
                    class0_auc = 0.5

                # Apply Tier 2 criteria
                passes_mutual_info = (
                    mutual_info >= self.thresholds["tier2_min_mutual_info"]
                )
                passes_significance = (
                    kw_pvalue <= self.thresholds["tier2_max_kw_pvalue"]
                )
                passes_class0_auc = (
                    class0_auc >= self.thresholds["tier2_min_class0_auc"]
                )

                passes_tier2 = (
                    passes_mutual_info and passes_significance and passes_class0_auc
                )

                # Store results
                self.results["tier2_discriminative_power"][feature] = {
                    "mutual_info": mutual_info,
                    "kw_pvalue": kw_pvalue,
                    "class0_auc": class0_auc,
                    "passes_mutual_info": passes_mutual_info,
                    "passes_significance": passes_significance,
                    "passes_class0_auc": passes_class0_auc,
                    "passes_tier2": passes_tier2,
                }

                if passes_tier2:
                    surviving_features.append(feature)

            except Exception as e:
                self.results["tier2_discriminative_power"][feature] = {
                    "passes_tier2": False,
                    "error": str(e)[:50],
                }

        elapsed = time.time() - start_time
        self.progress["tier2_survivors"] = len(surviving_features)

        self.emit_progress(
            "Tier 2 Complete",
            {
                "summary": f"{len(surviving_features)}/{len(feature_cols)} features passed discriminative assessment",
                "survivors": len(surviving_features),
                "tested": len(feature_cols),
                "elapsed_seconds": elapsed,
            },
        )

        return surviving_features

    def _run_tier3_comprehensive_validation(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str
    ):
        """Tier 3: Comprehensive validation for high-quality features only."""
        start_time = time.time()

        if not feature_cols:
            self.emit_progress(
                "Tier 3 Skipped", {"summary": "No features qualified for Tier 3"}
            )
            return

        # Use smaller sample for expensive Tier 3 operations
        sample_size = min(3000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)

        for feature in feature_cols:
            if feature not in df_sample.columns or target_col not in df_sample.columns:
                continue

            try:
                clean_data = df_sample[[feature, target_col]].dropna()
                if len(clean_data) < 50:
                    continue

                X = clean_data[[feature]].values
                y = clean_data[target_col].values

                # Cross-validation stability
                cv_stability = self._test_cv_stability(X, y)

                # Permutation importance
                perm_importance = self._test_permutation_importance(X, y)

                # VIF (multicollinearity) - simplified for efficiency
                vif_score = self._calculate_vif_fast(
                    df_sample, feature, feature_cols[:20]
                )

                # Store comprehensive results
                self.results["tier3_advanced_validation"][feature] = {
                    "cv_stability": cv_stability,
                    "permutation_importance": perm_importance,
                    "vif_score": vif_score,
                    "passes_cv_stability": cv_stability
                    >= self.thresholds["tier3_min_cv_stability"],
                    "passes_vif": vif_score <= self.thresholds["tier3_max_vif"],
                    "passes_permutation": perm_importance
                    >= self.thresholds["tier3_min_perm_importance"],
                }

            except Exception as e:
                self.results["tier3_advanced_validation"][feature] = {
                    "error": str(e)[:50]
                }

        elapsed = time.time() - start_time
        self.progress["tier3_tested"] = len(self.results["tier3_advanced_validation"])

        self.emit_progress(
            "Tier 3 Complete",
            {
                "summary": f"{len(self.results['tier3_advanced_validation'])} features underwent comprehensive validation",
                "tested": len(self.results["tier3_advanced_validation"]),
                "elapsed_seconds": elapsed,
            },
        )

    def _test_cv_stability(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fast cross-validation stability test."""
        try:
            y_binary = (y == 0).astype(int)
            if len(np.unique(y_binary)) < 2:
                return 0.0

            # Use 3-fold for speed
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            aucs = []
            scaler = StandardScaler()

            for train_idx, test_idx in skf.split(X, y_binary):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_binary[train_idx], y_binary[test_idx]

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Fast logistic regression
                model = LogisticRegression(
                    random_state=42, max_iter=500, solver="liblinear"
                )
                model.fit(X_train_scaled, y_train)

                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                aucs.append(auc)

            return 1 - (np.std(aucs) / (np.mean(aucs) + 1e-8))
        except:
            return 0.0

    def _test_permutation_importance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fast permutation importance test."""
        try:
            y_binary = (y == 0).astype(int)
            if len(np.unique(y_binary)) < 2:
                return 0.0

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fast model
            model = LogisticRegression(
                random_state=42, max_iter=500, solver="liblinear"
            )
            model.fit(X_scaled, y_binary)

            # Reduced permutations for speed
            perm_importance = permutation_importance(
                model, X_scaled, y_binary, n_repeats=3, random_state=42
            )

            return perm_importance.importances_mean[0]
        except:
            return 0.0

    def _calculate_vif_fast(
        self, df: pd.DataFrame, target_feature: str, feature_list: List[str]
    ) -> float:
        """Fast VIF calculation."""
        try:
            # Select only numeric features that exist and have variation
            numeric_features = []
            for f in feature_list:
                if f in df.columns and df[f].dtype in ["int64", "float64"]:
                    if df[f].std() > 0:
                        numeric_features.append(f)

            if target_feature not in numeric_features or len(numeric_features) < 3:
                return 1.0

            # Small sample for speed
            sample_df = (
                df[numeric_features]
                .dropna()
                .sample(n=min(500, len(df)), random_state=42)
            )

            if len(sample_df) < 10:
                return 1.0

            # Simple correlation-based approximation for speed
            corr_matrix = sample_df.corr()
            if target_feature not in corr_matrix.index:
                return 1.0

            # Max correlation as VIF proxy
            correlations = corr_matrix.loc[target_feature].drop(target_feature)
            max_corr = abs(correlations).max()

            # Convert correlation to VIF approximation
            if max_corr > 0.95:
                return 999.0
            else:
                return 1.0 / (1.0 - max_corr**2)

        except:
            return 1.0

    def _auto_blacklist_failed_features(self) -> List[str]:
        """Automatically identify features for blacklisting."""
        blacklisted = []
        blacklist_reasons = {}

        # Analyze all results to identify patterns for blacklisting
        all_features = set()
        for tier_results in self.results.values():
            if isinstance(tier_results, dict):
                all_features.update(tier_results.keys())

        for feature in all_features:
            reasons = []

            # Check Tier 1 failures
            if feature in self.results["tier1_basic_screening"]:
                t1 = self.results["tier1_basic_screening"][feature]
                if not t1.get("passes_tier1", False):
                    failure_reason = t1.get("failure_reason", "unknown")
                    if failure_reason in ["too_sparse", "size_bias", "leakage"]:
                        reasons.append(failure_reason)

            # Check severe discrimination failures
            if feature in self.results["tier2_discriminative_power"]:
                t2 = self.results["tier2_discriminative_power"][feature]
                if (
                    t2.get("class0_auc", 0.5)
                    < self.thresholds["blacklist_min_class0_auc"]
                ):
                    reasons.append("weak_discrimination")

            # Add to blacklist if severe failures
            if reasons:
                blacklisted.append(feature)
                blacklist_reasons[feature] = reasons

        # Store blacklist results
        self.results["blacklisted_features"] = blacklist_reasons
        self.progress["blacklisted_features"] = blacklisted

        self.emit_progress(
            "Auto-Blacklist Complete",
            {
                "summary": f"{len(blacklisted)} features identified for blacklisting",
                "blacklisted_count": len(blacklisted),
                "blacklist_reasons": blacklist_reasons,
            },
        )

        return blacklisted

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""

        # Aggregate all results
        all_features = set()
        for tier_results in self.results.values():
            if isinstance(tier_results, dict):
                all_features.update(tier_results.keys())

        feature_summary = {}

        for feature in all_features:
            if feature in ["feature_performance_summary", "validation_summary"]:
                continue

            summary = {
                "feature_name": feature,
                "tier1_passed": False,
                "tier2_passed": False,
                "tier3_tested": False,
                "blacklisted": False,
                "overall_score": 0.0,
                "performance_metrics": {},
                "test_results": {},
            }

            # Aggregate results from all tiers
            if feature in self.results["tier1_basic_screening"]:
                t1 = self.results["tier1_basic_screening"][feature]
                summary["tier1_passed"] = t1.get("passes_tier1", False)
                summary["test_results"]["tier1"] = t1

            if feature in self.results["tier2_discriminative_power"]:
                t2 = self.results["tier2_discriminative_power"][feature]
                summary["tier2_passed"] = t2.get("passes_tier2", False)
                summary["test_results"]["tier2"] = t2
                summary["performance_metrics"].update(
                    {
                        "class0_auc": t2.get("class0_auc", 0.5),
                        "mutual_info": t2.get("mutual_info", 0.0),
                        "kw_pvalue": t2.get("kw_pvalue", 1.0),
                    }
                )

            if feature in self.results["tier3_advanced_validation"]:
                t3 = self.results["tier3_advanced_validation"][feature]
                summary["tier3_tested"] = True
                summary["test_results"]["tier3"] = t3
                summary["performance_metrics"].update(
                    {
                        "cv_stability": t3.get("cv_stability", 0.0),
                        "permutation_importance": t3.get("permutation_importance", 0.0),
                        "vif_score": t3.get("vif_score", 999.0),
                    }
                )

            if feature in self.results["blacklisted_features"]:
                summary["blacklisted"] = True
                summary["blacklist_reasons"] = self.results["blacklisted_features"][
                    feature
                ]

            # Calculate overall score
            tests_passed = sum(
                [
                    summary["tier1_passed"],
                    summary["tier2_passed"],
                    summary.get("tier3_tested", False),
                ]
            )
            total_tests = sum(
                [
                    1 if feature in self.results["tier1_basic_screening"] else 0,
                    1 if feature in self.results["tier2_discriminative_power"] else 0,
                    1 if feature in self.results["tier3_advanced_validation"] else 0,
                ]
            )

            if total_tests > 0:
                summary["overall_score"] = tests_passed / total_tests

            feature_summary[feature] = summary

        # Generate validation summary
        validation_summary = {
            "total_features_tested": len(all_features),
            "tier1_survivors": self.progress["tier1_survivors"],
            "tier2_survivors": self.progress["tier2_survivors"],
            "tier3_tested": self.progress["tier3_tested"],
            "blacklisted_count": len(self.progress["blacklisted_features"]),
            "top_performers": self._identify_top_performers(feature_summary),
            "step_timings": self.progress["step_timings"],
            "total_runtime": time.time() - self.progress["start_time"],
        }

        # Store final results
        self.results["feature_performance_summary"] = feature_summary
        self.results["validation_summary"] = validation_summary

        # Save comprehensive results
        with open(self.output_dir / "unified_validation_results.json", "w") as f:
            json.dump(sanitize_for_json(self.results), f, indent=2, default=str)

        self.emit_progress(
            "Report Generation Complete",
            {
                "summary": f"Comprehensive report generated with {len(feature_summary)} features analyzed",
                "total_features": len(feature_summary),
                "top_performers": len(validation_summary["top_performers"]),
            },
        )

        return self.results

    def _identify_top_performers(
        self, feature_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify top performing features."""

        # Filter to features that passed Tier 2
        qualified_features = [
            (fname, fdata)
            for fname, fdata in feature_summary.items()
            if fdata["tier2_passed"] and not fdata["blacklisted"]
        ]

        # Sort by performance metrics
        qualified_features.sort(
            key=lambda x: (
                x[1]["overall_score"],
                x[1]["performance_metrics"].get("class0_auc", 0.5),
                -x[1]["performance_metrics"].get("kw_pvalue", 1.0),
            ),
            reverse=True,
        )

        return [
            {
                "feature_name": fname,
                "overall_score": fdata["overall_score"],
                "class0_auc": fdata["performance_metrics"].get("class0_auc", 0.5),
                "tier1_passed": fdata["tier1_passed"],
                "tier2_passed": fdata["tier2_passed"],
                "tier3_tested": fdata["tier3_tested"],
            }
            for fname, fdata in qualified_features[:20]
        ]

    def _update_column_governance(self, blacklisted_features: List[str]):
        """Update column governance with new blacklist patterns."""
        if not blacklisted_features:
            return

        governance_file = (
            Path(__file__).parent.parent
            / "src"
            / "corp_speech_risk_dataset"
            / "fully_interpretable"
            / "column_governance.py"
        )

        try:
            # Read existing governance
            with open(governance_file, "r") as f:
                content = f.read()

            # Generate new blacklist patterns
            new_patterns = []
            for feature in blacklisted_features:
                escaped_feature = feature.replace("_", r"\_")
                new_patterns.append(f'    r"^{escaped_feature}$",  # auto_blacklisted')

            # Insert before the closing bracket of BLOCKLIST_PATTERNS
            insert_text = (
                "\n    # AUTO-BLACKLISTED FEATURES\n" + "\n".join(new_patterns) + "\n"
            )

            # Find the last line of BLOCKLIST_PATTERNS and insert before the closing bracket
            if "# AUTO-BLACKLISTED FEATURES" not in content:
                insert_pos = content.rfind("]", content.find("BLOCKLIST_PATTERNS"))
                updated_content = (
                    content[:insert_pos] + insert_text + content[insert_pos:]
                )

                # Write updated governance
                with open(governance_file, "w") as f:
                    f.write(updated_content)

                self.emit_progress(
                    "Governance Updated",
                    {
                        "summary": f"Added {len(blacklisted_features)} features to governance blacklist",
                        "blacklisted_count": len(blacklisted_features),
                        "governance_file": str(governance_file),
                    },
                )

        except Exception as e:
            print(f"Warning: Could not update governance file: {e}")


def load_kfold_data_efficiently(
    fold_dir: str, fold: int, sample_size: int
) -> pd.DataFrame:
    """Load k-fold data efficiently for Mac M1."""
    print(f"📊 Loading data (fold {fold}, up to {sample_size:,} records)...")

    fold_path = Path(fold_dir) / f"fold_{fold}" / "train.jsonl"
    if not fold_path.exists():
        raise ValueError(f"K-fold file not found: {fold_path}")

    # Use chunked reading for memory efficiency
    records = []
    with open(fold_path, "r") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break

            try:
                record = json.loads(line)
                records.append(record)
            except:
                continue

    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df):,} records")

    return df


def extract_features_efficiently(df: pd.DataFrame) -> pd.DataFrame:
    """Extract interpretable features efficiently on Mac M1."""
    print("🔍 Extracting interpretable features...")

    extractor = InterpretableFeatureExtractor()

    # Optimize batch size for Mac M1
    batch_size = 500  # Smaller batches for better memory management
    all_features = []

    # Use efficient iteration
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(0, len(df), batch_size):
        batch_num = i // batch_size + 1
        batch = df.iloc[i : i + batch_size]
        batch_features = []

        for _, row in batch.iterrows():
            text = row.get("text", "")
            context = row.get("context", "")
            try:
                features = extractor.extract_features(text, context)
                batch_features.append(features)
            except:
                # Add empty features for failed extractions
                batch_features.append({})

        all_features.extend(batch_features)

        if batch_num % 5 == 0 or batch_num == total_batches:
            print(
                f"  Processed {min(i + batch_size, len(df)):,}/{len(df):,} records ({batch_num}/{total_batches} batches)"
            )

    # Convert to DataFrame efficiently
    features_df = pd.DataFrame(all_features)

    # Add interpretable_ prefix
    features_df.columns = [f"interpretable_{col}" for col in features_df.columns]

    # Combine with original data
    result_df = pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )

    feature_cols = [
        col for col in result_df.columns if col.startswith("interpretable_")
    ]
    print(f"✓ Extracted {len(feature_cols)} interpretable features")

    return result_df


def main():
    """Main execution optimized for Mac M1."""
    parser = argparse.ArgumentParser(
        description="Unified feature validation pipeline optimized for Mac M1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--fold-dir", required=True, help="K-fold directory path")
    parser.add_argument("--fold", type=int, default=3, help="Which fold to use")
    parser.add_argument(
        "--sample-size", type=int, default=8000, help="Sample size for validation"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/unified_validation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--auto-update-governance",
        action="store_true",
        help="Automatically update column governance with blacklisted features",
    )

    args = parser.parse_args()

    print("🚀 UNIFIED FEATURE VALIDATION PIPELINE (Mac M1 Optimized)")
    print("=" * 70)
    print(f"Fold directory: {args.fold_dir}")
    print(f"Using fold: {args.fold}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Output directory: {args.output_dir}")
    print(f"Auto-update governance: {args.auto_update_governance}")
    print()

    try:
        # Initialize validator
        output_dir = Path(args.output_dir)
        validator = UnifiedFeatureValidator(output_dir, args.auto_update_governance)

        # Load data efficiently
        df = load_kfold_data_efficiently(args.fold_dir, args.fold, args.sample_size)

        # Extract features efficiently
        df_with_features = extract_features_efficiently(df)

        # Run unified validation pipeline
        results = validator.run_validation_pipeline(df_with_features, "outcome_bin")

        # Print final summary
        print("\n" + "=" * 70)
        print("🎉 VALIDATION PIPELINE COMPLETE")
        print("=" * 70)

        vs = results.get("validation_summary", {})
        print(f"📊 Total features: {vs.get('total_features_tested', 0)}")
        print(f"✅ Tier 1 survivors: {vs.get('tier1_survivors', 0)}")
        print(f"🎯 Tier 2 survivors: {vs.get('tier2_survivors', 0)}")
        print(f"🔬 Tier 3 tested: {vs.get('tier3_tested', 0)}")
        print(f"❌ Blacklisted: {vs.get('blacklisted_count', 0)}")
        print(f"⏱️  Total runtime: {vs.get('total_runtime', 0):.1f}s")

        print(f"\n🏆 TOP 5 PERFORMING FEATURES:")
        for i, perf in enumerate(vs.get("top_performers", [])[:5], 1):
            print(
                f"{i}. {perf['feature_name']} (Score: {perf['overall_score']:.2f}, AUC: {perf['class0_auc']:.3f})"
            )

        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"📊 Progress reports: {output_dir}/progress_*.json")
        print(f"📋 Full results: {output_dir}/unified_validation_results.json")

        return 0

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
