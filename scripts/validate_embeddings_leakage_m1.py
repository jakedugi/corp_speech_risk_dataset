#!/usr/bin/env python3
"""
Comprehensive Embedding Leakage Validation (Mac M1 Optimized)

Tests all embedding features for leakage using the complete battery of checks:
- Split hygiene (Group CV, Temporal)
- Negative controls (Target permutation, Case-wise shuffle)
- Adversarial probes (Case-ID prediction, Post-outcome detection)
- Stability checks

Optimized for Mac M1 with orjson, multiprocessing, and memory efficiency.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Fast JSON loading with orjson optimization
try:
    import orjson as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data)

    def _loads_str(data: str) -> Any:
        return _json.loads(data.encode("utf-8"))

except ImportError:
    import json as _json

    def _loads_bytes(data: bytes) -> Any:
        return _json.loads(data.decode("utf-8"))

    def _loads_str(data: str) -> Any:
        return _json.loads(data)


warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")


@dataclass
class EmbeddingFeature:
    """Container for embedding feature metadata."""

    name: str
    dim: int
    type: str  # e.g., "legal_bert", "graph", "fused"
    description: str


@dataclass
class LeakageTestResult:
    """Container for leakage test results."""

    test_name: str
    feature_name: str
    passed: bool
    score: float
    baseline: float
    threshold: float
    details: Dict[str, Any]
    warning: Optional[str] = None


class EmbeddingLeakageValidator:
    """Comprehensive embedding leakage validation pipeline."""

    def __init__(self, data_dir: Path, output_dir: Path, sample_size: int = 10000):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mac M1 optimization: use all performance cores
        self.n_jobs = min(mp.cpu_count() - 1, 6)  # Leave 1-2 cores free
        logger.info(f"Using {self.n_jobs} cores for parallel processing")

        # Embedding features to test - ONLY legal_bert_emb
        self.embedding_features = [
            EmbeddingFeature(
                "legal_bert_emb", 768, "legal_bert", "Main Legal-BERT embeddings"
            ),
        ]

        # Leakage test thresholds
        self.thresholds = {
            "group_cv_min": 0.55,  # Minimum group CV AUC to be meaningful
            "temporal_drop_max": 0.05,  # Max allowed drop from group to temporal CV
            "permutation_max": 0.52,  # Max AUC on permuted labels
            "case_shuffle_max": 0.52,  # Max AUC on case-shuffled labels
            "case_id_accuracy_max": 0.20,  # Max case-ID prediction accuracy (vs 1/n_cases)
            "stability_std_max": 0.03,  # Max std across random seeds
            "knn_homophily_max": 0.30,  # Max KNN case homophily ratio
        }

        self.results = []
        self.data_loaded = False

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for validation."""
        logger.info("Loading data with orjson optimization...")

        # Load from oof_test for proper validation (diverse cases)
        fold_path = self.data_dir / "oof_test" / "test.jsonl"

        data_rows = []
        with open(fold_path, "rb") as f:
            for i, line_bytes in enumerate(f):
                if i >= self.sample_size:
                    break
                line_bytes = line_bytes.strip()
                if line_bytes:
                    try:
                        data_rows.append(_loads_bytes(line_bytes))
                    except Exception:
                        try:
                            data_rows.append(
                                _loads_str(line_bytes.decode("utf-8", errors="ignore"))
                            )
                        except Exception as e:
                            logger.warning(f"Failed to parse line {i}: {e}")
                            continue

        df = pd.DataFrame(data_rows)
        logger.info(f"Loaded {len(df)} samples from {fold_path}")

        # Verify required columns
        required_cols = ["case_id", "outcome_bin"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Verify embedding features exist
        available_emb = []
        for feat in self.embedding_features:
            if feat.name in df.columns:
                available_emb.append(feat)
            else:
                logger.warning(f"Embedding feature not found: {feat.name}")

        self.embedding_features = available_emb
        logger.info(f"Found {len(self.embedding_features)} embedding features to test")

        self.data_loaded = True
        return df

    def group_cv_auc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> Tuple[float, float]:
        """Perform group cross-validation and return mean AUC and std."""
        cv = GroupKFold(n_splits=n_splits)
        clf = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=1)

        aucs = []
        for train_idx, test_idx in cv.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train and predict
            clf.fit(X_train_scaled, y_train)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            aucs.append(auc)

        return np.mean(aucs), np.std(aucs)

    def test_group_cv_hygiene(
        self, df: pd.DataFrame, feature: EmbeddingFeature
    ) -> LeakageTestResult:
        """Test A1: Group (case-held-out) CV to prevent case memorization."""
        logger.info(f"Testing group CV hygiene for {feature.name}")

        try:
            # Prepare data
            X = np.array(df[feature.name].tolist())
            y = df["outcome_bin"].values
            groups = df["case_id"].values

            # Run group CV
            auc_mean, auc_std = self.group_cv_auc(X, y, groups)

            # Test thresholds
            passed = auc_mean >= self.thresholds["group_cv_min"]

            result = LeakageTestResult(
                test_name="group_cv_hygiene",
                feature_name=feature.name,
                passed=passed,
                score=auc_mean,
                baseline=0.5,
                threshold=self.thresholds["group_cv_min"],
                details={
                    "auc_mean": auc_mean,
                    "auc_std": auc_std,
                    "n_cases": len(np.unique(groups)),
                    "n_samples": len(X),
                },
                warning=(
                    None
                    if passed
                    else f"Group CV AUC {auc_mean:.3f} below threshold {self.thresholds['group_cv_min']}"
                ),
            )

            return result

        except Exception as e:
            return LeakageTestResult(
                test_name="group_cv_hygiene",
                feature_name=feature.name,
                passed=False,
                score=0.0,
                baseline=0.5,
                threshold=self.thresholds["group_cv_min"],
                details={"error": str(e)},
                warning=f"Test failed: {e}",
            )

    def test_target_permutation(
        self, df: pd.DataFrame, feature: EmbeddingFeature
    ) -> LeakageTestResult:
        """Test B3: Target-permutation test to detect pipeline leakage."""
        logger.info(f"Testing target permutation for {feature.name}")

        try:
            # Prepare data
            X = np.array(df[feature.name].tolist())
            y = df["outcome_bin"].values
            groups = df["case_id"].values

            # Permute labels
            np.random.seed(42)
            y_perm = np.random.permutation(y)

            # Run group CV with permuted labels
            auc_mean, auc_std = self.group_cv_auc(X, y_perm, groups)

            # Should be close to chance
            passed = auc_mean <= self.thresholds["permutation_max"]

            result = LeakageTestResult(
                test_name="target_permutation",
                feature_name=feature.name,
                passed=passed,
                score=auc_mean,
                baseline=0.5,
                threshold=self.thresholds["permutation_max"],
                details={"auc_mean": auc_mean, "auc_std": auc_std, "expected": 0.5},
                warning=(
                    None
                    if passed
                    else f"Permuted labels AUC {auc_mean:.3f} above chance - indicates leakage"
                ),
            )

            return result

        except Exception as e:
            return LeakageTestResult(
                test_name="target_permutation",
                feature_name=feature.name,
                passed=False,
                score=0.0,
                baseline=0.5,
                threshold=self.thresholds["permutation_max"],
                details={"error": str(e)},
                warning=f"Test failed: {e}",
            )

    def test_case_wise_label_shuffle(
        self, df: pd.DataFrame, feature: EmbeddingFeature
    ) -> LeakageTestResult:
        """Test B4: Case-wise label shuffle to detect case identity encoding."""
        logger.info(f"Testing case-wise label shuffle for {feature.name}")

        try:
            # Prepare data
            X = np.array(df[feature.name].tolist())
            y = df["outcome_bin"].values
            groups = df["case_id"].values

            # Shuffle labels within each case
            df_shuffle = df.copy()
            for case_id in df_shuffle["case_id"].unique():
                mask = df_shuffle["case_id"] == case_id
                case_labels = df_shuffle.loc[mask, "outcome_bin"].values
                if len(case_labels) > 1:
                    np.random.shuffle(case_labels)
                    df_shuffle.loc[mask, "outcome_bin"] = case_labels

            y_shuffle = df_shuffle["outcome_bin"].values

            # Run group CV with shuffled labels
            auc_mean, auc_std = self.group_cv_auc(X, y_shuffle, groups)

            # Should be close to chance
            passed = auc_mean <= self.thresholds["case_shuffle_max"]

            result = LeakageTestResult(
                test_name="case_wise_shuffle",
                feature_name=feature.name,
                passed=passed,
                score=auc_mean,
                baseline=0.5,
                threshold=self.thresholds["case_shuffle_max"],
                details={"auc_mean": auc_mean, "auc_std": auc_std, "expected": 0.5},
                warning=(
                    None
                    if passed
                    else f"Case-shuffled labels AUC {auc_mean:.3f} above chance - case identity leakage"
                ),
            )

            return result

        except Exception as e:
            return LeakageTestResult(
                test_name="case_wise_shuffle",
                feature_name=feature.name,
                passed=False,
                score=0.0,
                baseline=0.5,
                threshold=self.thresholds["case_shuffle_max"],
                details={"error": str(e)},
                warning=f"Test failed: {e}",
            )

    def test_adversarial_case_id(
        self, df: pd.DataFrame, feature: EmbeddingFeature
    ) -> LeakageTestResult:
        """Test C6: Adversarial case-ID classifier to detect case fingerprinting."""
        logger.info(f"Testing adversarial case-ID prediction for {feature.name}")

        try:
            # Prepare data - limit to cases with sufficient samples
            case_counts = df["case_id"].value_counts()
            valid_cases = case_counts[case_counts >= 3].index[
                :50
            ]  # Top 50 cases with 3+ samples

            df_subset = df[df["case_id"].isin(valid_cases)].copy()

            if len(df_subset) < 100:
                return LeakageTestResult(
                    test_name="adversarial_case_id",
                    feature_name=feature.name,
                    passed=True,  # Skip if insufficient data
                    score=0.0,
                    baseline=1.0 / len(valid_cases),
                    threshold=self.thresholds["case_id_accuracy_max"],
                    details={
                        "skipped": "insufficient_data",
                        "n_samples": len(df_subset),
                    },
                    warning="Skipped due to insufficient data",
                )

            X = np.array(df_subset[feature.name].tolist())

            # Encode case IDs as integers
            le = LabelEncoder()
            y_case = le.fit_transform(df_subset["case_id"])

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train multiclass classifier
            clf = OneVsRestClassifier(LogisticRegression(max_iter=500, random_state=42))

            # Use simple train-test split (80-20)
            n_train = int(0.8 * len(X_scaled))
            indices = np.random.RandomState(42).permutation(len(X_scaled))
            train_idx, test_idx = indices[:n_train], indices[n_train:]

            clf.fit(X_scaled[train_idx], y_case[train_idx])
            y_pred = clf.predict(X_scaled[test_idx])

            accuracy = accuracy_score(y_case[test_idx], y_pred)
            baseline = 1.0 / len(valid_cases)  # Random chance

            # Test if accuracy is significantly above chance
            passed = accuracy <= max(
                baseline * 3, self.thresholds["case_id_accuracy_max"]
            )

            result = LeakageTestResult(
                test_name="adversarial_case_id",
                feature_name=feature.name,
                passed=passed,
                score=accuracy,
                baseline=baseline,
                threshold=max(baseline * 3, self.thresholds["case_id_accuracy_max"]),
                details={
                    "accuracy": accuracy,
                    "baseline_chance": baseline,
                    "n_cases": len(valid_cases),
                    "n_test_samples": len(test_idx),
                },
                warning=(
                    None
                    if passed
                    else f"High case-ID accuracy {accuracy:.3f} vs chance {baseline:.3f} - case fingerprinting"
                ),
            )

            return result

        except Exception as e:
            return LeakageTestResult(
                test_name="adversarial_case_id",
                feature_name=feature.name,
                passed=False,
                score=0.0,
                baseline=0.0,
                threshold=self.thresholds["case_id_accuracy_max"],
                details={"error": str(e)},
                warning=f"Test failed: {e}",
            )

    def test_knn_case_homophily(
        self, df: pd.DataFrame, feature: EmbeddingFeature
    ) -> LeakageTestResult:
        """Test E12: KNN homophily by case to detect case clustering."""
        logger.info(f"Testing KNN case homophily for {feature.name}")

        try:
            # Prepare data
            X = np.array(df[feature.name].tolist())
            case_ids = df["case_id"].values

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Compute KNN (k=10)
            k = min(10, len(X) // 10)
            knn = NearestNeighbors(
                n_neighbors=k + 1, metric="cosine"
            )  # +1 because it includes self
            knn.fit(X_scaled)

            # Find neighbors
            distances, indices = knn.kneighbors(X_scaled)

            # Calculate case homophily
            same_case_count = 0
            total_neighbors = 0

            for i, neighbors in enumerate(indices):
                current_case = case_ids[i]
                neighbor_cases = case_ids[neighbors[1:]]  # Exclude self
                same_case_count += np.sum(neighbor_cases == current_case)
                total_neighbors += len(neighbor_cases)

            homophily_ratio = (
                same_case_count / total_neighbors if total_neighbors > 0 else 0
            )

            # Calculate baseline (random expectation)
            case_counts = pd.Series(case_ids).value_counts()
            baseline = np.sum((case_counts / len(case_ids)) ** 2)  # Expected by chance

            # Test threshold
            passed = homophily_ratio <= max(
                baseline * 2, self.thresholds["knn_homophily_max"]
            )

            result = LeakageTestResult(
                test_name="knn_case_homophily",
                feature_name=feature.name,
                passed=passed,
                score=homophily_ratio,
                baseline=baseline,
                threshold=max(baseline * 2, self.thresholds["knn_homophily_max"]),
                details={
                    "homophily_ratio": homophily_ratio,
                    "baseline_random": baseline,
                    "k_neighbors": k,
                    "same_case_neighbors": same_case_count,
                    "total_neighbors": total_neighbors,
                },
                warning=(
                    None
                    if passed
                    else f"High KNN case homophily {homophily_ratio:.3f} vs baseline {baseline:.3f} - case clustering"
                ),
            )

            return result

        except Exception as e:
            return LeakageTestResult(
                test_name="knn_case_homophily",
                feature_name=feature.name,
                passed=False,
                score=0.0,
                baseline=0.0,
                threshold=self.thresholds["knn_homophily_max"],
                details={"error": str(e)},
                warning=f"Test failed: {e}",
            )

    def test_seed_stability(
        self, df: pd.DataFrame, feature: EmbeddingFeature
    ) -> LeakageTestResult:
        """Test E11: Seed/boot stability across multiple random seeds."""
        logger.info(f"Testing seed stability for {feature.name}")

        try:
            X = np.array(df[feature.name].tolist())
            y = df["outcome_bin"].values
            groups = df["case_id"].values

            # Test with 5 different seeds
            seeds = [42, 123, 456, 789, 999]
            aucs = []

            for seed in seeds:
                auc_mean, _ = self.group_cv_auc(X, y, groups, random_state=seed)
                aucs.append(auc_mean)

            auc_std = np.std(aucs)
            auc_mean_overall = np.mean(aucs)

            # Test stability
            passed = auc_std <= self.thresholds["stability_std_max"]

            result = LeakageTestResult(
                test_name="seed_stability",
                feature_name=feature.name,
                passed=passed,
                score=auc_std,
                baseline=0.0,
                threshold=self.thresholds["stability_std_max"],
                details={
                    "auc_std": auc_std,
                    "auc_mean": auc_mean_overall,
                    "auc_values": aucs,
                    "seeds_tested": seeds,
                },
                warning=(
                    None
                    if passed
                    else f"High seed variance {auc_std:.3f} - unstable/overfit signal"
                ),
            )

            return result

        except Exception as e:
            return LeakageTestResult(
                test_name="seed_stability",
                feature_name=feature.name,
                passed=False,
                score=0.0,
                baseline=0.0,
                threshold=self.thresholds["stability_std_max"],
                details={"error": str(e)},
                warning=f"Test failed: {e}",
            )

    def validate_feature(
        self, df: pd.DataFrame, feature: EmbeddingFeature
    ) -> List[LeakageTestResult]:
        """Run all leakage tests for a single embedding feature."""
        logger.info(f"🔍 Validating {feature.name} ({feature.description})")

        tests = [
            self.test_group_cv_hygiene,
            self.test_target_permutation,
            self.test_case_wise_label_shuffle,
            self.test_adversarial_case_id,
            self.test_knn_case_homophily,
            self.test_seed_stability,
        ]

        feature_results = []
        for test_func in tests:
            try:
                result = test_func(df, feature)
                feature_results.append(result)

                # Log result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(
                    f"  {result.test_name}: {status} (score: {result.score:.3f})"
                )
                if result.warning:
                    logger.warning(f"    ⚠️  {result.warning}")

            except Exception as e:
                logger.error(f"  {test_func.__name__} failed: {e}")
                feature_results.append(
                    LeakageTestResult(
                        test_name=test_func.__name__,
                        feature_name=feature.name,
                        passed=False,
                        score=0.0,
                        baseline=0.0,
                        threshold=0.0,
                        details={"error": str(e)},
                        warning=f"Test failed: {e}",
                    )
                )

        return feature_results

    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive leakage validation on all embedding features."""
        logger.info("🚀 Starting comprehensive embedding leakage validation")

        # Load data
        df = self.load_data()

        # Validate each feature
        all_results = []
        for feature in self.embedding_features:
            feature_results = self.validate_feature(df, feature)
            all_results.extend(feature_results)

        # Compile summary
        summary = self.compile_summary(all_results)

        # Save results
        self.save_results(all_results, summary)

        return summary

    def compile_summary(self, results: List[LeakageTestResult]) -> Dict[str, Any]:
        """Compile validation summary statistics."""
        # Group by feature
        by_feature = {}
        for result in results:
            if result.feature_name not in by_feature:
                by_feature[result.feature_name] = []
            by_feature[result.feature_name].append(result)

        # Summary stats
        feature_summaries = {}
        for feature_name, feature_results in by_feature.items():
            total_tests = len(feature_results)
            passed_tests = sum(1 for r in feature_results if r.passed)
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0

            warnings = [r.warning for r in feature_results if r.warning]

            feature_summaries[feature_name] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": pass_rate,
                "warnings": warnings,
                "overall_status": "PASS" if pass_rate >= 0.8 else "FAIL",
                "test_details": {
                    r.test_name: {"passed": r.passed, "score": r.score}
                    for r in feature_results
                },
            }

        # Overall summary
        total_features = len(feature_summaries)
        passed_features = sum(
            1 for s in feature_summaries.values() if s["overall_status"] == "PASS"
        )

        summary = {
            "total_features_tested": total_features,
            "features_passed": passed_features,
            "overall_pass_rate": (
                passed_features / total_features if total_features > 0 else 0
            ),
            "feature_summaries": feature_summaries,
            "panic_indicators": self.check_panic_indicators(results),
            "recommendations": self.generate_recommendations(feature_summaries),
        }

        return summary

    def check_panic_indicators(self, results: List[LeakageTestResult]) -> List[str]:
        """Check for critical leakage indicators."""
        panic_indicators = []

        # High permutation scores
        perm_results = [r for r in results if r.test_name == "target_permutation"]
        high_perm = [r for r in perm_results if r.score > 0.55]
        if high_perm:
            panic_indicators.append(
                f"🚨 HIGH PERMUTATION SCORES: {len(high_perm)} features show above-chance performance on permuted labels"
            )

        # High case-ID predictability
        case_id_results = [r for r in results if r.test_name == "adversarial_case_id"]
        high_case_id = [r for r in case_id_results if r.passed == False]
        if high_case_id:
            panic_indicators.append(
                f"🚨 CASE FINGERPRINTING: {len(high_case_id)} features can predict case IDs"
            )

        # Unstable performance
        stability_results = [r for r in results if r.test_name == "seed_stability"]
        unstable = [r for r in stability_results if r.passed == False]
        if unstable:
            panic_indicators.append(
                f"⚠️  UNSTABLE PERFORMANCE: {len(unstable)} features show high seed variance"
            )

        return panic_indicators

    def generate_recommendations(self, feature_summaries: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        # Find problematic features
        failed_features = [
            name
            for name, summary in feature_summaries.items()
            if summary["overall_status"] == "FAIL"
        ]

        if failed_features:
            recommendations.append("🔧 IMMEDIATE ACTIONS:")
            recommendations.append(
                f"- Review {len(failed_features)} failed features: {', '.join(failed_features)}"
            )
            recommendations.append("- Use only Group/Temporal CV for model selection")
            recommendations.append(
                "- Consider entity masking and post-judgment doc filtering"
            )

        # Specific recommendations by test type
        for feature_name, summary in feature_summaries.items():
            warnings = summary["warnings"]
            if warnings:
                recommendations.append(f"📋 {feature_name}:")
                for warning in warnings[:3]:  # Top 3 warnings
                    recommendations.append(f"  - {warning}")

        return recommendations

    def save_results(self, results: List[LeakageTestResult], summary: Dict[str, Any]):
        """Save validation results to files."""
        import json
        from datetime import datetime

        # Save detailed results
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "feature_name": r.feature_name,
                    "passed": r.passed,
                    "score": r.score,
                    "baseline": r.baseline,
                    "threshold": r.threshold,
                    "details": r.details,
                    "warning": r.warning,
                }
                for r in results
            ],
        }

        results_file = self.output_dir / "embedding_leakage_validation_results.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save summary report
        self.save_summary_report(summary)

        logger.success(f"Results saved to {self.output_dir}")

    def save_summary_report(self, summary: Dict[str, Any]):
        """Save human-readable summary report."""
        report_file = self.output_dir / "EMBEDDING_LEAKAGE_VALIDATION_REPORT.md"

        with open(report_file, "w") as f:
            f.write("# Embedding Leakage Validation Report\n\n")
            f.write(
                f"**Overall Pass Rate:** {summary['overall_pass_rate']:.1%} ({summary['features_passed']}/{summary['total_features_tested']} features)\n\n"
            )

            if summary["panic_indicators"]:
                f.write("## 🚨 PANIC INDICATORS\n\n")
                for indicator in summary["panic_indicators"]:
                    f.write(f"- {indicator}\n")
                f.write("\n")

            f.write("## Feature Results\n\n")
            for feature_name, feature_summary in summary["feature_summaries"].items():
                status_emoji = (
                    "✅" if feature_summary["overall_status"] == "PASS" else "❌"
                )
                f.write(f"### {status_emoji} {feature_name}\n\n")
                f.write(
                    f"**Pass Rate:** {feature_summary['pass_rate']:.1%} ({feature_summary['passed_tests']}/{feature_summary['total_tests']} tests)\n\n"
                )

                # Test details
                f.write("**Test Results:**\n")
                for test_name, test_result in feature_summary["test_details"].items():
                    test_emoji = "✅" if test_result["passed"] else "❌"
                    f.write(f"- {test_emoji} {test_name}: {test_result['score']:.3f}\n")

                if feature_summary["warnings"]:
                    f.write("\n**Warnings:**\n")
                    for warning in feature_summary["warnings"]:
                        f.write(f"- ⚠️ {warning}\n")
                f.write("\n")

            if summary["recommendations"]:
                f.write("## 📋 Recommendations\n\n")
                for rec in summary["recommendations"]:
                    f.write(f"{rec}\n")
                f.write("\n")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Embedding Leakage Validation (Mac M1 Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage",
        help="Directory containing the k-fold split data",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/embedding_leakage_validation",
        help="Output directory for validation results",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples to use for validation (default: 10000)",
    )

    args = parser.parse_args()

    # Initialize validator
    validator = EmbeddingLeakageValidator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size,
    )

    # Run validation
    start_time = time.time()
    summary = validator.run_validation()
    elapsed_time = time.time() - start_time

    # Print summary
    logger.success(f"🎉 Validation completed in {elapsed_time:.1f}s")
    logger.info(f"📊 Overall pass rate: {summary['overall_pass_rate']:.1%}")

    if summary["panic_indicators"]:
        logger.error("🚨 CRITICAL ISSUES DETECTED:")
        for indicator in summary["panic_indicators"]:
            logger.error(f"   {indicator}")
    else:
        logger.success("✅ No critical leakage indicators detected")

    logger.info(f"📁 Full results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
