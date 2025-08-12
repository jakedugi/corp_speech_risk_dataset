"""Validation experiments for proving model validity and interpretability.

Includes negative controls, feature ablation, and case-level aggregation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from loguru import logger
import json

from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


class ValidationExperiments:
    """Run validation experiments to prove model captures true risk signals."""

    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        output_dir: Union[str, Path],
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all_experiments(
        self,
        data: List[Dict[str, Any]],
        labels: np.ndarray,
        case_outcomes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive validation experiments."""
        results = {}

        # 1. Negative control experiment
        logger.info("Running negative control experiment...")
        neg_control = self.negative_control_experiment(data, labels)
        results["negative_control"] = neg_control
        self._plot_negative_control(neg_control)

        # 2. Feature ablation study
        logger.info("Running feature ablation study...")
        ablation = self.feature_ablation_study(data, labels)
        results["feature_ablation"] = ablation
        self._plot_ablation_study(ablation)

        # 3. Case-level aggregation
        if case_outcomes:
            logger.info("Analyzing case-level aggregation...")
            case_analysis = self.case_level_analysis(data, labels, case_outcomes)
            results["case_analysis"] = case_analysis
            self._plot_case_analysis(case_analysis)

        # 4. Permutation importance
        logger.info("Computing permutation importance...")
        perm_importance = self.permutation_importance_analysis(data, labels)
        results["permutation_importance"] = perm_importance
        self._plot_permutation_importance(perm_importance)

        # Save all results
        with open(self.output_dir / "validation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def negative_control_experiment(
        self,
        data: List[Dict[str, Any]],
        labels: np.ndarray,
        n_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Replace quotes with random document spans to test content specificity."""
        from sklearn.model_selection import cross_val_score

        # Extract features for original data
        X_original = self._extract_features(data)

        # Get baseline performance
        baseline_scores = cross_val_score(
            self.model, X_original, labels, cv=5, scoring="accuracy"
        )
        baseline_qwk = cross_val_score(
            self.model,
            X_original,
            labels,
            cv=5,
            scoring=lambda est, X, y: cohen_kappa_score(
                y, est.predict(X), weights="quadratic"
            ),
        )

        # Run negative control iterations
        control_scores = []
        control_qwks = []

        for i in range(n_iterations):
            logger.info(f"Negative control iteration {i+1}/{n_iterations}")

            # Create control data by replacing quotes with random spans
            control_data = self._create_negative_control_data(data)
            X_control = self._extract_features(control_data)

            # Evaluate on control data
            scores = cross_val_score(
                self.model, X_control, labels, cv=5, scoring="accuracy"
            )
            qwks = cross_val_score(
                self.model,
                X_control,
                labels,
                cv=5,
                scoring=lambda est, X, y: cohen_kappa_score(
                    y, est.predict(X), weights="quadratic"
                ),
            )

            control_scores.append(scores.mean())
            control_qwks.append(qwks.mean())

        # Statistical test
        baseline_mean = baseline_scores.mean()
        control_mean = np.mean(control_scores)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(
            [baseline_mean] * len(control_scores), control_scores
        )

        results = {
            "baseline_accuracy": baseline_mean,
            "baseline_qwk": baseline_qwk.mean(),
            "control_accuracy_mean": control_mean,
            "control_accuracy_std": np.std(control_scores),
            "control_qwk_mean": np.mean(control_qwks),
            "control_qwk_std": np.std(control_qwks),
            "accuracy_drop": baseline_mean - control_mean,
            "qwk_drop": baseline_qwk.mean() - np.mean(control_qwks),
            "significance_test": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            },
            "interpretation": (
                "Model performance drops significantly on random text, "
                "confirming it captures quote-specific risk signals."
                if p_value < 0.05
                else "No significant difference found - model may not be capturing "
                "quote-specific signals."
            ),
        }

        return results

    def _create_negative_control_data(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Replace quotes with random spans from same documents."""
        control_data = []

        for record in data:
            control_record = record.copy()

            # Get full document text if available
            doc_text = record.get("full_text", record.get("context", ""))
            quote_len = len(record.get("text", "").split())

            if doc_text and quote_len > 0:
                # Extract random span of same length
                words = doc_text.split()
                if len(words) > quote_len:
                    start_idx = np.random.randint(0, len(words) - quote_len)
                    random_span = " ".join(words[start_idx : start_idx + quote_len])
                    control_record["text"] = random_span

            control_data.append(control_record)

        return control_data

    def feature_ablation_study(
        self,
        data: List[Dict[str, Any]],
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Test impact of removing feature groups."""
        # Baseline with all features
        X_full = self._extract_features(data)
        baseline_score = cross_val_score(
            self.model,
            X_full,
            labels,
            cv=5,
            scoring=lambda est, X, y: cohen_kappa_score(
                y, est.predict(X), weights="quadratic"
            ),
        ).mean()

        # Test different feature configurations
        ablation_configs = [
            ("no_lexicons", {"include_lexicons": False}),
            ("no_sequence", {"include_sequence": False}),
            ("no_linguistic", {"include_linguistic": False}),
            ("no_structural", {"include_structural": False}),
            (
                "lexicons_only",
                {
                    "include_lexicons": True,
                    "include_sequence": False,
                    "include_linguistic": False,
                    "include_structural": False,
                },
            ),
            (
                "sequence_only",
                {
                    "include_lexicons": False,
                    "include_sequence": True,
                    "include_linguistic": False,
                    "include_structural": False,
                },
            ),
        ]

        results = {
            "baseline_qwk": baseline_score,
            "ablations": {},
        }

        for name, config in ablation_configs:
            logger.info(f"Testing ablation: {name}")

            # Create modified feature extractor
            modified_extractor = type(self.feature_extractor)(**config)
            X_ablated = self._extract_features(data, modified_extractor)

            # Evaluate
            scores = cross_val_score(
                self.model,
                X_ablated,
                labels,
                cv=5,
                scoring=lambda est, X, y: cohen_kappa_score(
                    y, est.predict(X), weights="quadratic"
                ),
            )

            results["ablations"][name] = {
                "qwk_mean": scores.mean(),
                "qwk_std": scores.std(),
                "qwk_drop": baseline_score - scores.mean(),
                "relative_drop": (baseline_score - scores.mean())
                / baseline_score
                * 100,
            }

        # Rank features by importance
        ranked = sorted(
            results["ablations"].items(),
            key=lambda x: x[1]["qwk_drop"],
            reverse=True,
        )
        results["feature_importance_ranking"] = [
            {"feature_group": name, **data} for name, data in ranked
        ]

        return results

    def case_level_analysis(
        self,
        data: List[Dict[str, Any]],
        labels: np.ndarray,
        case_outcomes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze how quote-level predictions aggregate to case outcomes."""
        # Get predictions
        X = self._extract_features(data)
        predictions = self.model.predict(X)
        pred_proba = self.model.predict_proba(X)

        # Map to risk scores
        risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
        risk_scores = [risk_mapping.get(pred, 1) for pred in predictions]

        # Group by case
        case_aggregates = {}
        for i, record in enumerate(data):
            case_id = record.get("case_id", f"case_{i}")
            if case_id not in case_aggregates:
                case_aggregates[case_id] = {
                    "risk_scores": [],
                    "predictions": [],
                    "probabilities": [],
                }
            case_aggregates[case_id]["risk_scores"].append(risk_scores[i])
            case_aggregates[case_id]["predictions"].append(predictions[i])
            case_aggregates[case_id]["probabilities"].append(pred_proba[i])

        # Compute case-level metrics
        case_metrics = []
        for case_id, agg in case_aggregates.items():
            scores = agg["risk_scores"]
            case_metric = {
                "case_id": case_id,
                "n_quotes": len(scores),
                "mean_risk": np.mean(scores),
                "max_risk": np.max(scores),
                "high_risk_ratio": sum(s == 2 for s in scores) / len(scores),
                "outcome": case_outcomes.get(case_id, {}).get("severity", None),
                "amount": case_outcomes.get(case_id, {}).get("amount", None),
            }
            case_metrics.append(case_metric)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(case_metrics)
        df = df[df["outcome"].notna()]  # Filter to cases with known outcomes

        if len(df) == 0:
            return {"error": "No cases with outcome data found"}

        # Compute correlations
        correlations = {}
        for metric in ["mean_risk", "max_risk", "high_risk_ratio"]:
            if "amount" in df.columns and df["amount"].notna().sum() > 5:
                corr, p_value = stats.spearmanr(
                    df[metric].fillna(0), df["amount"].fillna(0)
                )
                correlations[f"{metric}_vs_amount"] = {
                    "correlation": corr,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }

        results = {
            "n_cases": len(df),
            "correlations": correlations,
            "case_metrics": case_metrics[:10],  # Sample for inspection
            "aggregate_stats": {
                "mean_quotes_per_case": df["n_quotes"].mean(),
                "mean_risk_by_outcome": df.groupby("outcome")["mean_risk"]
                .mean()
                .to_dict(),
            },
        }

        return results

    def permutation_importance_analysis(
        self,
        data: List[Dict[str, Any]],
        labels: np.ndarray,
        n_repeats: int = 10,
    ) -> Dict[str, Any]:
        """Analyze feature importance via permutation."""
        from sklearn.inspection import permutation_importance

        # Extract features
        X = self._extract_features(data)
        feature_names = list(data[0].keys()) if data else []

        # Fit model if needed
        self.model.fit(X, labels)

        # Compute permutation importance
        perm_imp = permutation_importance(
            self.model,
            X,
            labels,
            n_repeats=n_repeats,
            random_state=42,
            scoring=lambda est, X, y: cohen_kappa_score(
                y, est.predict(X), weights="quadratic"
            ),
        )

        # Create sorted results
        importances = perm_imp.importances_mean
        indices = np.argsort(importances)[::-1]

        results = {
            "top_features": [
                {
                    "rank": i + 1,
                    "feature": (
                        feature_names[idx]
                        if idx < len(feature_names)
                        else f"feature_{idx}"
                    ),
                    "importance": importances[idx],
                    "std": perm_imp.importances_std[idx],
                }
                for i, idx in enumerate(indices[:20])
            ],
            "total_features": len(importances),
        }

        return results

    def _extract_features(
        self,
        data: List[Dict[str, Any]],
        extractor: Optional[Any] = None,
    ) -> np.ndarray:
        """Extract features from data using feature extractor."""
        if extractor is None:
            extractor = self.feature_extractor

        # This is a placeholder - integrate with actual feature extraction
        feature_list = []
        for record in data:
            features = extractor.extract_features(
                record.get("text", ""), record.get("context", "")
            )
            feature_list.append(features)

        # Convert to matrix
        if feature_list:
            feature_names = sorted(set(k for f in feature_list for k in f))
            X = np.zeros((len(feature_list), len(feature_names)))
            for i, features in enumerate(feature_list):
                for j, name in enumerate(feature_names):
                    X[i, j] = features.get(name, 0)
            return X
        else:
            return np.array([])

    def _plot_negative_control(self, results: Dict[str, Any]):
        """Plot negative control experiment results."""
        fig, ax = plt.subplots(figsize=(8, 6))

        metrics = ["Accuracy", "QWK (Quadratic Weighted Kappa)"]
        baseline = [results["baseline_accuracy"], results["baseline_qwk"]]
        control = [results["control_accuracy_mean"], results["control_qwk_mean"]]
        control_std = [results["control_accuracy_std"], results["control_qwk_std"]]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, baseline, width, label="Original Quotes", alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2,
            control,
            width,
            yerr=control_std,
            label="Random Spans",
            alpha=0.8,
            capsize=5,
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title(
            "Negative Control Experiment:\nModel Performance on Quotes vs Random Text"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add significance annotation
        if results["significance_test"]["significant"]:
            ax.text(
                0.5,
                0.95,
                f"p < 0.05 (significant drop)",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / "negative_control_experiment.pdf")
        plt.close()

    def _plot_ablation_study(self, results: Dict[str, Any]):
        """Plot feature ablation results."""
        # Create DataFrame for plotting
        ablation_data = []
        for name, data in results["ablations"].items():
            ablation_data.append(
                {
                    "Configuration": name.replace("_", " ").title(),
                    "QWK": data["qwk_mean"],
                    "Drop": data["relative_drop"],
                }
            )

        df = pd.DataFrame(ablation_data)
        df = df.sort_values("Drop", ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # QWK scores
        bars1 = ax1.bar(df["Configuration"], df["QWK"], alpha=0.7)
        ax1.axhline(
            y=results["baseline_qwk"],
            color="red",
            linestyle="--",
            label=f"Baseline ({results['baseline_qwk']:.3f})",
        )
        ax1.set_xlabel("Feature Configuration")
        ax1.set_ylabel("Quadratic Weighted Kappa")
        ax1.set_title("Model Performance with Feature Ablation")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Relative drops
        colors = [
            "red" if d > 10 else "orange" if d > 5 else "green" for d in df["Drop"]
        ]
        bars2 = ax2.bar(df["Configuration"], df["Drop"], color=colors, alpha=0.7)
        ax2.set_xlabel("Feature Configuration")
        ax2.set_ylabel("Performance Drop (%)")
        ax2.set_title("Relative Performance Drop from Baseline")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_ablation_study.pdf")
        plt.close()

    def _plot_case_analysis(self, results: Dict[str, Any]):
        """Plot case-level analysis results."""
        if "error" in results:
            return

        # Create correlation heatmap if available
        if "correlations" in results and results["correlations"]:
            corr_data = []
            for metric, data in results["correlations"].items():
                parts = metric.split("_vs_")
                corr_data.append(
                    {
                        "Risk Metric": parts[0].replace("_", " ").title(),
                        "Outcome": parts[1].title(),
                        "Correlation": data["correlation"],
                        "Significant": data["significant"],
                    }
                )

            df = pd.DataFrame(corr_data)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Create bar plot
            bars = ax.bar(
                df["Risk Metric"],
                df["Correlation"],
                color=["green" if s else "gray" for s in df["Significant"]],
                alpha=0.7,
            )

            # Add significance markers
            for i, (bar, sig) in enumerate(zip(bars, df["Significant"])):
                if sig:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.01,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=16,
                    )

            ax.set_xlabel("Case-Level Risk Metric")
            ax.set_ylabel("Spearman Correlation with Outcome")
            ax.set_title("Case-Level Risk Aggregation vs Outcomes")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(-0.5, 0.8)

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="green", alpha=0.7, label="Significant (p < 0.05)"),
                Patch(facecolor="gray", alpha=0.7, label="Not Significant"),
            ]
            ax.legend(handles=legend_elements)

            plt.tight_layout()
            plt.savefig(self.output_dir / "case_level_correlation.pdf")
            plt.close()

    def _plot_permutation_importance(self, results: Dict[str, Any]):
        """Plot permutation importance results."""
        # Get top features
        top_features = results["top_features"][:15]

        fig, ax = plt.subplots(figsize=(10, 8))

        features = [f["feature"] for f in top_features]
        importances = [f["importance"] for f in top_features]
        stds = [f["std"] for f in top_features]

        y_pos = np.arange(len(features))

        ax.barh(y_pos, importances, xerr=stds, alpha=0.7, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Permutation Importance (QWK Drop)")
        ax.set_title("Top 15 Features by Permutation Importance")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(self.output_dir / "permutation_importance.pdf")
        plt.close()
