"""Interpretation and visualization for fully interpretable models.

Generates publication-ready figures and explanations for academic papers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from loguru import logger

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    mean_absolute_error,
)

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.utils import resample


class InterpretabilityReport:
    """Generate comprehensive interpretability reports and figures."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_names: List[str],
        output_dir: Union[str, Path],
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set publication-quality defaults
        plt.rcParams["font.size"] = 12
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"

    def generate_full_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Generate all interpretability analyses and figures."""
        report = {}

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # 1. Global feature importance
        logger.info("Generating feature importance plots...")
        feature_importance = self._extract_feature_importance()
        self._plot_feature_importance_forest(feature_importance)
        report["feature_importance"] = feature_importance

        # 2. Confusion matrix
        logger.info("Creating confusion matrix...")
        self._plot_confusion_matrix(y_test, y_pred)

        # 3. Calibration plots
        logger.info("Generating calibration curves...")
        self._plot_calibration_curves(y_test, y_proba)

        # 4. Performance metrics with confidence intervals
        logger.info("Computing metrics with bootstrap CIs...")
        metrics = self._compute_metrics_with_ci(y_test, y_pred, y_proba)
        report["metrics"] = metrics
        self._save_metrics_table(metrics)

        # 5. Local explanations for selected instances
        logger.info("Generating local explanations...")
        local_explanations = self._generate_local_explanations(
            X_test, y_test, y_pred, n_samples=10
        )
        self._plot_local_explanations(local_explanations)
        report["local_explanations"] = local_explanations

        # 6. Error analysis
        logger.info("Performing error analysis...")
        error_analysis = self._analyze_errors(X_test, y_test, y_pred)
        self._plot_error_analysis(error_analysis)
        report["error_analysis"] = error_analysis

        # 7. Feature interaction analysis (for tree/EBM models)
        if hasattr(self.model, "feature_importances_") or hasattr(
            self.model, "explain_global"
        ):
            logger.info("Analyzing feature interactions...")
            interactions = self._analyze_feature_interactions()
            if interactions:
                self._plot_feature_interactions(interactions)
                report["feature_interactions"] = interactions

        return report

    def _extract_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from various model types."""
        importances = None

        # Linear models (coefficients)
        if hasattr(self.model, "coef_"):
            if self.model.coef_.ndim == 1:
                importances = self.model.coef_
            else:
                # Multi-class: average absolute coefficients
                importances = np.mean(np.abs(self.model.coef_), axis=0)

        # Tree-based models
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

        # EBM models
        elif hasattr(self.model, "explain_global"):
            global_explanation = self.model.explain_global()
            importances = global_explanation.data()["scores"]

        # Ensemble models
        elif hasattr(self.model, "estimators_"):
            # Average importance across estimators
            all_importances = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, "coef_"):
                    all_importances.append(np.abs(estimator.coef_).ravel())
                elif hasattr(estimator, "feature_importances_"):
                    all_importances.append(estimator.feature_importances_)
            if all_importances:
                importances = np.mean(all_importances, axis=0)

        if importances is None:
            logger.warning("Could not extract feature importances")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
                "abs_importance": np.abs(importances),
            }
        )

        # Add standard errors if available
        if hasattr(self.model, "coef_std_"):
            df["std_error"] = self.model.coef_std_
            df["ci_lower"] = df["importance"] - 1.96 * df["std_error"]
            df["ci_upper"] = df["importance"] + 1.96 * df["std_error"]

        return df.sort_values("abs_importance", ascending=False)

    def _plot_feature_importance_forest(self, importance_df: pd.DataFrame):
        """Create forest plot of feature importances with confidence intervals."""
        # Select top features
        top_n = 25
        df = importance_df.head(top_n).sort_values("importance")

        fig, ax = plt.subplots(figsize=(8, 10))

        # Plot coefficients
        y_pos = np.arange(len(df))
        ax.scatter(df["importance"], y_pos, color="black", s=50, zorder=3)

        # Add confidence intervals if available
        if "ci_lower" in df.columns:
            for i, (_, row) in enumerate(df.iterrows()):
                ax.plot(
                    [row["ci_lower"], row["ci_upper"]],
                    [i, i],
                    "k-",
                    linewidth=1.5,
                    zorder=2,
                )

        # Add vertical line at 0
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["feature"])
        ax.set_xlabel("Feature Importance (Coefficient/Weight)")
        ax.set_title(f"Top {top_n} Features by Importance")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance_forest.pdf")
        plt.close()

        # Also create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 10))
        colors = ["red" if x < 0 else "blue" for x in df["importance"]]
        ax.barh(y_pos, df["importance"], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["feature"])
        ax.set_xlabel("Feature Importance")
        ax.set_title("Feature Importance by Direction")
        ax.axvline(x=0, color="black", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance_bars.pdf")
        plt.close()

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Create publication-quality confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax1,
            cbar_kws={"label": "Count"},
        )
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix (Counts)")

        # Normalized
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax2,
            cbar_kws={"label": "Proportion"},
        )
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_title("Normalized Confusion Matrix")

        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.pdf")
        plt.close()

    def _plot_calibration_curves(self, y_true: np.ndarray, y_proba: np.ndarray):
        """Plot calibration curves for each class."""
        n_classes = y_proba.shape[1]

        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
        if n_classes == 1:
            axes = [axes]

        for idx, (ax, class_name) in enumerate(zip(axes, self.class_names)):
            # Binary indicators for this class
            y_binary = (y_true == idx).astype(int)
            y_score = y_proba[:, idx]

            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_score, n_bins=10
            )

            # Plot
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker="o",
                linewidth=2,
                label="Model",
            )
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

            # Formatting
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(f"Calibration: {class_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / "calibration_curves.pdf")
        plt.close()

        # Overall calibration plot
        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, class_name in enumerate(self.class_names):
            y_binary = (y_true == idx).astype(int)
            y_score = y_proba[:, idx]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_score, n_bins=10
            )
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker="o",
                linewidth=2,
                label=class_name,
            )

        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curves (All Classes)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "calibration_curves_combined.pdf")
        plt.close()

    def _compute_metrics_with_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Compute metrics with bootstrap confidence intervals."""
        # Convert to numeric for ordinal metrics
        class_to_ordinal = {cls: i for i, cls in enumerate(self.class_names)}
        y_true_ord = np.array([class_to_ordinal.get(y, y) for y in y_true])
        y_pred_ord = np.array([class_to_ordinal.get(y, y) for y in y_pred])

        # Base metrics
        base_metrics = {
            "accuracy": (y_true == y_pred).mean(),
            "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "mae": mean_absolute_error(y_true_ord, y_pred_ord),
        }

        # Bootstrap for confidence intervals
        bootstrap_metrics = {metric: [] for metric in base_metrics}

        for _ in range(n_bootstrap):
            # Resample
            indices = resample(range(len(y_true)), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_true_ord_boot = y_true_ord[indices]
            y_pred_ord_boot = y_pred_ord[indices]

            # Compute metrics
            bootstrap_metrics["accuracy"].append((y_true_boot == y_pred_boot).mean())
            bootstrap_metrics["qwk"].append(
                cohen_kappa_score(y_true_boot, y_pred_boot, weights="quadratic")
            )
            bootstrap_metrics["mae"].append(
                mean_absolute_error(y_true_ord_boot, y_pred_ord_boot)
            )

        # Compute confidence intervals
        alpha = 1 - confidence
        metrics_with_ci = {}
        for metric, values in bootstrap_metrics.items():
            values = np.array(values)
            metrics_with_ci[metric] = {
                "value": base_metrics[metric],
                "ci_lower": np.percentile(values, 100 * alpha / 2),
                "ci_upper": np.percentile(values, 100 * (1 - alpha / 2)),
                "std": np.std(values),
            }

        # Add per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics_with_ci["per_class"] = report

        return metrics_with_ci

    def _save_metrics_table(self, metrics: Dict[str, Any]):
        """Save metrics as LaTeX table for publication."""
        # Create DataFrame
        rows = []
        for metric, data in metrics.items():
            if metric == "per_class":
                continue
            row = {
                "Metric": metric.upper(),
                "Value": f"{data['value']:.3f}",
                "CI": f"[{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]",
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save as LaTeX
        latex = df.to_latex(index=False, escape=False)
        with open(self.output_dir / "metrics_table.tex", "w") as f:
            f.write(latex)

        # Also save as CSV
        df.to_csv(self.output_dir / "metrics_table.csv", index=False)

    def _generate_local_explanations(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate local explanations for selected samples."""
        explanations = []

        # Select diverse samples
        correct_indices = np.where(y_true == y_pred)[0]
        incorrect_indices = np.where(y_true != y_pred)[0]

        # Get samples from each class and error type
        selected_indices = []

        # Correct predictions from each class
        for class_val in np.unique(y_true):
            class_correct = correct_indices[y_true[correct_indices] == class_val]
            if len(class_correct) > 0:
                selected_indices.extend(
                    np.random.choice(
                        class_correct, min(3, len(class_correct)), replace=False
                    )
                )

        # Incorrect predictions
        if len(incorrect_indices) > 0:
            selected_indices.extend(
                np.random.choice(
                    incorrect_indices, min(4, len(incorrect_indices)), replace=False
                )
            )

        # Generate explanations
        for idx in selected_indices[:n_samples]:
            explanation = self._explain_single_prediction(
                X[idx], y_true[idx], y_pred[idx]
            )
            explanations.append(explanation)

        return explanations

    def _explain_single_prediction(
        self,
        x: np.ndarray,
        y_true: Any,
        y_pred: Any,
    ) -> Dict[str, Any]:
        """Explain a single prediction."""
        # Get feature contributions
        if hasattr(self.model, "coef_"):
            # Linear model
            if self.model.coef_.ndim == 1:
                contributions = x * self.model.coef_
            else:
                # Multi-class: use coefficients for predicted class
                pred_idx = list(self.class_names).index(y_pred)
                contributions = x * self.model.coef_[pred_idx]
        else:
            # For other models, use feature values weighted by importance
            importance = self._extract_feature_importance()
            if not importance.empty:
                contributions = x * importance["importance"].values
            else:
                contributions = x

        # Sort by absolute contribution
        top_indices = np.argsort(np.abs(contributions))[-10:][::-1]

        explanation = {
            "true_label": y_true,
            "predicted_label": y_pred,
            "correct": y_true == y_pred,
            "top_features": [
                {
                    "name": self.feature_names[i],
                    "value": x[i],
                    "contribution": contributions[i],
                }
                for i in top_indices
            ],
        }

        return explanation

    def _plot_local_explanations(self, explanations: List[Dict[str, Any]]):
        """Create visualizations for local explanations."""
        # Create one figure per explanation
        for i, exp in enumerate(explanations[:5]):  # Limit to 5 for space
            fig, ax = plt.subplots(figsize=(8, 6))

            # Extract data
            features = [f["name"] for f in exp["top_features"]]
            contributions = [f["contribution"] for f in exp["top_features"]]
            values = [f["value"] for f in exp["top_features"]]

            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            colors = ["red" if c < 0 else "blue" for c in contributions]

            bars = ax.barh(y_pos, contributions, color=colors, alpha=0.7)

            # Add value annotations
            for j, (bar, val) in enumerate(zip(bars, values)):
                width = bar.get_width()
                ax.text(
                    width + 0.01 if width > 0 else width - 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}",
                    ha="left" if width > 0 else "right",
                    va="center",
                    fontsize=9,
                )

            # Formatting
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel("Feature Contribution")
            ax.set_title(
                f"Local Explanation {i+1}\n"
                f"True: {exp['true_label']}, Predicted: {exp['predicted_label']}"
                f" ({'✓' if exp['correct'] else '✗'})"
            )
            ax.axvline(x=0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()
            plt.savefig(self.output_dir / f"local_explanation_{i+1}.pdf")
            plt.close()

    def _analyze_errors(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]

        if len(error_indices) == 0:
            return {"no_errors": True}

        # Analyze feature patterns in errors
        X_errors = X[error_indices]
        X_correct = X[~errors]

        # Find features with largest difference between error and correct
        mean_errors = np.mean(X_errors, axis=0)
        mean_correct = np.mean(X_correct, axis=0)

        feature_diffs = mean_errors - mean_correct
        top_error_features = np.argsort(np.abs(feature_diffs))[-20:][::-1]

        analysis = {
            "error_rate": errors.mean(),
            "n_errors": len(error_indices),
            "error_patterns": [
                {
                    "feature": self.feature_names[i],
                    "mean_in_errors": mean_errors[i],
                    "mean_in_correct": mean_correct[i],
                    "difference": feature_diffs[i],
                }
                for i in top_error_features
            ],
            "confusion_pairs": self._analyze_confusion_pairs(y_true, y_pred),
        }

        return analysis

    def _analyze_confusion_pairs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Analyze most common confusion pairs."""
        confusion_counts = {}

        for true, pred in zip(y_true, y_pred):
            if true != pred:
                pair = (true, pred)
                confusion_counts[pair] = confusion_counts.get(pair, 0) + 1

        # Sort by frequency
        sorted_pairs = sorted(
            confusion_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return [
            {
                "true_class": pair[0],
                "predicted_class": pair[1],
                "count": count,
                "percentage": count / len(y_true) * 100,
            }
            for pair, count in sorted_pairs
        ]

    def _plot_error_analysis(self, error_analysis: Dict[str, Any]):
        """Visualize error analysis."""
        if error_analysis.get("no_errors"):
            return

        # Feature differences in errors vs correct
        fig, ax = plt.subplots(figsize=(10, 8))

        patterns = error_analysis["error_patterns"][:15]
        features = [p["feature"] for p in patterns]
        differences = [p["difference"] for p in patterns]

        y_pos = np.arange(len(features))
        colors = ["red" if d < 0 else "blue" for d in differences]

        ax.barh(y_pos, differences, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Mean Difference (Errors - Correct)")
        ax.set_title("Feature Differences in Misclassified Samples")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(self.output_dir / "error_analysis_features.pdf")
        plt.close()

    def _analyze_feature_interactions(self) -> Optional[Dict[str, Any]]:
        """Analyze feature interactions for compatible models."""
        if hasattr(self.model, "explain_global"):
            # EBM model with interactions
            try:
                explanation = self.model.explain_global()
                if hasattr(explanation, "data") and "names" in explanation.data():
                    interactions = []
                    for i, name in enumerate(explanation.data()["names"]):
                        if " x " in name:  # Interaction term
                            interactions.append(
                                {
                                    "features": name.split(" x "),
                                    "importance": explanation.data()["scores"][i],
                                }
                            )
                    return {"interactions": interactions}
            except:
                pass

        return None

    def _plot_feature_interactions(self, interactions: Dict[str, Any]):
        """Plot feature interactions if available."""
        if "interactions" not in interactions:
            return

        interaction_list = interactions["interactions"]
        if not interaction_list:
            return

        # Sort by importance
        interaction_list.sort(key=lambda x: abs(x["importance"]), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Take top 10
        top_interactions = interaction_list[:10]
        labels = [" x ".join(i["features"]) for i in top_interactions]
        importances = [i["importance"] for i in top_interactions]

        y_pos = np.arange(len(labels))
        colors = ["red" if imp < 0 else "blue" for imp in importances]

        ax.barh(y_pos, importances, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Interaction Importance")
        ax.set_title("Top Feature Interactions")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_interactions.pdf")
        plt.close()
