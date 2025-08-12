"""Academic-quality reporting and visualization for case-level modeling.

Generates publication-ready figures and interpretability reports:
- Feature importance visualizations
- Cross-validation performance with confidence intervals
- Statistical significance tests between models
- Calibration plots
- Partial dependence plots
- Fairness and bias analysis
- LaTeX-ready tables
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Optional, Tuple, Any
import warnings

import numpy as np
import polars as pl
from scipy import stats

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True

    # Set publication-quality defaults
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )
    sns.set_style("whitegrid")
    sns.set_context("paper")
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None


def plot_feature_importance(
    importance_dict: Dict[str, float],
    out_path: str,
    top_k: int = 15,
    title: Optional[str] = None,
) -> None:
    """Plot feature importance as horizontal bar chart."""
    if not HAS_PLOTTING or not importance_dict:
        return

    # Sort and select top features
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_k]

    if not top_features:
        return

    names, values = zip(*top_features)

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(names))

    ax.barh(y_pos, values, color=sns.color_palette("viridis", len(names)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cv_comparison(
    results: Dict[str, Dict[str, Any]],
    out_path: str,
    metric: str = "cv_accuracy_mean",
    ci_key: str = "accuracy_ci",
) -> None:
    """Plot cross-validation results with confidence intervals."""
    if not HAS_PLOTTING:
        return

    models = []
    means = []
    lowers = []
    uppers = []

    for name, info in results.items():
        if metric in info and ci_key in info:
            models.append(name)
            means.append(info[metric])
            ci = info[ci_key]
            if isinstance(ci, (tuple, list)) and len(ci) >= 3:
                _, lower, upper = ci[:3]
                lowers.append(lower)
                uppers.append(upper)
            else:
                lowers.append(info[metric])
                uppers.append(info[metric])

    if not models:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(len(models))

    # Plot bars
    bars = ax.bar(x_pos, means, color=sns.color_palette("muted")[0])

    # Add error bars
    errors = np.array(
        [np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)]
    )
    ax.errorbar(
        x_pos, means, yerr=errors, fmt="none", capsize=5, color="black", alpha=0.7
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.1)

    # Add value labels
    for i, (m, l, u) in enumerate(zip(means, lowers, uppers)):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix_academic(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    out_path: str,
    normalize: bool = True,
) -> None:
    """Create academic-quality confusion matrix visualization."""
    if not HAS_PLOTTING:
        return

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: str,
    n_bins: int = 10,
) -> None:
    """Plot calibration curve for probability predictions."""
    if not HAS_PLOTTING:
        return

    # Compute calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    empirical_probs = []
    predicted_probs = []

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            empirical_probs.append(y_true[mask].mean())
            predicted_probs.append(y_proba[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Actual calibration
    ax.plot(
        predicted_probs,
        empirical_probs,
        "o-",
        color=sns.color_palette("deep")[0],
        label="Model",
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_latex_table(
    results: Dict[str, Dict[str, Any]],
    out_path: str,
    metrics: List[str] = ["test_accuracy", "test_r2"],
) -> None:
    """Generate LaTeX table of model results."""
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Model Performance Comparison}")
    lines.append("\\label{tab:model_performance}")

    # Determine columns
    col_spec = "l" + "r" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = "Model"
    for metric in metrics:
        header += f" & {metric.replace('_', ' ').title()}"
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for model, info in sorted(results.items()):
        row = model.replace("_", "\\_")
        for metric in metrics:
            if metric in info:
                value = info[metric]
                if isinstance(value, float):
                    # Check for confidence intervals
                    ci_key = metric.replace("test_", "") + "_ci"
                    if ci_key in info and isinstance(info[ci_key], (tuple, list)):
                        _, lower, upper = info[ci_key][:3]
                        row += f" & ${value:.3f}$ (${lower:.3f}$--${upper:.3f}$)"
                    else:
                        row += f" & ${value:.3f}$"
                else:
                    row += f" & {value}"
            else:
                row += " & --"
        lines.append(row + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def plot_spearman_scatter(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: Optional[str] = None,
) -> Optional[float]:
    """Create scatter plot with Spearman correlation."""
    if not HAS_PLOTTING or df.is_empty():
        return None

    if x_col not in df.columns or y_col not in df.columns:
        return None

    data = df.select([x_col, y_col]).drop_nulls()
    if data.height < 10:
        return None

    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()

    # Compute Spearman correlation
    rho, p_value = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, color=sns.color_palette("deep")[0])

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Spearman ρ = {rho:.3f}, p = {p_value:.4f}")

    # Add text box with correlation
    textstr = f"ρ = {rho:.3f}\np = {p_value:.4f}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(
        0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return float(rho)


def plot_statistical_significance(
    model_comparisons: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    """Plot statistical significance matrix between models."""
    if not HAS_PLOTTING or not model_comparisons:
        return

    # Extract unique models
    models = set()
    for comp in model_comparisons.keys():
        m1, m2 = comp.split("_vs_")
        models.add(m1)
        models.add(m2)
    models = sorted(list(models))

    # Build p-value matrix
    n = len(models)
    p_matrix = np.ones((n, n))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i != j:
                key = f"{m1}_vs_{m2}"
                if key in model_comparisons:
                    p_matrix[i, j] = model_comparisons[key].get("p_value", 1.0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create mask for diagonal
    mask = np.eye(n, dtype=bool)

    sns.heatmap(
        p_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        xticklabels=models,
        yticklabels=models,
        mask=mask,
        vmin=0,
        vmax=0.1,
        cbar_kws={"label": "p-value"},
    )

    ax.set_title("Statistical Significance (p-values)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_fairness_analysis(
    fairness_results: Dict[str, Any],
    out_path: str,
) -> None:
    """Visualize fairness metrics across classes."""
    if not HAS_PLOTTING or not fairness_results:
        return

    class_metrics = fairness_results.get("class_metrics", {})
    if not class_metrics:
        return

    classes = list(class_metrics.keys())
    recalls = [class_metrics[c]["recall"] for c in classes]
    precisions = [class_metrics[c]["precision"] for c in classes]
    supports = [class_metrics[c]["support"] for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Recall/Precision by class
    x = np.arange(len(classes))
    width = 0.35

    ax1.bar(x - width / 2, recalls, width, label="Recall", alpha=0.8)
    ax1.bar(x + width / 2, precisions, width, label="Precision", alpha=0.8)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Score")
    ax1.set_title("Performance by Class")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Support distribution
    ax2.pie(supports, labels=classes, autopct="%1.1f%%")
    ax2.set_title("Class Distribution (Support)")

    # Add disparate impact ratio
    dir_text = f"Disparate Impact Ratio: {fairness_results.get('disparate_impact_ratio', 0):.3f}"
    fig.text(0.5, 0.02, dir_text, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_comprehensive_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    threshold_name: str,
) -> None:
    """Generate comprehensive academic report for a threshold."""
    os.makedirs(output_dir, exist_ok=True)

    # Feature importance for best model
    best_model = max(results.items(), key=lambda x: x[1].get("test_accuracy", 0))
    if "feature_importance" in best_model[1]:
        plot_feature_importance(
            best_model[1]["feature_importance"],
            os.path.join(output_dir, "feature_importance.png"),
            title=f"Feature Importance ({best_model[0]})",
        )

    # Cross-validation comparison
    plot_cv_comparison(results, os.path.join(output_dir, "cv_comparison.png"))

    # LaTeX table
    generate_latex_table(results, os.path.join(output_dir, "results_table.tex"))

    # Write summary statistics
    with open(os.path.join(output_dir, "summary_stats.txt"), "w") as f:
        f.write(f"Threshold: {threshold_name}\n")
        f.write("=" * 50 + "\n\n")

        for model, info in sorted(results.items()):
            f.write(f"Model: {model}\n")
            f.write("-" * 30 + "\n")

            # Key metrics
            if "test_accuracy" in info:
                f.write(f"Test Accuracy: {info['test_accuracy']:.4f}\n")
                if "accuracy_ci" in info:
                    _, lower, upper = info["accuracy_ci"][:3]
                    f.write(f"  95% CI: [{lower:.4f}, {upper:.4f}]\n")

            if "cv_accuracy_mean" in info:
                f.write(
                    f"CV Accuracy: {info['cv_accuracy_mean']:.4f} "
                    f"(±{info.get('cv_accuracy_std', 0):.4f})\n"
                )

            if "best_params" in info and info["best_params"]:
                f.write(f"Best Params: {info['best_params']}\n")

            f.write("\n")
