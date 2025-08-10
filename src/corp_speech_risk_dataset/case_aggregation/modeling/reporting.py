"""Publication-oriented reporting utilities for case-level modeling.

Generates:
- Class distribution plot for labels
- Per-threshold model accuracy bars
- Best-model confusion matrix (counts + normalized)
- Best-model per-class precision/recall/F1 bar chart
- Cross-threshold summary accuracy bars
- Box/violin plots by outcome

All figures are saved as PNG with tight layout, suitable for academic use.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import polars as pl


def _safe_import_matplotlib():  # pragma: no cover - optional dependency
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def save_class_distribution(y: pl.DataFrame, out_dir: str, title: str) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    counts = y.group_by("outcome_bucket").count().sort("count", descending=True)
    labels = counts["outcome_bucket"].to_list()
    values = counts["count"].to_list()
    plt.figure(figsize=(5, 3.5))
    plt.bar(labels, values, color="#4C78A8")
    plt.xlabel("Outcome bucket")
    plt.ylabel("Count of cases")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close()


def boxplot_by_outcome(
    df: pl.DataFrame,
    value_col: str,
    out_path: str,
    title: str,
) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return
    if df.is_empty() or value_col not in df.columns:
        return
    # Expect columns: outcome_bucket, <value_col>
    d = df.select(["outcome_bucket", value_col]).drop_nulls()
    # Use implode to collect group values into a list (portable across polars versions)
    groups = d.group_by("outcome_bucket").agg(pl.col(value_col).implode().alias("vals"))
    labels = groups.select("outcome_bucket").to_series().to_list()
    data = groups.select("vals").to_series().to_list()
    # Sort labels/data in a canonical order if possible
    order = ["low", "medium", "high"]
    try:
        idx = [labels.index(o) for o in order if o in labels]
        if idx:
            labels = [labels[i] for i in idx]
            data = [data[i] for i in idx]
    except Exception:
        pass
    plt.figure(figsize=(6, 3.5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def violin_by_outcome(
    df: pl.DataFrame,
    value_col: str,
    out_path: str,
    title: str,
) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return
    if df.is_empty() or value_col not in df.columns:
        return
    d = df.select(["outcome_bucket", value_col]).drop_nulls()
    groups = d.group_by("outcome_bucket").agg(pl.col(value_col).implode().alias("vals"))
    labels = groups.select("outcome_bucket").to_series().to_list()
    data = groups.select("vals").to_series().to_list()
    order = ["low", "medium", "high"]
    try:
        idx = [labels.index(o) for o in order if o in labels]
        if idx:
            labels = [labels[i] for i in idx]
            data = [data[i] for i in idx]
    except Exception:
        pass
    plt.figure(figsize=(6, 3.5))
    parts = plt.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#A0CBE8")
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def spearman_correlation(df: pl.DataFrame, x_col: str, y_col: str) -> float | None:
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:
        return None
    if df.is_empty() or x_col not in df.columns or y_col not in df.columns:
        return None
    sub = df.select([x_col, y_col]).drop_nulls()
    if sub.height < 3:
        return None
    x = sub[x_col].to_list()
    y = sub[y_col].to_list()
    rho, _ = spearmanr(x, y)
    try:
        return float(rho)
    except Exception:
        return None


def plot_model_accuracies(
    results: Dict[str, Dict[str, object]], out_path: str, title: str
) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return
    names: List[str] = []
    accs: List[float] = []
    for name, info in results.items():
        acc_raw = info.get("accuracy") if isinstance(info, dict) else 0.0
        acc_val = float(acc_raw) if isinstance(acc_raw, (int, float)) else 0.0
        names.append(name)
        accs.append(acc_val)
    plt.figure(figsize=(6, 3.5))
    plt.bar(names, accs, color="#54A24B")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    out_path_counts: str,
    out_path_norm: str,
) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return
    try:
        from sklearn.metrics import confusion_matrix
    except Exception:
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    # Normalized per true class (rows)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    im0 = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_title("Confusion (counts)")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels)
    ax[0].set_yticks(range(len(labels)))
    ax[0].set_yticklabels(labels)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    # Normalized
    im1 = ax[1].imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Greens")
    ax[1].set_title("Confusion (row-normalized)")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels)
    ax[1].set_yticks(range(len(labels)))
    ax[1].set_yticklabels(labels)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path_counts, dpi=150)
    plt.close(fig)
    # Save normalized separately as well
    plt.figure(figsize=(4, 3.5))
    plt.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path_norm, dpi=150)
    plt.close()


def plot_per_class_metrics(
    report: Dict[str, Dict[str, float]],
    labels: Sequence[str],
    out_path: str,
    title: str,
) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return
    # Use sklearn classification_report dict format per label
    prec = [float(report.get(lbl, {}).get("precision", 0.0)) for lbl in labels]
    rec = [float(report.get(lbl, {}).get("recall", 0.0)) for lbl in labels]
    f1 = [float(report.get(lbl, {}).get("f1-score", 0.0)) for lbl in labels]
    x = range(len(labels))
    width = 0.25
    plt.figure(figsize=(6, 3.5))
    plt.bar([i - width for i in x], prec, width=width, label="Precision")
    plt.bar(x, rec, width=width, label="Recall")
    plt.bar([i + width for i in x], f1, width=width, label="F1")
    plt.xticks(list(x), labels)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_threshold_summary(
    summary: Dict[str, Dict[str, object]], out_path: str, title: str
) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return
    names: List[str] = []
    accs: List[float] = []
    covs: List[float] = []
    for thr, info in summary.items():
        names.append(thr)
        acc_raw = info.get("accuracy") if isinstance(info, dict) else 0.0
        cov_raw = info.get("coverage") if isinstance(info, dict) else 0.0
        accs.append(float(acc_raw) if isinstance(acc_raw, (int, float)) else 0.0)
        covs.append(float(cov_raw) if isinstance(cov_raw, (int, float)) else 0.0)
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.bar(names, accs, color="#F58518", label="Best accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Accuracy")
    ax2 = ax1.twinx()
    ax2.plot(names, covs, color="#4C78A8", marker="o", label="Coverage")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Coverage")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
