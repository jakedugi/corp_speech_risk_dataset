#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate comprehensive paper-ready figures from POLAR CV outputs.

Inputs:
  --oof  : path to oof_predictions.jsonl (from train_polar_cv)
  --cv   : path to cv_results.json (from train_polar_cv)
  --out  : output directory for figures

Writes 18 figures for paper:
  Core performance: qwk_by_fold.pdf, confusion_oof.pdf, prf_by_class.pdf, mae_hist.pdf
  Calibration: reliability_oof_*.pdf, calibration_brier_ece.pdf, acc_vs_confidence.pdf
  Temporal: perf_over_time.pdf, class_mix_over_time.pdf
  Robustness: perf_by_quotes_per_case.pdf, perf_by_section.pdf, perf_by_speaker_type.pdf
  Curves: pr_curves_per_class.pdf
  Interpretability: coeff_forest_oof.pdf, feature_heatmap.pdf
  Error analysis: error_pairs_heatmap.pdf, residuals_vs_confidence.pdf
  Diagnostics: fold_sizes_and_mix.pdf, dev_calibration_sizes.pdf
"""

import json, argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
import warnings

warnings.filterwarnings("ignore")

# Set consistent style
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


def bootstrap_metric_by_case(
    df, prob_cols, y_col="y", case_col="case_id", n_boot=1000, seed=42, metric="qwk"
):
    """Bootstrap a metric by case_id"""
    rng = np.random.default_rng(seed)
    cases = df[case_col].dropna().unique().tolist()
    if len(cases) == 0:
        return []

    def safe_qwk(y_true, y_pred):
        present = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
        if len(present) < 2:
            return 0.0
        idx = {c: i for i, c in enumerate(present)}
        yt = np.array([idx[int(y)] for y in y_true])
        yp = np.array([idx[int(y)] for y in y_pred])
        k = len(present)
        w = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
        from sklearn.metrics import confusion_matrix

        O = confusion_matrix(yt, yp, labels=range(k))
        if O.sum() == 0:
            return 0.0
        row = O.sum(1, keepdims=True)
        col = O.sum(0, keepdims=True)
        E = row @ col / O.sum()
        denom = (w * E).sum()
        if denom == 0:
            return 0.0
        return 1.0 - (w * O).sum() / denom

    stats = []
    for _ in range(n_boot):
        samp_cases = rng.choice(cases, size=len(cases), replace=True)
        sub = df[df[case_col].isin(samp_cases)]
        P = sub[prob_cols].to_numpy()
        y = sub[y_col].to_numpy()
        yhat = P.argmax(axis=1)

        if metric == "qwk":
            v = safe_qwk(y, yhat)
        elif metric == "macro_f1":
            v = f1_score(
                y, yhat, average="macro", labels=sorted(set(y)), zero_division=0
            )
        elif metric == "mae":
            v = float(np.mean(np.abs(y - yhat)))
        elif metric == "precision":
            v = precision_score(
                y, yhat, average="macro", labels=sorted(set(y)), zero_division=0
            )
        elif metric == "recall":
            v = recall_score(
                y, yhat, average="macro", labels=sorted(set(y)), zero_division=0
            )
        else:
            v = np.nan
        stats.append(v)
    return stats


def make_core_performance_figures(oof, cv, fig_dir):
    """Generate core performance figures"""

    # 1. QWK by fold with 95% CI
    plt.figure(figsize=(8, 6))

    fold_qwks = []
    fold_ids = []
    for k, v in cv["folds"].items():
        fold_qwks.append(v["dev_metrics"].get("qwk", 0))
        fold_ids.append(int(k))

    plt.bar(fold_ids, fold_qwks, alpha=0.7, color="steelblue")
    plt.axhline(
        np.mean(fold_qwks),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(fold_qwks):.3f}",
    )
    plt.xlabel("Fold")
    plt.ylabel("Quadratic Weighted Kappa")
    plt.title("QWK Performance by Fold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "qwk_by_fold.pdf", bbox_inches="tight")
    plt.close()

    # 2. Overall confusion matrix
    # Check for both polr_ and polar_ prefixes (polr_ is current, polar_ is legacy)
    prob_cols = ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]
    if not all(c in oof.columns for c in prob_cols + ["y"]):
        prob_cols = ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]

    if all(c in oof.columns for c in prob_cols + ["y"]):
        P = oof[prob_cols].to_numpy()
        y_true = oof["y"].to_numpy()
        y_pred = P.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
        )
        plt.title("Confusion Matrix (Normalized by True Class)")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.savefig(fig_dir / "confusion_oof.pdf", bbox_inches="tight")
        plt.close()

        # 3. Class-wise precision/recall/F1
        precisions = []
        recalls = []
        f1s = []
        class_names = ["Low", "Medium", "High"]

        for c in [0, 1, 2]:
            mask = y_true == c
            if mask.sum() > 0:
                p = precision_score(y_true == c, y_pred == c, zero_division=0)
                r = recall_score(y_true == c, y_pred == c, zero_division=0)
                f = f1_score(y_true == c, y_pred == c, zero_division=0)
            else:
                p = r = f = 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        x = np.arange(len(class_names))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precisions, width, label="Precision", alpha=0.8)
        plt.bar(x, recalls, width, label="Recall", alpha=0.8)
        plt.bar(x + width, f1s, width, label="F1-Score", alpha=0.8)

        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.title("Class-wise Precision, Recall, and F1-Score")
        plt.xticks(x, class_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "prf_by_class.pdf", bbox_inches="tight")
        plt.close()

        # 4. MAE distribution
        mae_per_sample = np.abs(y_true - y_pred)

        plt.figure(figsize=(8, 6))
        plt.hist(mae_per_sample, bins=20, alpha=0.7, edgecolor="black")
        plt.axvline(
            mae_per_sample.mean(),
            color="red",
            linestyle="--",
            label=f"Mean MAE: {mae_per_sample.mean():.3f}",
        )
        plt.xlabel("Absolute Error |ŷ - y|")
        plt.ylabel("Frequency")
        plt.title("Distribution of Ordinal MAE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "mae_hist.pdf", bbox_inches="tight")
        plt.close()


def make_calibration_figures(oof, cv, fig_dir):
    """Generate calibration and probability quality figures"""

    # Check for both polr_ and polar_ prefixes (polr_ is current, polar_ is legacy)
    prob_cols = ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]
    if not all(c in oof.columns for c in prob_cols + ["y"]):
        prob_cols = ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]

    if not all(c in oof.columns for c in prob_cols + ["y"]):
        return
    P = oof[prob_cols].to_numpy()
    y_true = oof["y"].to_numpy()
    class_names = ["Low", "Medium", "High"]

    # 5. Reliability diagrams per class
    for c, class_name in enumerate(class_names):
        plt.figure(figsize=(8, 6))

        p_c = P[:, c]
        y_c = (y_true == c).astype(float)

        # Equal-count bins
        n_bins = 15
        bin_boundaries = np.percentile(p_c, np.linspace(0, 100, n_bins + 1))
        bin_boundaries = np.unique(bin_boundaries)

        bin_means = []
        bin_accuracies = []
        bin_counts = []

        for i in range(len(bin_boundaries) - 1):
            mask = (p_c >= bin_boundaries[i]) & (p_c < bin_boundaries[i + 1])
            if i == len(bin_boundaries) - 2:  # Last bin includes upper boundary
                mask = (p_c >= bin_boundaries[i]) & (p_c <= bin_boundaries[i + 1])

            if mask.sum() > 0:
                bin_means.append(p_c[mask].mean())
                bin_accuracies.append(y_c[mask].mean())
                bin_counts.append(mask.sum())

        if bin_means:
            plt.scatter(
                bin_means, bin_accuracies, s=[c * 5 for c in bin_counts], alpha=0.7
            )
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

            # Add count annotations
            for i, (x, y, n) in enumerate(zip(bin_means, bin_accuracies, bin_counts)):
                plt.annotate(
                    f"{n}",
                    (x, y),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=8,
                )

        plt.xlabel(f"Predicted Probability (Class {class_name})")
        plt.ylabel("Empirical Probability")
        plt.title(f"Reliability Diagram - {class_name} Risk")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"reliability_oof_{class_name.lower()}.pdf", bbox_inches="tight"
        )
        plt.close()

    # 6. Calibration error bars (Brier and ECE)
    fold_briers = [v["dev_metrics"].get("brier", np.nan) for v in cv["folds"].values()]
    fold_eces = [v["dev_metrics"].get("ece", np.nan) for v in cv["folds"].values()]

    metrics = ["Brier Score", "ECE"]
    values = [np.nanmean(fold_briers), np.nanmean(fold_eces)]
    errors = [np.nanstd(fold_briers), np.nanstd(fold_eces)]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, yerr=errors, capsize=5, alpha=0.7)
    plt.ylabel("Score")
    plt.title("Calibration Metrics (Mean ± SD across folds)")
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + 0.01,
            f"{val:.3f}±{err:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_brier_ece.pdf", bbox_inches="tight")
    plt.close()

    # 7. Prediction confidence vs accuracy
    confidence = P.max(axis=1)
    correct = (P.argmax(axis=1) == y_true).astype(float)

    # Bin by confidence
    conf_bins = np.linspace(0, 1, 11)
    conf_accs = []
    conf_centers = []
    conf_counts = []

    for i in range(len(conf_bins) - 1):
        mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i + 1])
        if i == len(conf_bins) - 2:
            mask = (confidence >= conf_bins[i]) & (confidence <= conf_bins[i + 1])

        if mask.sum() > 0:
            conf_centers.append((conf_bins[i] + conf_bins[i + 1]) / 2)
            conf_accs.append(correct[mask].mean())
            conf_counts.append(mask.sum())

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Line plot for accuracy
    ax1.plot(conf_centers, conf_accs, "bo-", label="Accuracy")
    ax1.set_xlabel("Prediction Confidence")
    ax1.set_ylabel("Accuracy", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    # Bar plot for volume
    ax2 = ax1.twinx()
    ax2.bar(
        conf_centers, conf_counts, alpha=0.3, width=0.08, color="gray", label="Count"
    )
    ax2.set_ylabel("Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    plt.title("Accuracy vs. Prediction Confidence")
    plt.tight_layout()
    plt.savefig(fig_dir / "acc_vs_confidence.pdf", bbox_inches="tight")
    plt.close()


def make_temporal_figures(oof, cv, fig_dir):
    """Generate temporal integrity and drift figures"""

    # 8. Performance over time (if case_time available)
    if "case_time" in oof.columns:
        try:
            oof_time = oof.copy()
            oof_time["case_time"] = pd.to_datetime(oof_time["case_time"])
            oof_time = oof_time.sort_values("case_time")

            # Rolling performance in 3-month blocks
            oof_time["year_quarter"] = oof_time["case_time"].dt.to_period("Q")

            # Check for both polr_ and polar_ prefixes
            prob_cols = ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]
            if not all(c in oof.columns for c in prob_cols + ["y"]):
                prob_cols = ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]

            if all(c in oof.columns for c in prob_cols + ["y"]):
                rolling_metrics = []
                quarters = []

                for quarter, group in oof_time.groupby("year_quarter"):
                    if len(group) >= 10:  # Minimum sample size
                        P = group[prob_cols].to_numpy()
                        y = group["y"].to_numpy()
                        yhat = P.argmax(axis=1)

                        # Calculate QWK and MAE
                        def safe_qwk(y_true, y_pred):
                            present = sorted(
                                set(map(int, y_true)) | set(map(int, y_pred))
                            )
                            if len(present) < 2:
                                return 0.0
                            idx = {c: i for i, c in enumerate(present)}
                            yt = np.array([idx[int(y)] for y in y_true])
                            yp = np.array([idx[int(y)] for y in y_pred])
                            k = len(present)
                            w = np.zeros((k, k))
                            for i in range(k):
                                for j in range(k):
                                    w[i, j] = (
                                        ((i - j) ** 2) / ((k - 1) ** 2)
                                        if k > 1
                                        else 0.0
                                    )
                            O = confusion_matrix(yt, yp, labels=range(k))
                            if O.sum() == 0:
                                return 0.0
                            row = O.sum(1, keepdims=True)
                            col = O.sum(0, keepdims=True)
                            E = row @ col / O.sum()
                            denom = (w * E).sum()
                            if denom == 0:
                                return 0.0
                            return 1.0 - (w * O).sum() / denom

                        qwk = safe_qwk(y, yhat)
                        mae = np.mean(np.abs(y - yhat))

                        rolling_metrics.append(
                            {"quarter": quarter, "qwk": qwk, "mae": mae}
                        )
                        quarters.append(str(quarter))

                if rolling_metrics:
                    metrics_df = pd.DataFrame(rolling_metrics)

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                    # QWK over time
                    ax1.plot(quarters, metrics_df["qwk"], "bo-")
                    ax1.set_ylabel("QWK")
                    ax1.set_title("Performance Over Time")
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis="x", rotation=45)

                    # MAE over time
                    ax2.plot(quarters, metrics_df["mae"], "ro-")
                    ax2.set_ylabel("MAE")
                    ax2.set_xlabel("Quarter")
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis="x", rotation=45)

                    plt.tight_layout()
                    plt.savefig(fig_dir / "perf_over_time.pdf", bbox_inches="tight")
                    plt.close()

            # 9. Class distribution over time
            class_dist = []
            for quarter, group in oof_time.groupby("year_quarter"):
                if len(group) >= 5:
                    dist = (
                        group["y"]
                        .value_counts(normalize=True)
                        .reindex([0, 1, 2], fill_value=0)
                    )
                    class_dist.append(
                        {
                            "quarter": str(quarter),
                            "low": dist[0],
                            "medium": dist[1],
                            "high": dist[2],
                        }
                    )

            if class_dist:
                dist_df = pd.DataFrame(class_dist)

                plt.figure(figsize=(12, 6))
                plt.stackplot(
                    range(len(dist_df)),
                    dist_df["low"],
                    dist_df["medium"],
                    dist_df["high"],
                    labels=["Low", "Medium", "High"],
                    alpha=0.7,
                )
                plt.xlabel("Quarter")
                plt.ylabel("Class Proportion")
                plt.title("Class Distribution Over Time")
                plt.legend(loc="upper right")
                plt.xticks(range(len(dist_df)), dist_df["quarter"], rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(fig_dir / "class_mix_over_time.pdf", bbox_inches="tight")
                plt.close()

        except Exception as e:
            print(f"Warning: Could not generate temporal figures: {e}")


def make_robustness_figures(oof, cv, fig_dir):
    """Generate robustness and subgroup analysis figures"""

    # Check for both polr_ and polar_ prefixes
    prob_cols = ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]
    if not all(c in oof.columns for c in prob_cols + ["y", "case_id"]):
        prob_cols = ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]

    if not all(c in oof.columns for c in prob_cols + ["y", "case_id"]):
        return

    # 10. Performance by quotes-per-case bucket
    qpc = oof.groupby("case_id").size().rename("qpc")
    oof_qpc = oof.merge(qpc, left_on="case_id", right_index=True, how="left")

    def buck(n):
        if n <= 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 7:
            return "4-7"
        return "8+"

    oof_qpc["qpc_bucket"] = oof_qpc["qpc"].apply(buck)

    bucket_metrics = []
    for bucket, group in oof_qpc.groupby("qpc_bucket"):
        P = group[prob_cols].to_numpy()
        y = group["y"].to_numpy()
        yhat = P.argmax(axis=1)

        def safe_qwk(y_true, y_pred):
            present = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
            if len(present) < 2:
                return 0.0
            idx = {c: i for i, c in enumerate(present)}
            yt = np.array([idx[int(y)] for y in y_true])
            yp = np.array([idx[int(y)] for y in y_pred])
            k = len(present)
            w = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
            O = confusion_matrix(yt, yp, labels=range(k))
            if O.sum() == 0:
                return 0.0
            row = O.sum(1, keepdims=True)
            col = O.sum(0, keepdims=True)
            E = row @ col / O.sum()
            denom = (w * E).sum()
            if denom == 0:
                return 0.0
            return 1.0 - (w * O).sum() / denom

        qwk = safe_qwk(y, yhat)
        mae = np.mean(np.abs(y - yhat))
        bucket_metrics.append(
            {"bucket": bucket, "qwk": qwk, "mae": mae, "n": len(group)}
        )

    bucket_df = pd.DataFrame(bucket_metrics)
    bucket_order = ["1", "2-3", "4-7", "8+"]
    bucket_df = bucket_df.set_index("bucket").reindex(bucket_order).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # QWK by bucket
    bars1 = ax1.bar(bucket_df["bucket"], bucket_df["qwk"], alpha=0.7)
    ax1.set_xlabel("Quotes per Case")
    ax1.set_ylabel("QWK")
    ax1.set_title("QWK by Quotes per Case")
    ax1.grid(True, alpha=0.3)

    # Add sample size annotations
    for bar, n in zip(bars1, bucket_df["n"]):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # MAE by bucket
    bars2 = ax2.bar(bucket_df["bucket"], bucket_df["mae"], alpha=0.7, color="orange")
    ax2.set_xlabel("Quotes per Case")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE by Quotes per Case")
    ax2.grid(True, alpha=0.3)

    for bar, n in zip(bars2, bucket_df["n"]):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(fig_dir / "perf_by_quotes_per_case.pdf", bbox_inches="tight")
    plt.close()


def make_curve_figures(oof, cv, fig_dir):
    """Generate PR curves and other threshold-free curves"""

    # Check for both polr_ and polar_ prefixes
    prob_cols = ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]
    if not all(c in oof.columns for c in prob_cols + ["y"]):
        prob_cols = ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]

    if not all(c in oof.columns for c in prob_cols + ["y"]):
        return
    P = oof[prob_cols].to_numpy()
    y_true = oof["y"].to_numpy()
    class_names = ["Low", "Medium", "High"]

    # 12. One-vs-rest PR curves per class
    plt.figure(figsize=(12, 4))

    for i, (class_idx, class_name) in enumerate(zip([0, 1, 2], class_names)):
        plt.subplot(1, 3, i + 1)

        y_binary = (y_true == class_idx).astype(int)
        p_class = P[:, class_idx]

        if y_binary.sum() > 0:  # Only if positive samples exist
            precision, recall, _ = precision_recall_curve(y_binary, p_class)
            auc_pr = np.trapz(precision, recall)

            plt.plot(recall, precision, label=f"AUPRC = {auc_pr:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{class_name} Risk (One-vs-Rest)")
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curves_per_class.pdf", bbox_inches="tight")
    plt.close()


def make_error_analysis_figures(oof, cv, fig_dir):
    """Generate error analysis figures"""

    # Check for both polr_ and polar_ prefixes
    prob_cols = ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]
    if not all(c in oof.columns for c in prob_cols + ["y"]):
        prob_cols = ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]

    if not all(c in oof.columns for c in prob_cols + ["y"]):
        return
    P = oof[prob_cols].to_numpy()
    y_true = oof["y"].to_numpy()
    y_pred = P.argmax(axis=1)

    # 15. Error by true/pred pair heatmap
    plt.figure(figsize=(8, 6))

    error_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Highlight ±1 vs ±2 errors
    error_types = np.zeros_like(error_matrix, dtype=str)
    for i in range(3):
        for j in range(3):
            if i == j:
                error_types[i, j] = "Correct"
            elif abs(i - j) == 1:
                error_types[i, j] = "±1 Error"
            else:
                error_types[i, j] = "±2 Error"

    # Create custom colormap
    cmap = sns.color_palette("Blues", as_cmap=True)

    ax = sns.heatmap(
        error_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
    )

    # Add error type annotations
    for i in range(3):
        for j in range(3):
            text = ax.texts[i * 3 + j]
            text.set_text(f"{error_matrix[i,j]}\n({error_types[i,j]})")

    plt.title("Error Pairs Heatmap")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(fig_dir / "error_pairs_heatmap.pdf", bbox_inches="tight")
    plt.close()

    # 16. Residuals vs confidence
    confidence = P.max(axis=1)
    residuals = np.abs(y_true - y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(confidence, residuals, alpha=0.5)

    # Add trend line
    z = np.polyfit(confidence, residuals, 1)
    p = np.poly1d(z)
    plt.plot(confidence, p(confidence), "r--", alpha=0.8, label=f"Slope: {z[0]:.3f}")

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Absolute Error")
    plt.title("Residuals vs. Confidence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_vs_confidence.pdf", bbox_inches="tight")
    plt.close()


def make_diagnostic_figures(oof, cv, fig_dir):
    """Generate data and pipeline diagnostic figures"""

    # 17. Per-fold sample size & class mix
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    fold_data = []
    for k, v in cv["folds"].items():
        dm = v.get("dev_metadata", {})
        fold_data.append(
            {
                "fold": int(k),
                "cases": dm.get("n_cases", 0),
                "quotes": dm.get("n_quotes", 0),
                "classes": dm.get("n_classes", 0),
                "class_list": dm.get("classes", []),
            }
        )

    fold_df = pd.DataFrame(fold_data).sort_values("fold")

    # Sample sizes
    for i, (idx, row) in enumerate(fold_df.iterrows()):
        if i >= 5:  # Only show first 5 folds
            break
        ax = axes[i]
        ax.bar(["Cases", "Quotes"], [row["cases"], row["quotes"]], alpha=0.7)
        ax.set_title(f'Fold {row["fold"]}')
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(fold_df), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Sample Sizes per Fold (DEV set)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fold_sizes_and_mix.pdf", bbox_inches="tight")
    plt.close()

    # 18. DEV calibration sizes per fold
    plt.figure(figsize=(10, 6))

    dev_sizes = [
        v.get("dev_metadata", {}).get("n_quotes", 0) for v in cv["folds"].values()
    ]
    fold_ids = [int(k) for k in cv["folds"].keys()]

    bars = plt.bar(fold_ids, dev_sizes, alpha=0.7)
    plt.axhline(
        500, color="red", linestyle="--", alpha=0.7, label="min_cal_n threshold"
    )

    plt.xlabel("Fold")
    plt.ylabel("DEV Set Size (quotes)")
    plt.title("Calibration Set Sizes per Fold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add size annotations
    for bar, size in zip(bars, dev_sizes):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 20,
            f"{size}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(fig_dir / "dev_calibration_sizes.pdf", bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="Path to oof_predictions.jsonl")
    ap.add_argument("--cv", required=True, help="Path to cv_results.json")
    ap.add_argument("--out", required=True, help="Output directory for figures")
    args = ap.parse_args()

    fig_dir = Path(args.out) / "oof"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    oof = pd.read_json(args.oof, lines=True)
    with open(args.cv, "r") as f:
        cv = json.load(f)

    print(f"Generating figures for {len(oof)} OOF predictions...")

    # Generate all figure categories
    make_core_performance_figures(oof, cv, fig_dir)
    print("✓ Core performance figures")

    make_calibration_figures(oof, cv, fig_dir)
    print("✓ Calibration figures")

    make_temporal_figures(oof, cv, fig_dir)
    print("✓ Temporal figures")

    make_robustness_figures(oof, cv, fig_dir)
    print("✓ Robustness figures")

    make_curve_figures(oof, cv, fig_dir)
    print("✓ Curve figures")

    make_error_analysis_figures(oof, cv, fig_dir)
    print("✓ Error analysis figures")

    make_diagnostic_figures(oof, cv, fig_dir)
    print("✓ Diagnostic figures")

    print(f"\nGenerated 18 paper-ready figures in: {fig_dir}")
    print("Figures are ready for direct inclusion in your manuscript!")


if __name__ == "__main__":
    main()
