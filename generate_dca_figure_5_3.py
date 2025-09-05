#!/usr/bin/env python3
"""
Generate Figure 5.3: Decision Curve Analysis comparing E-only vs E+3 case models
Based on Vickers & Elkin (2006) decision curve analysis methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_net_benefit(y_true, y_prob, threshold):
    """
    Calculate net benefit at a given threshold probability.

    Net Benefit = (TP/n) - (FP/n) × (pt/(1-pt))

    Args:
        y_true: True binary outcomes (0/1)
        y_prob: Predicted probabilities [0,1]
        threshold: Threshold probability for classification

    Returns:
        net_benefit: Net benefit at this threshold
    """
    n = len(y_true)

    # Classify predictions at threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate confusion matrix components
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    # Net benefit formula
    if threshold == 1.0:
        # Edge case: treat all as negative
        return 0.0

    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit


def calculate_dca_curve(y_true, y_prob, thresholds=None):
    """
    Calculate decision curve across threshold range.

    Args:
        y_true: True binary outcomes
        y_prob: Predicted probabilities
        thresholds: Array of threshold probabilities (default: 0.01 to 0.99 by 0.01)

    Returns:
        thresholds, net_benefits: Arrays for plotting
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    net_benefits = []
    for threshold in thresholds:
        nb = calculate_net_benefit(y_true, y_prob, threshold)
        net_benefits.append(nb)

    return thresholds, np.array(net_benefits)


def calculate_reference_strategies(y_true, thresholds):
    """
    Calculate net benefit for reference strategies.

    Args:
        y_true: True binary outcomes
        thresholds: Array of threshold probabilities

    Returns:
        treat_all_nb, treat_none_nb: Net benefits for reference strategies
    """
    n = len(y_true)
    prevalence = np.mean(y_true)

    # Treat all: net benefit = prevalence - (1-prevalence) × pt/(1-pt)
    treat_all_nb = []
    for pt in thresholds:
        if pt >= prevalence:
            # When threshold > prevalence, treat-all has negative net benefit
            nb = prevalence - (1 - prevalence) * (pt / (1 - pt))
        else:
            nb = prevalence - (1 - prevalence) * (pt / (1 - pt))
        treat_all_nb.append(max(0, nb))  # Floor at 0

    # Treat none: always 0 net benefit
    treat_none_nb = np.zeros_like(thresholds)

    return np.array(treat_all_nb), treat_none_nb


def create_dca_figure():
    """Generate Figure 5.3: Decision Curve Analysis"""

    # Load prediction data
    e_only = pd.read_csv("results/case_lr_hero_E/case_predictions.csv")
    e_plus3 = pd.read_csv("results/case_lr_hero_E_3/case_predictions.csv")

    # Verify same test cases
    assert e_only["case_id"].equals(e_plus3["case_id"]), "Case IDs must match"

    # Extract data
    y_true = e_only["y_true"].values
    prob_e = e_only["prob_cal"].values
    prob_e3 = e_plus3["prob_cal"].values

    print(f"Test set: n={len(y_true)}, prevalence={np.mean(y_true):.3f}")
    print(f"E-only AUC: {e_only[['y_true', 'prob_cal']].corr().iloc[0,1]:.3f}")
    print(f"E+3 AUC: {e_plus3[['y_true', 'prob_cal']].corr().iloc[0,1]:.3f}")

    # Define threshold range focused on decision-relevant region
    thresholds = np.arange(0.01, 0.80, 0.01)  # Up to 80% threshold

    # Calculate decision curves
    thresh_e, nb_e = calculate_dca_curve(y_true, prob_e, thresholds)
    thresh_e3, nb_e3 = calculate_dca_curve(y_true, prob_e3, thresholds)

    # Calculate reference strategies
    treat_all_nb, treat_none_nb = calculate_reference_strategies(y_true, thresholds)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Plot decision curves
    ax.plot(thresh_e, nb_e, "b-", linewidth=2.5, label="E-only Case Model", alpha=0.9)
    ax.plot(thresh_e3, nb_e3, "r-", linewidth=2.5, label="E+3 Case Model", alpha=0.9)

    # Plot reference strategies
    ax.plot(
        thresholds, treat_all_nb, "k--", linewidth=1.5, label="Treat All", alpha=0.7
    )
    ax.plot(
        thresholds, treat_none_nb, "k:", linewidth=1.5, label="Treat None", alpha=0.7
    )

    # Styling
    ax.set_xlabel("Threshold Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net Benefit", fontsize=12, fontweight="bold")
    ax.set_title(
        "Decision Curve Analysis: E-only vs E+3 Case Models\n(Test Split Performance)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Grid and layout
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)

    # Set reasonable y-axis limits
    y_max = max(np.max(nb_e), np.max(nb_e3), np.max(treat_all_nb)) * 1.1
    y_min = min(np.min(nb_e), np.min(nb_e3)) * 1.1
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 0.8)

    # Add performance annotations
    ax.text(
        0.05,
        0.95 * y_max,
        f"Test Prevalence: {np.mean(y_true):.1%}",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Highlight key operating points from model cards
    e_threshold_mcc = 0.675  # From E-only model card
    e3_threshold_mcc = 0.38  # From E+3 model card

    # Calculate net benefits at operating points
    nb_e_op = calculate_net_benefit(y_true, prob_e, e_threshold_mcc)
    nb_e3_op = calculate_net_benefit(y_true, prob_e3, e3_threshold_mcc)

    # Mark operating points
    ax.scatter(
        [e_threshold_mcc],
        [nb_e_op],
        color="blue",
        s=100,
        marker="o",
        edgecolors="white",
        linewidth=2,
        label="E-only τ_MCC=0.675",
        zorder=5,
    )
    ax.scatter(
        [e3_threshold_mcc],
        [nb_e3_op],
        color="red",
        s=100,
        marker="s",
        edgecolors="white",
        linewidth=2,
        label="E+3 τ_MCC=0.38",
        zorder=5,
    )

    plt.tight_layout()

    # Save figure
    fig_path = Path("docs/figures/fig_5_3_dca_comparison.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to: {fig_path}")

    # Print summary statistics
    print("\n=== Decision Curve Analysis Summary ===")
    print(f"E-only model:")
    print(
        f"  - Peak net benefit: {np.max(nb_e):.4f} at threshold {thresh_e[np.argmax(nb_e)]:.3f}"
    )
    print(f"  - Operating point (τ_MCC=0.675): {nb_e_op:.4f}")

    print(f"E+3 model:")
    print(
        f"  - Peak net benefit: {np.max(nb_e3):.4f} at threshold {thresh_e3[np.argmax(nb_e3)]:.3f}"
    )
    print(f"  - Operating point (τ_MCC=0.38): {nb_e3_op:.4f}")

    # Model comparison
    area_e = np.trapz(nb_e, thresh_e)
    area_e3 = np.trapz(nb_e3, thresh_e3)
    print(f"\nArea under decision curve:")
    print(f"  - E-only: {area_e:.4f}")
    print(f"  - E+3: {area_e3:.4f}")
    print(f"  - Difference: {area_e - area_e3:.4f} (positive favors E-only)")

    plt.show()
    return fig, ax


if __name__ == "__main__":
    create_dca_figure()
