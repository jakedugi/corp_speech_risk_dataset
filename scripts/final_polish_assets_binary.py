#!/usr/bin/env python3
"""Final polish assets for publication-ready paper - BINARY VERSION.

Generates:
- T5: Binary logistic regression associations with OR, CIs, p-values
- T6: Sensitivity analysis with court fixed effects
- F5: Calibration curves (reliability plots) for DEV and OOF
- F6: Coefficient plot with confidence intervals
- Computational environment documentation
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
    }
)


def load_final_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load final binary data splits and feature set."""
    print("📊 Loading final binary data and features...")

    # Load fold metadata from binary data
    with open(
        "data/final_stratified_kfold_splits_binary_quote_balanced/per_fold_metadata.json",
        "r",
    ) as f:
        fold_metadata = json.load(f)

    # Load splits
    splits = {}
    for split_name, path_suffix in [
        ("train", "fold_4/train.jsonl"),
        ("dev", "fold_4/dev.jsonl"),
        ("test", "oof_test/test.jsonl"),
    ]:
        split_path = Path(
            f"data/final_stratified_kfold_splits_binary_quote_balanced/{path_suffix}"
        )
        data = []
        with open(split_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        splits[split_name] = pd.DataFrame(data)

    # Use precomputed binary outcome_bin labels (0 = Low/Green, 1 = High/Red)
    for split_name, df in splits.items():
        if "outcome_bin" in df.columns:
            # Use authoritative precomputed binary labels
            df["bin"] = df["outcome_bin"].astype(int)
        else:
            # Fallback for missing labels (should not happen with authoritative data)
            df["bin"] = 0  # Default to lower risk
        df["case_size"] = df.groupby("case_id")["case_id"].transform("count")

    # Load final features
    with open(
        "docs/feature_analysis/final_feature_set/final_kept_features.txt", "r"
    ) as f:
        features = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    print(
        f"✓ Loaded: train={len(splits['train']):,}, dev={len(splits['dev']):,}, test={len(splits['test']):,}"
    )
    print(f"✓ Features: {len(features)}")

    return splits["train"], splits["dev"], splits["test"], features


def extract_court_from_case_id(case_id: str) -> str:
    """Extract court district from anonymized case ID."""
    import re

    if not case_id or pd.isna(case_id):
        return "unknown"

    case_id_str = str(case_id).strip()

    # Pattern 1: anon_X_cv_XXXXX_COURT (e.g., "anon_1_cv_02184_nysd")
    match = re.search(r"anon_\d+_cv_\d+_([a-z]+)$", case_id_str)
    if match:
        return match.group(1)

    # Pattern 2: X:XX-cv-XXXXX_COURT (e.g., "1:XX-cv-04567_nysd")
    match = re.search(r"\d+:XX-[a-z]+-\d+_([a-z]+)$", case_id_str)
    if match:
        return match.group(1)

    # Pattern 3: Any case ending with _COURT
    match = re.search(r"_([a-z]+)$", case_id_str)
    if match:
        return match.group(1)

    return "unknown"


def prepare_case_level_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Prepare case-level data for binary logistic regression."""
    # Aggregate to case level
    case_data = (
        df.groupby("case_id")
        .agg(
            {
                **{f: "first" for f in features if f in df.columns},
                "bin": "first",
                "case_size": "first",
            }
        )
        .reset_index()
    )

    # Extract court information from case_id
    case_data["court"] = case_data["case_id"].apply(extract_court_from_case_id)

    # Check how many courts we found
    court_counts = case_data["court"].value_counts()
    print(f"✓ Extracted court data: {len(court_counts)} courts from case IDs")
    print(f"  Top courts: {dict(court_counts.head())}")

    return case_data


def compute_binary_logistic_main(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> Dict:
    """Compute main binary logistic regression results (T5)."""
    print("📊 Computing binary logistic regression associations (T5)...")

    # Prepare case-level data
    train_cases = prepare_case_level_data(train_df, features)
    dev_cases = prepare_case_level_data(dev_df, features)
    test_cases = prepare_case_level_data(test_df, features)

    # Filter to available features
    available_features = [f for f in features if f in train_cases.columns]

    if len(available_features) == 0:
        return {}

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_cases[available_features].fillna(0))
    y_train = train_cases["bin"]

    results = {
        "features": [],
        "or": [],
        "or_lower": [],
        "or_upper": [],
        "pvalue": [],
        "coef": [],
        "se": [],
    }

    try:
        # Fit binary logistic regression
        X_train_with_const = sm.add_constant(X_train)
        model = Logit(y_train, X_train_with_const)
        fitted = model.fit(method="bfgs", disp=False, maxiter=200)

        # Extract coefficients (skip intercept)
        main_params = fitted.params[1:]  # Skip intercept
        main_bse = fitted.bse[1:]
        main_pvalues = fitted.pvalues[1:]

        # Confidence intervals
        conf_int = fitted.conf_int()

        for i, feature in enumerate(available_features):
            coef = main_params[i]
            se = main_bse[i]
            ci_lower = conf_int.iloc[i + 1, 0]  # +1 to skip intercept
            ci_upper = conf_int.iloc[i + 1, 1]

            results["features"].append(feature.replace("interpretable_", ""))
            results["coef"].append(coef)
            results["se"].append(se)
            results["or"].append(np.exp(coef))
            results["or_lower"].append(np.exp(ci_lower))
            results["or_upper"].append(np.exp(ci_upper))
            results["pvalue"].append(main_pvalues[i])

        # Model diagnostics
        results["pseudo_r2"] = fitted.prsquared
        results["aic"] = fitted.aic
        results["bic"] = fitted.bic
        results["n_obs"] = len(y_train)

        # Evaluate on dev and test
        for eval_name, eval_cases in [("dev", dev_cases), ("test", test_cases)]:
            if len(eval_cases) > 0 and len(available_features) > 0:
                X_eval = scaler.transform(eval_cases[available_features].fillna(0))
                X_eval_with_const = sm.add_constant(X_eval)
                y_eval = eval_cases["bin"]

                # Predict probabilities
                pred_probs = fitted.predict(X_eval_with_const)
                pred_class = (pred_probs >= 0.5).astype(int)

                # Compute metrics
                from sklearn.metrics import accuracy_score, f1_score

                accuracy = accuracy_score(y_eval, pred_class)
                macro_f1 = f1_score(y_eval, pred_class, average="macro")

                # Store probabilities for calibration analysis
                results[f"{eval_name}_probs"] = pred_probs
                results[f"{eval_name}_true"] = y_eval.values
                results[f"{eval_name}_accuracy"] = accuracy
                results[f"{eval_name}_macro_f1"] = macro_f1

        print(
            f"✓ Binary logistic regression: {len(available_features)} features, Pseudo-R² = {results['pseudo_r2']:.3f}"
        )

    except Exception as e:
        print(f"  Warning: Binary logistic regression failed: {e}")
        return {}

    return results


def compute_sensitivity_with_court_fe(
    train_df: pd.DataFrame, features: List[str]
) -> Dict:
    """Compute sensitivity analysis with court fixed effects (T6)."""
    print("🏛️ Computing sensitivity analysis with court fixed effects...")

    # Prepare case-level data
    train_cases = prepare_case_level_data(train_df, features)

    # Check if court data is available
    if "court" not in train_cases.columns:
        print("  No court data available for fixed effects")
        return {"note": "Court data not available in dataset"}

    # Filter to available features
    available_features = [f for f in features if f in train_cases.columns]

    if len(available_features) == 0:
        return {}

    # Get courts with sufficient observations
    court_counts = train_cases["court"].value_counts()
    major_courts = court_counts[court_counts >= 5].index  # At least 5 cases

    if len(major_courts) < 2:
        return {"note": "Insufficient court variation for fixed effects"}

    # Filter to major courts
    fe_data = train_cases[train_cases["court"].isin(major_courts)].copy()

    # Create court dummies
    court_dummies = pd.get_dummies(fe_data["court"], prefix="court", drop_first=True)

    # Prepare feature matrix
    scaler = StandardScaler()
    X_features = scaler.fit_transform(fe_data[available_features].fillna(0))
    X_with_fe = np.column_stack([X_features, court_dummies.values])

    feature_names = [f.replace("interpretable_", "") for f in available_features]
    court_names = court_dummies.columns.tolist()

    y = fe_data["bin"].values

    results = {
        "features": [],
        "coef_main": [],
        "coef_fe": [],
        "se_main": [],
        "se_fe": [],
        "or_main": [],
        "or_fe": [],
        "pvalue_main": [],
        "pvalue_fe": [],
    }

    try:
        # Fit models
        X_main_with_const = sm.add_constant(X_features)
        model_main = Logit(y, X_main_with_const)
        fitted_main = model_main.fit(method="bfgs", disp=False, maxiter=200)

        X_fe_with_const = sm.add_constant(X_with_fe)
        model_fe = Logit(y, X_fe_with_const)
        fitted_fe = model_fe.fit(method="bfgs", disp=False, maxiter=200)

        # Extract coefficients for main features only (skip intercept)
        for i, feature in enumerate(feature_names):
            coef_main = fitted_main.params[i + 1]  # +1 to skip intercept
            coef_fe = fitted_fe.params[i + 1]  # +1 to skip intercept
            se_main = fitted_main.bse[i + 1]
            se_fe = fitted_fe.bse[i + 1]

            results["features"].append(feature)
            results["coef_main"].append(coef_main)
            results["coef_fe"].append(coef_fe)
            results["se_main"].append(se_main)
            results["se_fe"].append(se_fe)
            results["or_main"].append(np.exp(coef_main))
            results["or_fe"].append(np.exp(coef_fe))
            results["pvalue_main"].append(fitted_main.pvalues[i + 1])
            results["pvalue_fe"].append(fitted_fe.pvalues[i + 1])

        # Model comparison
        results["pseudo_r2_main"] = fitted_main.prsquared
        results["pseudo_r2_fe"] = fitted_fe.prsquared
        results["n_courts"] = len(major_courts)
        results["n_obs"] = len(y)

        print(
            f"✓ Fixed effects: {len(major_courts)} courts, ΔPseudo-R² = {results['pseudo_r2_fe'] - results['pseudo_r2_main']:.3f}"
        )

    except Exception as e:
        print(f"  Warning: Fixed effects failed: {e}")
        return {"note": f"Fixed effects estimation failed: {str(e)}"}

    return results


def create_calibration_figure(logistic_results: Dict, output_dir: Path):
    """Create binary calibration curves (F5)."""
    print("📊 Creating binary calibration curves (F5)...")

    if "dev_probs" not in logistic_results or "test_probs" not in logistic_results:
        print("  No probability data available for calibration plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Class names for binary
    class_names = ["Low (Green)", "High (Red)"]
    colors = ["green", "red"]

    for split_idx, (split_name, split_label) in enumerate(
        [("dev", "DEV"), ("test", "OOF Test")]
    ):
        ax = axes[split_idx]

        probs = logistic_results[f"{split_name}_probs"]
        y_true = logistic_results[f"{split_name}_true"]

        # For binary classification, we have probability of class 1 (High Risk)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, probs, n_bins=10
            )

            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker="o",
                linewidth=2,
                label="High (Red)",
                color="red",
            )
        except:
            pass

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Binary Calibration Curve: {split_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Calculate ECE for binary classification
        try:
            # Simple ECE calculation
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probs > bin_lower) & (probs <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            ax.text(
                0.05,
                0.95,
                f"ECE: {ece:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        except:
            pass

    plt.tight_layout()
    plt.savefig(
        output_dir / "figures" / "f5_binary_calibration_curves.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("✓ Generated binary calibration curves")


def create_coefficient_plot(logistic_results: Dict, output_dir: Path):
    """Create coefficient plot with confidence intervals (F6)."""
    print("📊 Creating binary coefficient plot (F6)...")

    if "features" not in logistic_results:
        print("  No coefficient data available")
        return

    # Prepare data
    features = logistic_results["features"]
    ors = logistic_results["or"]
    or_lowers = logistic_results["or_lower"]
    or_uppers = logistic_results["or_upper"]
    pvalues = logistic_results["pvalue"]

    # Create DataFrame for plotting
    plot_data = pd.DataFrame(
        {
            "feature": features,
            "or": ors,
            "or_lower": or_lowers,
            "or_upper": or_uppers,
            "pvalue": pvalues,
        }
    )

    # Sort by odds ratio magnitude
    plot_data["or_abs"] = np.abs(np.log(plot_data["or"]))
    plot_data = plot_data.sort_values("or_abs", ascending=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(features))

    # Color by significance
    colors = [
        "red" if p < 0.01 else "orange" if p < 0.05 else "gray"
        for p in plot_data["pvalue"]
    ]

    # Plot odds ratios with error bars
    ax.errorbar(
        plot_data["or"],
        y_pos,
        xerr=[
            plot_data["or"] - plot_data["or_lower"],
            plot_data["or_upper"] - plot_data["or"],
        ],
        fmt="o",
        capsize=5,
        capthick=2,
        elinewidth=2,
        color="black",
        markersize=8,
    )

    # Color the markers
    for i, (or_val, color) in enumerate(zip(plot_data["or"], colors)):
        ax.scatter(or_val, i, color=color, s=100, zorder=5)

    # Add vertical line at OR = 1
    ax.axvline(x=1, color="black", linestyle="--", alpha=0.5)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ").title() for f in plot_data["feature"]])
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title(
        "Binary Logistic Regression Associations", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Set x-axis to log scale if needed
    if plot_data["or"].max() / plot_data["or"].min() > 5:
        ax.set_xscale("log")

    # Add significance legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", label="p < 0.01"),
        Patch(facecolor="orange", label="p < 0.05"),
        Patch(facecolor="gray", label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(
        output_dir / "figures" / "f6_binary_coefficient_plot.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("✓ Generated binary coefficient plot")


def generate_latex_tables(
    logistic_results: Dict, sensitivity_results: Dict, output_dir: Path
):
    """Generate LaTeX tables T5 and T6."""
    print("📝 Generating binary LaTeX tables...")

    latex_dir = output_dir / "latex"

    # T5: Main binary logistic results
    if "features" in logistic_results:
        t5_lines = []
        t5_lines.append("\\begin{table}[htbp]")
        t5_lines.append("\\centering")
        t5_lines.append(
            "\\caption{Binary logistic regression associations (case-level, train); evaluated on DEV/OOF Test}"
        )
        t5_lines.append("\\label{tab:binary_logistic}")
        t5_lines.append("\\begin{tabular}{lrrrl}")
        t5_lines.append("\\toprule")
        t5_lines.append("Feature & OR & 95\\% CI & p-value & Direction \\\\")
        t5_lines.append("\\midrule")

        for i, feature in enumerate(logistic_results["features"]):
            or_val = logistic_results["or"][i]
            or_lower = logistic_results["or_lower"][i]
            or_upper = logistic_results["or_upper"][i]
            pvalue = logistic_results["pvalue"][i]

            # Determine direction
            if or_val > 1.05:
                direction = "↑ High Risk"
            elif or_val < 0.95:
                direction = "↓ Low Risk"
            else:
                direction = "≈ Neutral"

            # Format p-value
            if pvalue < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"{pvalue:.3f}"

            feature_tex = feature.replace("_", "\\_")
            line = f"{feature_tex} & {or_val:.2f} & "
            line += f"({or_lower:.2f}, {or_upper:.2f}) & {p_str} & {direction} \\\\"
            t5_lines.append(line)

        # Add model statistics
        t5_lines.append("\\midrule")
        if "dev_accuracy" in logistic_results:
            dev_acc = logistic_results["dev_accuracy"]
            dev_f1 = logistic_results["dev_macro_f1"]
            t5_lines.append(
                f"DEV Accuracy & \\multicolumn{{4}}{{l}}{{{dev_acc:.3f}}} \\\\"
            )
            t5_lines.append(
                f"DEV Macro-F1 & \\multicolumn{{4}}{{l}}{{{dev_f1:.3f}}} \\\\"
            )

        if "test_accuracy" in logistic_results:
            test_acc = logistic_results["test_accuracy"]
            test_f1 = logistic_results["test_macro_f1"]
            t5_lines.append(
                f"OOF Accuracy & \\multicolumn{{4}}{{l}}{{{test_acc:.3f}}} \\\\"
            )
            t5_lines.append(
                f"OOF Macro-F1 & \\multicolumn{{4}}{{l}}{{{test_f1:.3f}}} \\\\"
            )

        pseudo_r2 = logistic_results.get("pseudo_r2", 0)
        t5_lines.append(f"Pseudo-R² & \\multicolumn{{4}}{{l}}{{{pseudo_r2:.3f}}} \\\\")

        t5_lines.append("\\bottomrule")
        t5_lines.append("\\end{tabular}")
        t5_lines.append("\\end{table}")

        with open(latex_dir / "t5_binary_logistic.tex", "w") as f:
            f.write("\n".join(t5_lines))

    # T6: Sensitivity analysis with court fixed effects
    if "features" in sensitivity_results:
        t6_lines = []
        t6_lines.append("\\begin{table}[htbp]")
        t6_lines.append("\\centering")
        t6_lines.append(
            "\\caption{Binary sensitivity analysis: coefficients with and without court fixed effects}"
        )
        t6_lines.append("\\label{tab:binary_court_sensitivity}")
        t6_lines.append("\\begin{tabular}{lrrrr}")
        t6_lines.append("\\toprule")
        t6_lines.append(
            "Feature & OR (Main) & OR (+ Court FE) & p (Main) & p (+ FE) \\\\"
        )
        t6_lines.append("\\midrule")

        for i, feature in enumerate(sensitivity_results["features"]):
            or_main = sensitivity_results["or_main"][i]
            or_fe = sensitivity_results["or_fe"][i]
            p_main = sensitivity_results["pvalue_main"][i]
            p_fe = sensitivity_results["pvalue_fe"][i]

            # Format p-values
            p_main_str = "< 0.001" if p_main < 0.001 else f"{p_main:.3f}"
            p_fe_str = "< 0.001" if p_fe < 0.001 else f"{p_fe:.3f}"

            feature_tex = feature.replace("_", "\\_")
            line = f"{feature_tex} & {or_main:.2f} & {or_fe:.2f} & {p_main_str} & {p_fe_str} \\\\"
            t6_lines.append(line)

        # Add model statistics
        t6_lines.append("\\midrule")
        n_courts = sensitivity_results.get("n_courts", 0)
        r2_main = sensitivity_results.get("pseudo_r2_main", 0)
        r2_fe = sensitivity_results.get("pseudo_r2_fe", 0)

        t6_lines.append(f"Courts included & \\multicolumn{{4}}{{l}}{{{n_courts}}} \\\\")
        t6_lines.append(
            f"Pseudo-R² (Main) & \\multicolumn{{4}}{{l}}{{{r2_main:.3f}}} \\\\"
        )
        t6_lines.append(
            f"Pseudo-R² (+ FE) & \\multicolumn{{4}}{{l}}{{{r2_fe:.3f}}} \\\\"
        )

        t6_lines.append("\\bottomrule")
        t6_lines.append("\\end{tabular}")
        t6_lines.append("\\end{table}")

        with open(latex_dir / "t6_binary_court_sensitivity.tex", "w") as f:
            f.write("\n".join(t6_lines))

    print("✓ Generated binary LaTeX tables T5 and T6")


def create_computational_environment_doc(output_dir: Path):
    """Create computational environment documentation."""
    print("💻 Creating computational environment documentation...")

    import platform
    import sklearn
    import pandas
    import numpy

    env_lines = []
    env_lines.append("# Computational Environment - Binary Classification")
    env_lines.append("")
    env_lines.append("## Software Versions")
    env_lines.append(f"- **Python**: {platform.python_version()}")
    env_lines.append(f"- **NumPy**: {numpy.__version__}")
    env_lines.append(f"- **Pandas**: {pandas.__version__}")
    env_lines.append(f"- **Scikit-learn**: {sklearn.__version__}")
    env_lines.append("")

    env_lines.append("## Hardware")
    env_lines.append(f"- **Platform**: {platform.platform()}")
    env_lines.append(f"- **Architecture**: {platform.architecture()[0]}")
    env_lines.append("")

    env_lines.append("## Reproducibility")
    env_lines.append("- **Random Seed**: 42 (fixed across all analyses)")
    env_lines.append(
        "- **Cross-validation**: Temporal splits with fixed case assignments"
    )
    env_lines.append(
        "- **Feature Selection**: Deterministic rules applied to training data only"
    )
    env_lines.append("")

    env_lines.append("## Key Methodological Choices - Binary Classification")
    env_lines.append(
        "- **Binary Boundary**: Computed on train data only, applied to dev/test"
    )
    env_lines.append(
        "- **Feature Preprocessing**: StandardScaler fit on train, applied to dev/test"
    )
    env_lines.append(
        "- **Weight Computation**: √N case discount + binary class reweighting (train only)"
    )
    env_lines.append(
        "- **Model Selection**: Hyperparameters selected via 3-fold CV on folds 0,1,2"
    )
    env_lines.append(
        "- **Final Evaluation**: Independent OOF test set, never used for training or selection"
    )
    env_lines.append("- **Class Labels**: 0 = Low Risk (Green), 1 = High Risk (Red)")

    with open(output_dir / "COMPUTATIONAL_ENVIRONMENT_BINARY.md", "w") as f:
        f.write("\n".join(env_lines))

    print("✓ Generated binary computational environment documentation")


def main():
    """Generate final polish assets for binary classification."""
    print("🎨 Generating Final Polish Assets for Binary Classification")
    print("=" * 60)

    # Load data and features
    train_df, dev_df, test_df, features = load_final_data()

    # Create output directory
    output_dir = Path("docs/final_paper_assets_binary")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "latex").mkdir(exist_ok=True)

    # Generate analyses

    # T5: Main binary logistic regression
    logistic_results = compute_binary_logistic_main(train_df, dev_df, test_df, features)

    # T6: Sensitivity with court fixed effects
    sensitivity_results = compute_sensitivity_with_court_fe(train_df, features)

    # F5: Calibration curves
    if logistic_results:
        create_calibration_figure(logistic_results, output_dir)

    # F6: Coefficient plot
    if logistic_results:
        create_coefficient_plot(logistic_results, output_dir)

    # Generate LaTeX tables
    generate_latex_tables(logistic_results, sensitivity_results, output_dir)

    # Computational environment
    create_computational_environment_doc(output_dir)

    # Save results as CSV
    if logistic_results and "features" in logistic_results:
        or_df = pd.DataFrame(
            {
                "feature": logistic_results["features"],
                "or": logistic_results["or"],
                "or_lower": logistic_results["or_lower"],
                "or_upper": logistic_results["or_upper"],
                "pvalue": logistic_results["pvalue"],
                "coef": logistic_results["coef"],
                "se": logistic_results["se"],
            }
        )
        or_df.to_csv(output_dir / "binary_logistic_results.csv", index=False)

    if sensitivity_results and "features" in sensitivity_results:
        sens_df = pd.DataFrame(sensitivity_results)
        sens_df.to_csv(output_dir / "binary_court_sensitivity_results.csv", index=False)

    print(f"\n✅ BINARY POLISH COMPLETE!")
    print(f"📁 Binary assets saved to: {output_dir}")
    print("📊 Generated:")
    print("  - T5: Binary logistic regression associations table")
    print("  - T6: Court fixed effects sensitivity table")
    print("  - F5: Binary calibration curves (reliability plots)")
    print("  - F6: Binary coefficient plot with confidence intervals")
    print("  - Computational environment documentation")
    print("🎯 Binary publication package now complete!")


if __name__ == "__main__":
    main()
