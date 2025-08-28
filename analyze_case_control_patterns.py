#!/usr/bin/env python3
"""
Case control analysis: Two specific tests only.
1. Performance with case controls - regression with case ID as control
2. Case vs quote variation - variance ratios
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def test_performance_with_case_controls(df, feature):
    """Test feature performance when controlling for case ID using regression"""

    # Standard AUC without controls
    feature_clean = df[feature].fillna(0)
    standard_auc = roc_auc_score(df["outcome_bin"], feature_clean)

    # Create case ID dummies for control
    case_dummies = pd.get_dummies(df["case_id"], prefix="case")

    # Build feature matrix with case controls
    X_feature_only = feature_clean.values.reshape(-1, 1)
    X_with_controls = np.hstack([X_feature_only, case_dummies.values])
    y = df["outcome_bin"].values

    # Standardize features
    scaler = StandardScaler()
    X_feature_scaled = scaler.fit_transform(X_feature_only)
    X_controls_scaled = scaler.fit_transform(X_with_controls)

    # Model 1: Feature only
    lr_feature = LogisticRegression(max_iter=1000, random_state=42)
    lr_feature.fit(X_feature_scaled, y)
    pred_feature = lr_feature.predict_proba(X_feature_scaled)[:, 1]
    auc_feature_only = roc_auc_score(y, pred_feature)

    # Model 2: Feature + Case ID controls
    lr_controlled = LogisticRegression(max_iter=1000, random_state=42)
    lr_controlled.fit(X_controls_scaled, y)
    pred_controlled = lr_controlled.predict_proba(X_controls_scaled)[:, 1]
    auc_with_controls = roc_auc_score(y, pred_controlled)

    # Model 3: Case ID only (baseline)
    lr_case_only = LogisticRegression(max_iter=1000, random_state=42)
    lr_case_only.fit(case_dummies.values, y)
    pred_case_only = lr_case_only.predict_proba(case_dummies.values)[:, 1]
    auc_case_only = roc_auc_score(y, pred_case_only)

    # Delta: How much does feature add beyond case ID?
    delta_auc = auc_with_controls - auc_case_only

    return {
        "standard_auc": standard_auc,
        "auc_feature_only": auc_feature_only,
        "auc_case_only": auc_case_only,
        "auc_with_controls": auc_with_controls,
        "delta_auc": delta_auc,
        "feature_coefficient": float(
            lr_controlled.coef_[0][0]
        ),  # Feature coefficient in controlled model
    }


def test_case_vs_quote_variation(df, features):
    """Test if features vary more between cases than within cases"""
    results = {}
    for feature in features:  # Test ALL features
        if feature in df.columns:
            # Within-case variance (average variance within each case)
            within_case_var = df.groupby("case_id")[feature].var().mean()

            # Between-case variance (variance of case means)
            case_means = df.groupby("case_id")[feature].mean()
            between_case_var = case_means.var()

            # Ratio: if >>1, features mainly distinguish cases, not quotes
            variance_ratio = between_case_var / (within_case_var + 1e-6)
            results[feature] = variance_ratio

    return results  # Ratios >10 suggest case-level patterns


def main():
    # Load validated features
    results_dir = Path("results/all_features_with_graphsage_comprehensive_validation")
    summary_path = results_dir / "ALL_FEATURES_SUMMARY.md"

    passed_features = []
    with open(summary_path, "r") as f:
        lines = f.readlines()
        in_table = False
        for line in lines:
            line = line.strip()
            if line.startswith("| Rank | Feature | AUC |"):
                in_table = True
                continue
            if in_table and line.startswith("|") and not line.startswith("|---"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    feature = parts[2].strip("`")
                    passed_features.append(feature)

    print(f"Found {len(passed_features)} validated features")

    # Load data
    data_dir = Path(
        "data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage"
    )
    train_path = data_dir / "fold_4" / "train.jsonl"

    records = []
    with open(train_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 50000:  # Limit for memory
                break
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {df['case_id'].nunique()} cases")

    # Test 1: Performance with case controls for ALL features
    print("\nRunning case control regression analysis...")
    case_control_results = {}
    for i, feature in enumerate(passed_features):
        if feature in df.columns:
            print(f"  Testing {i+1}/{len(passed_features)}: {feature}")
            try:
                result = test_performance_with_case_controls(df, feature)
                case_control_results[feature] = result
            except Exception as e:
                print(f"    Error: {e}")
                case_control_results[feature] = {"error": str(e)}

    # Test 2: Case vs quote variation for ALL features
    print("\nRunning variance ratio analysis...")
    variance_results = test_case_vs_quote_variation(df, passed_features)

    # Save results
    output_path = results_dir / "CASE_CONTROL_ANALYSIS.md"
    with open(output_path, "w") as f:
        f.write("# Case Control Analysis Results\n\n")
        f.write(f"Analyzed {len(passed_features)} validated features\n")
        f.write(f"Dataset: {len(df)} quotes from {df['case_id'].nunique()} cases\n\n")

        # Case control results
        f.write("## Test 1: Regression with Case ID Controls\n\n")
        f.write(
            "| Feature | Standard AUC | Feature Only | Case Only | With Controls | Delta AUC | Feature Coef |\n"
        )
        f.write(
            "|---------|-------------|--------------|-----------|---------------|-----------|-------------|\n"
        )

        # Sort by delta AUC
        sorted_features = sorted(
            case_control_results.items(),
            key=lambda x: x[1].get("delta_auc", -1) if "error" not in x[1] else -999,
            reverse=True,
        )

        for feature, result in sorted_features:
            if "error" not in result:
                f.write(
                    f"| `{feature}` | {result['standard_auc']:.3f} | "
                    f"{result['auc_feature_only']:.3f} | {result['auc_case_only']:.3f} | "
                    f"{result['auc_with_controls']:.3f} | {result['delta_auc']:+.3f} | "
                    f"{result['feature_coefficient']:+.3f} |\n"
                )

        # Key insights
        positive_delta = sum(
            1
            for _, r in case_control_results.items()
            if "error" not in r and r["delta_auc"] > 0
        )
        f.write(
            f"\n**Key Finding**: {positive_delta}/{len(case_control_results)} features "
            f"add predictive value beyond case ID\n\n"
        )

        # Variance results
        f.write("## Test 2: Case vs Quote Variance Ratios\n\n")
        f.write("| Feature | Variance Ratio | Pattern Type |\n")
        f.write("|---------|---------------:|-------------|\n")

        sorted_variance = sorted(
            variance_results.items(), key=lambda x: x[1], reverse=True
        )
        for feature, ratio in sorted_variance:
            pattern = "case-level" if ratio > 10 else "quote-level"
            f.write(f"| `{feature}` | {ratio:.2f} | {pattern} |\n")

        # Summary statistics
        high_ratio = sum(1 for r in variance_results.values() if r > 10)
        f.write(
            f"\n**Variance Summary**: {high_ratio}/{len(variance_results)} features "
            f"show primarily case-level patterns (ratio >10)\n\n"
        )

        # Combined interpretation
        f.write("## Combined Interpretation\n\n")
        f.write("Features that:\n")
        f.write("- Have positive Delta AUC → Add value beyond case context\n")
        f.write("- Have low variance ratio (<10) → Capture quote-specific patterns\n")
        f.write("- Have high variance ratio (>10) → Mostly distinguish between cases\n")

    print(f"\nResults saved to: {output_path}")

    # Also save raw JSON results
    json_output = {
        "case_control_results": case_control_results,
        "variance_results": variance_results,
    }
    json_path = results_dir / "case_control_results.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Raw results saved to: {json_path}")


if __name__ == "__main__":
    main()
