#!/usr/bin/env python3
"""
Leave-One-Case-Out (LOCO) analysis to test quote-level predictive ability.
Tests if features learned from other cases can predict quotes from a held-out case.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")


def leave_one_case_out_analysis(df, features):
    """Test feature generalization using leave-one-case-out."""
    results = []
    case_ids = df["case_id"].unique()

    for i, test_case in enumerate(case_ids):
        print(f"  Testing case {i+1}/{len(case_ids)}: {test_case}")

        # Split data
        train_mask = df["case_id"] != test_case
        test_mask = df["case_id"] == test_case

        # Skip if test case too small
        if test_mask.sum() < 10:
            continue

        # Prepare data
        X_train = df.loc[train_mask, features].fillna(0)
        y_train = df.loc[train_mask, "outcome_bin"]
        X_test = df.loc[test_mask, features].fillna(0)
        y_test = df.loc[test_mask, "outcome_bin"]

        # Skip if no variation in test (will be true for all)
        if y_test.nunique() == 1:
            test_outcome = y_test.iloc[0]
        else:
            test_outcome = y_test.mode()[0]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        try:
            lr = LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            )
            lr.fit(X_train_scaled, y_train)

            # Predict
            y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
            y_pred = lr.predict(X_test_scaled)

            # Metrics (at case level since all quotes have same outcome)
            case_pred_proba = y_pred_proba.mean()  # Average prediction for case
            case_pred = int(case_pred_proba > 0.5)

            results.append(
                {
                    "case_id": test_case,
                    "true_outcome": test_outcome,
                    "n_quotes": test_mask.sum(),
                    "pred_proba": case_pred_proba,
                    "pred_class": case_pred,
                    "correct": case_pred == test_outcome,
                    "confidence_std": y_pred_proba.std(),  # Variation in predictions within case
                    "prop_positive_preds": (y_pred == 1).mean(),
                }
            )
        except Exception as e:
            print(f"    Error: {e}")
            continue

    return pd.DataFrame(results)


def cross_case_clustering(df, features, n_clusters=5):
    """Cluster quotes ignoring case ID and analyze outcome distribution."""
    # Prepare data
    X = df[features].fillna(0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensions for better clustering
    pca = PCA(n_components=min(10, len(features)))
    X_pca = pca.fit_transform(X_scaled)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    # Analyze clusters
    df["cluster"] = clusters
    cluster_analysis = []

    for cluster_id in range(n_clusters):
        cluster_mask = df["cluster"] == cluster_id
        cluster_data = df[cluster_mask]

        cluster_analysis.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_mask.sum(),
                "n_cases": cluster_data["case_id"].nunique(),
                "outcome_mean": cluster_data["outcome_bin"].mean(),
                "outcome_std": cluster_data.groupby("case_id")["outcome_bin"]
                .first()
                .std(),
                "high_risk_cases": (
                    cluster_data.groupby("case_id")["outcome_bin"].first() == 1
                ).sum(),
                "low_risk_cases": (
                    cluster_data.groupby("case_id")["outcome_bin"].first() == 0
                ).sum(),
            }
        )

    return pd.DataFrame(cluster_analysis), df


def feature_discrimination_within_risk(df, features):
    """Test if features discriminate between cases within same risk level."""
    results = {}

    for risk_level in [0, 1]:
        risk_cases = df[df["outcome_bin"] == risk_level]["case_id"].unique()

        if len(risk_cases) < 2:
            continue

        # For each pair of cases at same risk level
        discrimination_scores = []

        for i in range(min(10, len(risk_cases))):  # Limit comparisons
            for j in range(i + 1, min(10, len(risk_cases))):
                case1 = risk_cases[i]
                case2 = risk_cases[j]

                # Get quotes from each case
                quotes1 = df[df["case_id"] == case1][features].fillna(0)
                quotes2 = df[df["case_id"] == case2][features].fillna(0)

                if len(quotes1) < 5 or len(quotes2) < 5:
                    continue

                # Create binary labels (case1=0, case2=1)
                X = pd.concat([quotes1, quotes2])
                y = [0] * len(quotes1) + [1] * len(quotes2)

                # Can features distinguish between these same-risk cases?
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    lr = LogisticRegression(max_iter=1000, random_state=42)
                    lr.fit(X_scaled, y)
                    y_pred = lr.predict_proba(X_scaled)[:, 1]

                    auc = roc_auc_score(y, y_pred)
                    discrimination_scores.append(auc)
                except:
                    continue

        results[f"risk_{risk_level}"] = {
            "mean_discrimination": (
                np.mean(discrimination_scores) if discrimination_scores else 0.5
            ),
            "std_discrimination": (
                np.std(discrimination_scores) if discrimination_scores else 0.0
            ),
            "n_comparisons": len(discrimination_scores),
        }

    return results


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
            if i >= 20000:  # Smaller sample for LOCO
                break
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {df['case_id'].nunique()} cases")
    print(f"Outcome distribution: {df.groupby('outcome_bin').size()}")

    # Test 1: Leave-One-Case-Out
    print("\n1. Running Leave-One-Case-Out analysis...")
    loco_results = leave_one_case_out_analysis(df, passed_features)

    # Test 2: Cross-case clustering
    print("\n2. Running cross-case clustering...")
    cluster_results, df_clustered = cross_case_clustering(df, passed_features)

    # Test 3: Within-risk discrimination
    print("\n3. Testing feature discrimination within risk levels...")
    within_risk_results = feature_discrimination_within_risk(df, passed_features)

    # Save results
    output_path = results_dir / "QUOTE_CONTENT_ANALYSIS.md"
    with open(output_path, "w") as f:
        f.write("# Quote Content Predictive Analysis\n\n")
        f.write(
            "Testing if features capture quote-level content independent of case context.\n\n"
        )

        # LOCO Results
        f.write("## 1. Leave-One-Case-Out (LOCO) Analysis\n\n")
        f.write(
            "Train on N-1 cases, predict held-out case based on quote features.\n\n"
        )

        if len(loco_results) > 0:
            case_accuracy = loco_results["correct"].mean()
            f.write(
                f"**Case-level accuracy**: {case_accuracy:.1%} ({loco_results['correct'].sum()}/{len(loco_results)})\n"
            )
            f.write(
                f"**Random baseline**: {max(loco_results['true_outcome'].mean(), 1-loco_results['true_outcome'].mean()):.1%}\n\n"
            )

            # Confusion matrix
            f.write("### Confusion Matrix (Case Level)\n\n")
            f.write("| True \\ Predicted | Low Risk | High Risk |\n")
            f.write("|------------------|----------|----------|\n")

            for true_val in [0, 1]:
                true_label = "Low Risk" if true_val == 0 else "High Risk"
                row = f"| {true_label} | "
                for pred_val in [0, 1]:
                    count = (
                        (loco_results["true_outcome"] == true_val)
                        & (loco_results["pred_class"] == pred_val)
                    ).sum()
                    row += f"{count} | "
                f.write(row + "\n")

            # Confidence analysis
            f.write("\n### Prediction Confidence\n\n")
            f.write(
                f"- Mean confidence (std within case): {loco_results['confidence_std'].mean():.3f}\n"
            )
            f.write(
                f"- Cases with high confidence (std < 0.1): {(loco_results['confidence_std'] < 0.1).sum()}\n"
            )
            f.write(
                f"- Cases with mixed predictions: {(loco_results['confidence_std'] > 0.2).sum()}\n"
            )

        # Clustering Results
        f.write("\n## 2. Case-Blind Clustering Analysis\n\n")
        f.write("Clusters formed using quote features only (ignoring case ID).\n\n")
        f.write("| Cluster | Size | Cases | Outcome Mean | High Risk | Low Risk |\n")
        f.write("|---------|------|-------|--------------|-----------|----------|\n")

        for _, row in cluster_results.iterrows():
            f.write(
                f"| {row['cluster_id']} | {row['size']} | {row['n_cases']} | "
                f"{row['outcome_mean']:.3f} | {row['high_risk_cases']} | {row['low_risk_cases']} |\n"
            )

        # Within-risk discrimination
        f.write("\n## 3. Within-Risk Level Discrimination\n\n")
        f.write(
            "Can features distinguish between different cases at the same risk level?\n\n"
        )

        for risk_level, results in within_risk_results.items():
            risk_label = "Low Risk" if risk_level.endswith("0") else "High Risk"
            f.write(f"**{risk_label} Cases**:\n")
            f.write(
                f"- Mean discrimination AUC: {results['mean_discrimination']:.3f} ± {results['std_discrimination']:.3f}\n"
            )
            f.write(f"- Comparisons made: {results['n_comparisons']}\n\n")

        f.write("### Interpretation\n\n")
        f.write(
            "- **LOCO accuracy > baseline**: Features capture generalizable patterns\n"
        )
        f.write("- **Cluster outcome variation**: Features separate risk levels\n")
        f.write("- **Within-risk AUC > 0.5**: Features capture case-specific style\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
