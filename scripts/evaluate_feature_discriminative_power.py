#!/usr/bin/env python3
"""Quick Feature Discriminative Power Analysis

Evaluates how well the 10 interpretable features can discriminate between classes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def load_and_analyze_features():
    """Load data and analyze feature discriminative power."""
    print("🔍 Loading data for feature discriminative analysis...")

    # Load fold_3 train data (largest training set)
    train_file = Path(
        "data/final_stratified_kfold_splits_authoritative/fold_3/train.jsonl"
    )
    df = pd.read_json(train_file, lines=True)

    print(f"Loaded {len(df):,} samples from fold_3/train.jsonl")

    # Get the 10 approved interpretable features
    approved_features = [
        "interpretable_lex_deception_norm",
        "interpretable_lex_deception_present",
        "interpretable_lex_guarantee_norm",
        "interpretable_lex_guarantee_present",
        "interpretable_lex_hedges_norm",
        "interpretable_lex_hedges_present",
        "interpretable_lex_pricing_claims_present",
        "interpretable_lex_superlatives_present",
        "interpretable_ling_high_certainty",
        "interpretable_seq_discourse_additive",
    ]

    # Filter to features that exist in the data
    available_features = [f for f in approved_features if f in df.columns]
    print(f"Available features: {len(available_features)}/10")

    if len(available_features) == 0:
        print("❌ No approved features found in data!")
        return None

    # Extract features and labels
    X = df[available_features].fillna(0)
    y = df["outcome_bin"].astype(int)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().sort_index().to_dict()}")

    return X, y, available_features


def analyze_feature_discriminative_power(X, y, feature_names):
    """Analyze discriminative power of features."""
    print("\n📊 FEATURE DISCRIMINATIVE POWER ANALYSIS")
    print("=" * 60)

    results = {}

    # 1. Per-feature mutual information with target
    print("\n1. MUTUAL INFORMATION SCORES (higher = more discriminative)")
    print("-" * 50)

    mi_scores = {}
    for i, feature in enumerate(feature_names):
        # Convert to discrete bins for MI calculation
        feature_vals = X.iloc[:, i]
        if feature_vals.nunique() > 10:
            # Bin continuous features
            feature_discrete = pd.cut(
                feature_vals, bins=10, labels=False, duplicates="drop"
            )
        else:
            feature_discrete = feature_vals

        mi = mutual_info_score(y, feature_discrete)
        mi_scores[feature] = mi
        print(f"  {feature:<40}: {mi:.4f}")

    results["mutual_info"] = mi_scores

    # 2. Feature means by class
    print("\n2. FEATURE MEANS BY CLASS")
    print("-" * 50)

    class_means = {}
    for class_val in sorted(y.unique()):
        class_mask = y == class_val
        class_data = X[class_mask]
        class_means[class_val] = class_data.mean()

        print(f"\nClass {class_val} (n={class_mask.sum():,}):")
        for feature in feature_names:
            if feature in class_data.columns:
                mean_val = class_data[feature].mean()
                std_val = class_data[feature].std()
                print(f"  {feature:<40}: {mean_val:.4f} ± {std_val:.4f}")

    results["class_means"] = class_means

    # 3. Feature separability analysis
    print("\n3. PAIRWISE CLASS SEPARABILITY")
    print("-" * 50)

    from scipy.stats import ttest_ind

    separability = {}
    class_pairs = [(0, 1), (0, 2), (1, 2)]

    for c1, c2 in class_pairs:
        print(f"\nClass {c1} vs Class {c2}:")
        pair_key = f"{c1}_vs_{c2}"
        separability[pair_key] = {}

        mask1 = y == c1
        mask2 = y == c2

        for feature in feature_names:
            if feature in X.columns:
                vals1 = X.loc[mask1, feature]
                vals2 = X.loc[mask2, feature]

                # T-test for difference in means
                if len(vals1) > 1 and len(vals2) > 1:
                    stat, p_value = ttest_ind(vals1, vals2)
                    effect_size = abs(vals1.mean() - vals2.mean()) / np.sqrt(
                        (vals1.var() + vals2.var()) / 2
                    )

                    separability[pair_key][feature] = {
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "significant": p_value < 0.05,
                    }

                    sig_marker = (
                        "***"
                        if p_value < 0.001
                        else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    )
                    print(
                        f"  {feature:<40}: p={p_value:.4f} {sig_marker}, effect={effect_size:.3f}"
                    )

    results["separability"] = separability

    # 4. Overall discriminative power assessment
    print("\n4. OVERALL ASSESSMENT")
    print("-" * 50)

    # Check if any features can discriminate Class 0 well
    class_0_discriminators = []
    for feature in feature_names:
        if feature in mi_scores and mi_scores[feature] > 0.01:
            # Check if this feature separates Class 0 from others
            c0_vs_others_seps = []
            for pair_key in ["0_vs_1", "0_vs_2"]:
                if pair_key in separability and feature in separability[pair_key]:
                    sep_info = separability[pair_key][feature]
                    if sep_info["significant"] and sep_info["effect_size"] > 0.2:
                        c0_vs_others_seps.append(True)

            if (
                len(c0_vs_others_seps) >= 1
            ):  # Separates Class 0 from at least one other class
                class_0_discriminators.append(feature)

    results["class_0_discriminators"] = class_0_discriminators

    print(f"Features that can discriminate Class 0: {len(class_0_discriminators)}")
    for feature in class_0_discriminators:
        print(f"  - {feature}")

    if len(class_0_discriminators) == 0:
        print("🚨 CRITICAL: NO features can reliably discriminate Class 0!")
        print("   This explains why the model never predicts Class 0")

    # 5. Feature variance analysis
    print("\n5. FEATURE VARIANCE ANALYSIS")
    print("-" * 50)

    low_variance_features = []
    for feature in feature_names:
        if feature in X.columns:
            variance = X[feature].var()
            unique_vals = X[feature].nunique()

            print(f"  {feature:<40}: var={variance:.6f}, unique={unique_vals}")

            if variance < 1e-6 or unique_vals <= 2:
                low_variance_features.append(feature)

    results["low_variance_features"] = low_variance_features

    if low_variance_features:
        print(f"\n⚠️  Low variance features: {low_variance_features}")

    # 6. Linear discriminant analysis
    print("\n6. LINEAR DISCRIMINANT ANALYSIS")
    print("-" * 50)

    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_scaled, y)

        # Transform to discriminant space
        X_lda = lda.transform(X_scaled)

        print(f"LDA components: {X_lda.shape[1]}")
        print(f"Explained variance ratio: {lda.explained_variance_ratio_}")

        # Check separability in LDA space
        lda_sep_score = 0
        for class_val in sorted(y.unique()):
            class_mask = y == class_val
            class_lda = X_lda[class_mask]
            if len(class_lda) > 0:
                # Distance from origin (higher = more separated)
                dist = np.linalg.norm(class_lda.mean(axis=0))
                print(f"Class {class_val} distance from origin: {dist:.4f}")
                lda_sep_score += dist

        results["lda_separability_score"] = lda_sep_score
        print(f"Overall LDA separability score: {lda_sep_score:.4f}")

    except Exception as e:
        print(f"LDA analysis failed: {e}")
        results["lda_separability_score"] = 0

    # 7. Diagnosis and recommendations
    print("\n🎯 DIAGNOSIS AND RECOMMENDATIONS")
    print("=" * 60)

    if len(class_0_discriminators) == 0:
        print("🚨 PRIMARY ISSUE: No features discriminate Class 0")
        print("   → Model cannot learn to predict Class 0")
        print("   → Need features that are specifically different for low-risk cases")

    if results.get("lda_separability_score", 0) < 1.0:
        print("🚨 SECONDARY ISSUE: Poor linear separability")
        print("   → Classes not linearly separable in feature space")
        print("   → Consider non-linear model or feature engineering")

    if len(low_variance_features) > 0:
        print(f"⚠️  {len(low_variance_features)} features have low variance")
        print("   → These features provide little information")

    return results


def main():
    """Quick feature discriminative power evaluation."""
    print("🚀 QUICK FEATURE DISCRIMINATIVE POWER ANALYSIS")
    print("=" * 70)

    try:
        # Load and analyze
        data = load_and_analyze_features()
        if data is None:
            return

        X, y, feature_names = data

        # Analyze discriminative power
        results = analyze_feature_discriminative_power(X, y, feature_names)

        print(f"\n✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()
