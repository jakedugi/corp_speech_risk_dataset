#!/usr/bin/env python3
"""
Quick Case Fingerprinting Diagnostics for Validated Features

Runs the 3 case fingerprinting tests on the validated features:
1. Case-ID Predictability (fingerprinting test)
2. Case→Label Permutation Null (memorization test)
3. Identity Suppression Test (robustness test)

Features tested:
- new4_neutral_to_disclaimer_transition_rate
- lex_disclaimers_present
- feat_interact_hedge_x_guarantee
"""

import sys
import os
import pandas as pd
import numpy as np
import orjson
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Sklearn imports
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Validated features (from corrected_dnt_validation results)
VALIDATED_FEATURES = [
    "feat_new4_neutral_to_disclaimer_transition_rate",
    "feat_lex_disclaimers_present",
    "feat_interact_hedge_x_guarantee",
]


def case_id_predictability(df, feature_cols, group_col="case_id"):
    """Test if features can predict case ID (fingerprinting test)."""
    try:
        # Prepare data
        X = df[feature_cols].fillna(0).values
        g = df[group_col].values
        cases, y = np.unique(g, return_inverse=True)  # labels 0..C-1

        # Skip if too few cases or samples
        if len(cases) < 5 or len(X) < 50:
            return {
                "acc": 0.0,
                "chance": 1.0 / max(len(cases), 1),
                "acc_minus_chance": -1.0,
                "warning": "insufficient_data",
            }

        # Cross-validation with GroupKFold (optimized for speed)
        clf = LogisticRegression(
            max_iter=100, multi_class="auto", random_state=42, solver="liblinear"
        )
        gkf = GroupKFold(n_splits=min(3, len(cases) // 2))
        accs = []

        for tr, va in gkf.split(X, groups=g):
            xs, ys = X[tr], y[tr]
            xv, yv = X[va], y[va]

            # Scale features (sparse-safe)
            scaler = StandardScaler(with_mean=False)
            xs = scaler.fit_transform(xs)
            xv = scaler.transform(xv)

            clf.fit(xs, ys)
            accs.append(accuracy_score(yv, clf.predict(xv)))

        acc = float(np.mean(accs))
        chance = 1.0 / len(cases)
        acc_minus_chance = acc - chance

        return {
            "acc": acc,
            "chance": chance,
            "acc_minus_chance": acc_minus_chance,
            "warning": "case_id_signal_present" if acc_minus_chance > 0.05 else None,
        }

    except Exception as e:
        return {"acc": 0.0, "chance": 1.0, "acc_minus_chance": -1.0, "error": str(e)}


def case_level_auc_single(
    df, score_col, y_col="outcome_bin", group="case_id", agg="mean"
):
    """Compute case-level AUC for a single feature."""
    try:
        x = df.groupby(group)[score_col].agg(agg)
        y = df[[group, y_col]].drop_duplicates(subset=[group]).set_index(group)[y_col]
        y = y.reindex(x.index)

        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 5 or y_clean.nunique() < 2:
            return 0.5

        return float(roc_auc_score(y_clean, x_clean))
    except Exception:
        return 0.5


def permutation_null_case_auc(
    df, score_col, y_col="outcome_bin", group="case_id", n=50
):
    """Test if case-level AUC persists under case→label permutation (memorization test)."""
    try:
        # True case-level AUC
        true_auc = case_level_auc_single(df, score_col, y_col, group)

        # Get unique cases and their labels
        cases = df[group].drop_duplicates().values
        y_map = (
            df[[group, y_col]].drop_duplicates(subset=[group]).set_index(group)[y_col]
        )

        # Run permutation test
        rng = np.random.default_rng(42)
        null_aucs = []

        for i in range(n):
            if i % 10 == 0:
                logger.info(f"    Permutation {i+1}/{n}")
            # Permute case→label mapping (faster vectorized approach)
            perm_cases = rng.permutation(cases)
            y_perm = dict(zip(perm_cases, y_map.values))
            y_perm_series = df[group].map(y_perm)

            # Create temp dataframe with permuted labels (avoid full copy)
            tmp = df[[group, score_col]].copy()
            tmp["_perm_y"] = y_perm_series
            auc = case_level_auc_single(tmp, score_col, "_perm_y", group)
            null_aucs.append(auc)

        null_aucs = [x for x in null_aucs if not np.isnan(x)]

        if not null_aucs:
            return {
                "true_auc": true_auc,
                "null_mean": 0.5,
                "null_ci": (0.5, 0.5),
                "warning": "no_valid_nulls",
            }

        null_mean = float(np.mean(null_aucs))
        null_ci = (
            float(np.percentile(null_aucs, 2.5)),
            float(np.percentile(null_aucs, 97.5)),
        )

        # Check if null distribution is centered around 0.5 (should be for valid features)
        warning = None
        if null_mean > 0.55 or null_ci[0] > 0.55:
            warning = "learning_identity_shortcuts"

        return {
            "true_auc": true_auc,
            "null_mean": null_mean,
            "null_ci": null_ci,
            "warning": warning,
        }

    except Exception as e:
        return {
            "true_auc": 0.5,
            "null_mean": 0.5,
            "null_ci": (0.5, 0.5),
            "error": str(e),
        }


def identity_suppression(df, feature_cols, group="case_id"):
    """Z-score features within cases to remove case-specific identity signals."""
    try:
        z = df[feature_cols].copy()
        for c in feature_cols:
            mu = df.groupby(group)[c].transform("mean")
            sd = df.groupby(group)[c].transform("std").replace(0, 1.0)
            z[c] = (df[c] - mu) / sd
        return z
    except Exception:
        return df[feature_cols].copy()


def identity_suppression_test(df, feature_col, y_col="outcome_bin", group="case_id"):
    """Test if AUC holds after within-case z-scoring (identity removal)."""
    try:
        # Original case-level AUC
        original_auc = case_level_auc_single(df, feature_col, y_col, group)

        # Create z-scored version
        z_scored = identity_suppression(df, [feature_col], group)
        df_z = df.copy()
        df_z[feature_col] = z_scored[feature_col]

        # Z-scored case-level AUC
        z_scored_auc = case_level_auc_single(df_z, feature_col, y_col, group)

        # Calculate retention ratio
        retention_ratio = z_scored_auc / max(original_auc, 1e-6)

        warning = None
        if retention_ratio < 0.7:  # AUC drops by >30%
            warning = "signal_mostly_identity"

        return {
            "original_auc": original_auc,
            "z_scored_auc": z_scored_auc,
            "retention_ratio": retention_ratio,
            "warning": warning,
        }

    except Exception as e:
        return {
            "original_auc": 0.5,
            "z_scored_auc": 0.5,
            "retention_ratio": 1.0,
            "error": str(e),
        }


def run_fingerprinting_diagnostics(data_dir: str, output_dir: str, fold: int = 4):
    """Run fingerprinting diagnostics on validated features."""

    logger.info("=" * 80)
    logger.info("CASE FINGERPRINTING DIAGNOSTICS FOR VALIDATED FEATURES")
    logger.info("=" * 80)

    # Load data
    data_path = Path(data_dir)
    train_path = data_path / f"fold_{fold}" / "train.jsonl"
    dev_path = data_path / f"fold_{fold}" / "dev.jsonl"

    logger.info(f"Loading data from fold {fold}...")
    # Accelerated loading with orjson
    with open(train_path, "rb") as f:
        train_data = [orjson.loads(line) for line in f]
    train_df = pd.DataFrame(train_data)

    with open(dev_path, "rb") as f:
        dev_data = [orjson.loads(line) for line in f]
    dev_df = pd.DataFrame(dev_data)

    # Combine for full dataset
    df = pd.concat([train_df, dev_df], ignore_index=True)

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique cases: {df['case_id'].nunique()}")
    logger.info(
        f"Binary distribution: {dict(df['outcome_bin'].value_counts(normalize=True).sort_index())}"
    )

    # Check features exist
    missing_features = [f for f in VALIDATED_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Run diagnostics for each feature
    results = {}

    for i, feature in enumerate(VALIDATED_FEATURES, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE {i}/{len(VALIDATED_FEATURES)}: {feature}")
        logger.info(f"{'='*60}")

        feature_results = {
            "feature_name": feature,
            "basic_stats": {
                "mean": float(df[feature].mean()),
                "std": float(df[feature].std()),
                "min": float(df[feature].min()),
                "max": float(df[feature].max()),
                "non_null_count": int(df[feature].count()),
            },
        }

        # Test 1: Case-ID Predictability
        logger.info("Running Case-ID Predictability test...")
        case_id_test = case_id_predictability(df, [feature])
        feature_results["case_id_predictability"] = case_id_test

        logger.info(f"  Accuracy: {case_id_test['acc']:.4f}")
        logger.info(f"  Chance: {case_id_test['chance']:.4f}")
        logger.info(f"  Above chance: {case_id_test['acc_minus_chance']:.4f}")
        if case_id_test.get("warning"):
            logger.warning(f"  ⚠️  {case_id_test['warning']}")

        # Test 2: Case→Label Permutation Null
        logger.info("Running Case→Label Permutation Null test...")
        perm_test = permutation_null_case_auc(df, feature)
        feature_results["permutation_null"] = perm_test

        logger.info(f"  True AUC: {perm_test['true_auc']:.4f}")
        logger.info(f"  Null mean: {perm_test['null_mean']:.4f}")
        logger.info(f"  Null CI: {perm_test['null_ci']}")
        if perm_test.get("warning"):
            logger.warning(f"  ⚠️  {perm_test['warning']}")

        # Test 3: Identity Suppression
        logger.info("Running Identity Suppression test...")
        identity_test = identity_suppression_test(df, feature)
        feature_results["identity_suppression"] = identity_test

        logger.info(f"  Original AUC: {identity_test['original_auc']:.4f}")
        logger.info(f"  Z-scored AUC: {identity_test['z_scored_auc']:.4f}")
        logger.info(f"  Retention ratio: {identity_test['retention_ratio']:.4f}")
        if identity_test.get("warning"):
            logger.warning(f"  ⚠️  {identity_test['warning']}")

        # Summary assessment
        warnings_count = sum(
            [
                1
                for test in [case_id_test, perm_test, identity_test]
                if test.get("warning") is not None
            ]
        )

        if warnings_count == 0:
            risk_level = "low"
            logger.info(f"  ✅ Overall: LOW fingerprinting risk")
        elif warnings_count == 1:
            risk_level = "medium"
            logger.info(f"  ⚠️  Overall: MEDIUM fingerprinting risk")
        else:
            risk_level = "high"
            logger.warning(f"  🔴 Overall: HIGH fingerprinting risk")

        feature_results["fingerprinting_risk"] = risk_level
        feature_results["total_warnings"] = warnings_count

        results[feature] = feature_results

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON results (accelerated with orjson)
    json_file = output_path / "validated_features_fingerprinting_diagnostics.json"
    with open(json_file, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    logger.info(f"\nResults saved to: {json_file}")

    # Generate summary report
    report_file = output_path / "VALIDATED_FEATURES_FINGERPRINTING_REPORT.md"

    with open(report_file, "w") as f:
        f.write("# Case Fingerprinting Diagnostics: Validated Features\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Fold**: {fold}\n")
        f.write(f"**Total samples**: {len(df):,}\n")
        f.write(f"**Unique cases**: {df['case_id'].nunique():,}\n\n")

        f.write("## 🔍 Validated Features Tested\n\n")
        for i, feature in enumerate(VALIDATED_FEATURES, 1):
            f.write(f"{i}. `{feature}`\n")
        f.write("\n")

        f.write("## 📊 Fingerprinting Test Results\n\n")
        f.write(
            "| Feature | Case ID<br/>Predictability | Permutation<br/>Null AUC | Identity<br/>Suppression | Risk<br/>Level |\n"
        )
        f.write("|---------|:----------:|:----------:|:----------:|:----------:|\n")

        for feature, result in results.items():
            case_id = result["case_id_predictability"]
            perm = result["permutation_null"]
            identity = result["identity_suppression"]
            risk = result["fingerprinting_risk"]

            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "❓")

            f.write(f"| `{feature}` | ")
            f.write(f"{case_id['acc_minus_chance']:+.3f} | ")
            f.write(f"{perm['null_mean']:.3f} | ")
            f.write(f"{identity['retention_ratio']:.3f} | ")
            f.write(f"{risk_emoji} {risk.upper()} |\n")

        f.write("\n## 🎯 Key Insights\n\n")

        low_risk = sum(1 for r in results.values() if r["fingerprinting_risk"] == "low")
        medium_risk = sum(
            1 for r in results.values() if r["fingerprinting_risk"] == "medium"
        )
        high_risk = sum(
            1 for r in results.values() if r["fingerprinting_risk"] == "high"
        )

        f.write(f"- **Low risk features**: {low_risk}/{len(VALIDATED_FEATURES)}\n")
        f.write(
            f"- **Medium risk features**: {medium_risk}/{len(VALIDATED_FEATURES)}\n"
        )
        f.write(f"- **High risk features**: {high_risk}/{len(VALIDATED_FEATURES)}\n\n")

        f.write("### 🔍 Test Interpretations\n\n")
        f.write(
            "- **Case ID Predictability**: How well features predict which case a quote came from\n"
        )
        f.write("  - Values > +0.05 suggest case fingerprinting\n")
        f.write(
            "- **Permutation Null AUC**: Whether AUC persists when case→label mapping is shuffled\n"
        )
        f.write("  - Values > 0.55 suggest learning identity shortcuts\n")
        f.write(
            "- **Identity Suppression**: AUC retention after within-case z-scoring\n"
        )
        f.write("  - Values < 0.7 suggest signal is mostly case identity\n\n")

        f.write("## 📈 Feature-Level Details\n\n")

        for feature, result in results.items():
            f.write(f"### `{feature}`\n\n")

            stats = result["basic_stats"]
            f.write(f"**Basic Statistics:**\n")
            f.write(f"- Mean: {stats['mean']:.4f}\n")
            f.write(f"- Std: {stats['std']:.4f}\n")
            f.write(f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write(f"- Non-null: {stats['non_null_count']:,}\n\n")

            case_id = result["case_id_predictability"]
            f.write(f"**Case-ID Predictability:**\n")
            f.write(f"- Accuracy: {case_id['acc']:.4f}\n")
            f.write(f"- Chance: {case_id['chance']:.4f}\n")
            f.write(f"- Above chance: {case_id['acc_minus_chance']:+.4f}\n")
            if case_id.get("warning"):
                f.write(f"- ⚠️ Warning: {case_id['warning']}\n")
            f.write("\n")

            perm = result["permutation_null"]
            f.write(f"**Permutation Null:**\n")
            f.write(f"- True case-level AUC: {perm['true_auc']:.4f}\n")
            f.write(f"- Null mean: {perm['null_mean']:.4f}\n")
            f.write(
                f"- Null 95% CI: [{perm['null_ci'][0]:.3f}, {perm['null_ci'][1]:.3f}]\n"
            )
            if perm.get("warning"):
                f.write(f"- ⚠️ Warning: {perm['warning']}\n")
            f.write("\n")

            identity = result["identity_suppression"]
            f.write(f"**Identity Suppression:**\n")
            f.write(f"- Original AUC: {identity['original_auc']:.4f}\n")
            f.write(f"- Z-scored AUC: {identity['z_scored_auc']:.4f}\n")
            f.write(f"- Retention ratio: {identity['retention_ratio']:.4f}\n")
            if identity.get("warning"):
                f.write(f"- ⚠️ Warning: {identity['warning']}\n")
            f.write("\n")

    logger.info(f"Summary report saved to: {report_file}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("FINGERPRINTING DIAGNOSTICS SUMMARY")
    print("=" * 80)

    for feature, result in results.items():
        risk = result["fingerprinting_risk"]
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "❓")
        print(f"{risk_emoji} {feature}: {risk.upper()} risk")

        case_id = result["case_id_predictability"]
        perm = result["permutation_null"]
        identity = result["identity_suppression"]

        print(f"   Case ID predictability: {case_id['acc_minus_chance']:+.3f}")
        print(f"   Permutation null mean: {perm['null_mean']:.3f}")
        print(f"   Identity retention: {identity['retention_ratio']:.3f}")
        print()

    print("=" * 80)

    return results


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run fingerprinting diagnostics on validated features"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to k-fold data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="results/corrected_dnt_validation",
        help="Output directory for results",
    )
    parser.add_argument("--fold", type=int, default=4, help="Fold number to use")

    args = parser.parse_args()

    run_fingerprinting_diagnostics(
        data_dir=args.data_dir, output_dir=args.output_dir, fold=args.fold
    )


if __name__ == "__main__":
    main()
