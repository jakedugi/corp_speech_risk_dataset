#!/usr/bin/env python3
"""Debug POLAR implementation issues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from corp_speech_risk_dataset.fully_interpretable.polar_pipeline import (
    load_fold_data,
    compute_class_weights,
    compute_alpha_normalized_weights,
    prepare_features,
)
from corp_speech_risk_dataset.fully_interpretable.models import POLR


def debug_polr():
    """Debug the POLR implementation step by step."""

    print("=" * 60)
    print("DEBUGGING POLR IMPLEMENTATION")
    print("=" * 60)

    # Load a fold's data
    fold_dir = Path("data/final_stratified_kfold_splits_FINAL_CLEAN/fold_0")
    print(f"\n1. Loading data from {fold_dir}")

    train_df, val_df, test_df = load_fold_data(fold_dir)
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df)}")

    # Map labels
    print(f"\n2. Mapping labels from coral_pred_bucket:")
    label_map = {"low": 0, "medium": 1, "high": 2}
    if "coral_pred_bucket" in train_df.columns:
        train_df["y"] = train_df["coral_pred_bucket"].map(label_map)
        val_df["y"] = val_df["coral_pred_bucket"].map(label_map)
        test_df["y"] = test_df["coral_pred_bucket"].map(label_map)
        print("   ✓ Labels mapped successfully")
    else:
        print(f"   Available columns: {list(train_df.columns)[:10]}...")
        raise ValueError("No coral_pred_bucket found")

    # Check label distribution
    print(f"\n   Label distribution in training:")
    print(train_df["y"].value_counts().sort_index())

    # Check case distribution
    print(f"\n3. Case distribution:")
    case_counts = train_df.groupby("case_id").size()
    print(f"   Total cases: {len(case_counts)}")
    print(
        f"   Quotes per case - mean: {case_counts.mean():.2f}, min: {case_counts.min()}, max: {case_counts.max()}"
    )

    # Compute class weights
    print(f"\n4. Computing class weights:")
    class_weights = compute_class_weights(train_df["y"].values)
    for cls, weight in class_weights.items():
        print(f"   Class {cls}: {weight:.4f}")

    # Compute alpha-normalized weights
    print(f"\n5. Computing alpha-normalized weights:")
    weights_train, alpha, weight_stats = compute_alpha_normalized_weights(
        train_df, class_weights
    )
    print(f"   Alpha: {alpha:.4f}")
    print(f"   Mean weight: {weights_train.mean():.6f} (should be 1.0)")
    print(f"   Weight range: [{weights_train.min():.4f}, {weights_train.max():.4f}]")

    # Prepare features
    print(f"\n6. Preparing features:")
    X_train, preprocessor = prepare_features(train_df, fit_preprocessor=True)
    print(f"   Feature shape: {X_train.shape}")
    print(f"   Feature type: {type(X_train)}")
    if hasattr(X_train, "values"):
        # It's a DataFrame
        print(f"   Sample feature values (first row, first 5 features):")
        print(f"   {X_train.values[0, :5]}")
    else:
        # It's a numpy array
        print(f"   Sample feature values (first row, first 5 features):")
        print(f"   {X_train[0, :5]}")

    # Test POLR model
    print(f"\n7. Testing POLR model:")
    try:
        model = POLR(C=1.0, max_iter=100)
        print("   Created POLR instance")

        # Fit with small subset
        n_samples = min(100, len(train_df))
        # Convert to numpy if needed
        X_array = X_train.values if hasattr(X_train, "values") else X_train
        X_subset = X_array[:n_samples]
        y_subset = train_df["y"].values[:n_samples]
        weights_subset = weights_train[:n_samples]

        print(f"   Fitting on {n_samples} samples...")
        print(f"   X shape: {X_subset.shape}, y shape: {y_subset.shape}")
        model.fit(X_subset, y_subset, sample_weight=weights_subset)
        print("   ✓ Model fitted successfully!")

        # Test predictions
        print("   Testing predictions...")
        preds = model.predict(X_subset[:5])
        print(f"   Sample predictions: {preds}")

        probs = model.predict_proba(X_subset[:5])
        print(f"   Sample probabilities shape: {probs.shape}")
        print(f"   First sample probabilities: {probs[0]}")
        print(f"   Probabilities sum: {probs[0].sum():.6f}")

        # Test interpretable odds ratios
        print("   Testing interpretable odds ratios...")
        odds_ratios = model.get_odds_ratios()
        print(f"   Odds ratios shape: {odds_ratios.shape}")
        print(f"   Sample odds ratios (first 5 features): {odds_ratios[:5]}")
        print(f"   Log odds (coefficients, first 5): {model.coef_[:5]}")
        print(
            f"   ✓ Odds ratios = exp(coefficients): {np.allclose(odds_ratios, np.exp(model.coef_))}"
        )

    except Exception as e:
        print(f"   ✗ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    debug_polr()
