#!/usr/bin/env python3
"""Quick test of POLAR pipeline on a small subset of data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger
from corp_speech_risk_dataset.fully_interpretable.polar_pipeline import (
    POLARConfig,
    train_polar_cv,
    train_final_polar_model,
)

# Test on fold 0 only with reduced hyperparameters
config = POLARConfig(
    kfold_dir="data/final_stratified_kfold_splits_FINAL_CLEAN",
    output_dir="runs/polar_test",
    hyperparameter_grid={
        "C": [0.1, 1.0],  # Just 2 values for quick test
        "solver": ["lbfgs"],
        "max_iter": [200],
        "tol": [1e-4],
    },
    compute_tertiles=False,  # Labels already exist
    seed=42,
)

# Quick check of data structure
logger.info("Checking data structure...")
train_df = pd.read_json(
    "data/final_stratified_kfold_splits_FINAL_CLEAN/fold_0/train.jsonl",
    lines=True,
    nrows=10,
)

# Check for label fields
label_fields = [col for col in train_df.columns if "bucket" in col or "class" in col]
logger.info(f"Label fields found: {label_fields}")

# Check interpretable features
interp_fields = [col for col in train_df.columns if col.startswith("interpretable_")]
logger.info(f"Found {len(interp_fields)} interpretable features")

# Check raw_features
if "raw_features" in train_df.columns:
    sample_raw = train_df.iloc[0]["raw_features"]
    if isinstance(sample_raw, dict):
        logger.info(f"raw_features has {len(sample_raw)} fields")

logger.info("\nRunning quick POLAR test...")
try:
    # Override to process only first fold
    config.kfold_dir = "data/final_stratified_kfold_splits_FINAL_CLEAN"

    # Mock single fold by creating temp structure
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy just fold 0
        fold0_src = Path(config.kfold_dir) / "fold_0"
        fold0_dst = Path(tmpdir) / "fold_0"
        shutil.copytree(fold0_src, fold0_dst)

        # Update config
        config.kfold_dir = tmpdir

        # Run CV on single fold
        cv_results = train_polar_cv(config)

        logger.info("\nTest completed successfully!")
        logger.info(f"Results saved to: {config.output_dir}")

except Exception as e:
    logger.error(f"Test failed: {e}")
    import traceback

    traceback.print_exc()
