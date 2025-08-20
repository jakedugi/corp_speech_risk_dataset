# POLAR Pipeline Update Summary

## Overview
The POLAR (Proportional Odds Logistic Regression) training pipeline has been updated to work with the new 3-fold cross-validation setup with adaptive out-of-fold (OOF) test set.

## Key Changes Implemented

### 1. **New Weighting Strategy**
- Implemented `compute_tempered_alpha_weights()` function that uses:
  - **Case support weight**: `clip(n_c^(-α), s_min, s_max)` with α=0.5 (square-root discount)
  - **Tempered class weight**: `clip((1/p_k*)^β, c_min, c_max)` with β=0.5
  - **Combined weight**: `w_q = case_support × class_weight`, normalized to mean=1
- Default parameters:
  - α = 0.5 (square-root discount)
  - β = 0.5 (tempered inverse frequency)
  - Support clip: s_min=0.25, s_max=4.0
  - Class clip: c_min=0.5, c_max=2.0

### 2. **Pre-computed Tertile Boundaries**
- Pipeline now loads pre-computed tertile boundaries from `per_fold_metadata.json`
- Each fold has its own tertile edges (q1, q2) computed on its training set
- Labels are generated using these fixed boundaries (NOT recomputed)

### 3. **3-Fold CV for Hyperparameter Search**
- Only folds 0, 1, 2 are used for hyperparameter search
- Each fold's specific tertile boundaries and class weights are used
- Cross-validation metrics are aggregated only from these 3 folds

### 4. **Final Model Training**
- Final model is trained on fold 3's train + dev data
- Uses fold 3's pre-computed tertile boundaries and class weights
- Calibration is performed on a held-out subset
- Final evaluation is done on the OOF test set

### 5. **Data Structure**
```
data/final_stratified_kfold_splits_adaptive_oof/
├── fold_0/           # CV fold
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── fold_1/           # CV fold
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── fold_2/           # CV fold
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── fold_3/           # Final training fold
│   ├── train.jsonl
│   └── dev.jsonl
├── oof_test/         # Out-of-fold test set
│   └── test.jsonl
└── per_fold_metadata.json  # Pre-computed boundaries & weights
```

## Usage

To run the updated pipeline:

```bash
# Using default settings
uv run python scripts/run_polar_cv.py \
    --output-dir runs/polar_final_adaptive \
    --max-categories 50

# With custom parameters
uv run python scripts/run_polar_cv.py \
    --output-dir runs/polar_experiment \
    --kfold-dir data/final_stratified_kfold_splits_adaptive_oof \
    --dev-tail-frac 0.20 \
    --min-dev-cases 3 \
    --min-dev-quotes 150 \
    --embargo-days 90 \
    --min-cal-n 100 \
    --iso-bins 30 \
    --max-categories 50
```

## Key Features Preserved
- Column governance (interpretable features only)
- Temporal DEV splitting with embargo period
- Cumulative isotonic calibration
- Comprehensive evaluation metrics
- Paper asset generation from final OOF results

## Important Notes
1. **DO NOT RECALCULATE TERTILE BOUNDARIES** - Always use the pre-computed boundaries from `per_fold_metadata.json`
2. **Weighting is applied at quote level** - Each quote gets a weight based on its case size and class
3. **Class weights can be pre-computed or calculated** - The pipeline prefers pre-computed fold-specific weights when available
4. **Final model uses fold 3 data** - This ensures no data leakage while maximizing training data for the production model

## Files Modified
1. `src/corp_speech_risk_dataset/fully_interpretable/polar_pipeline.py`:
   - Added `compute_tempered_alpha_weights()` function
   - Updated CV to use only folds 0-2
   - Modified final training to use fold 3
   - Added loading of pre-computed boundaries and weights

2. `scripts/run_polar_cv.py`:
   - Updated default data directory to `data/final_stratified_kfold_splits_adaptive_oof`

## Validation
A test script is provided at `scripts/test_polar_pipeline.py` to verify:
- Data loading and structure
- Tertile boundary application
- Weight computation
- Fold structure integrity

Run with: `uv run python scripts/test_polar_pipeline.py`
