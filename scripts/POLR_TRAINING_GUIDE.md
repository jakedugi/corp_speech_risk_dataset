# 🚀 Complete POLR Training Pipeline Guide

This guide explains how to run the full **POLR (Proportional Odds Logistic Regression)** training pipeline that:
- ✅ **Inherits all metadata** from pre-computed files (no recomputation)
- ✅ **Uses `polr_` prefix** for all output predictions
- ✅ **Runs 3-fold CV** for hyperparameter search (folds 0, 1, 2)
- ✅ **Trains final model** on fold 3 data for calibration
- ✅ **Evaluates on OOF test set** for final metrics
- ✅ **Generates paper assets** automatically after training

---

## 📊 **Data Structure Overview**

The pipeline uses the pre-computed, stratified K-fold splits:

```
data/final_stratified_kfold_splits_adaptive_oof/
├── per_fold_metadata.json          # Pre-computed tertile boundaries & class weights
├── fold_statistics.json            # Methodology and validation info
├── fold_0/                         # CV fold for hyperparameter search
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── fold_1/                         # CV fold for hyperparameter search
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── fold_2/                         # CV fold for hyperparameter search
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── fold_3/                         # Final training fold
│   ├── train.jsonl                 # Combined training data
│   └── dev.jsonl                   # Development set for calibration
└── oof_test/                       # Out-of-fold test set (held-out)
    └── test.jsonl                  # Final evaluation set
```

---

## 🔧 **Key Metadata Inheritance**

The pipeline **strictly inherits** these from `per_fold_metadata.json`:

### **Tertile Boundaries (fold-specific)**
```json
"binning": {
  "fold_edges": {
    "fold_0": [720000.0, 7900000.0],
    "fold_1": [615000.0, 6732156.0],
    "fold_2": [720000.0, 8000000.0],
    "fold_3": [710257.86, 9600000.0]
  }
}
```

### **Class Weights (fold-specific)**
```json
"weights": {
  "fold_3": {
    "class_weights": {"0": 1.012, "1": 1.012, "2": 0.977},
    "support_weight_method": "inverse_sqrt_clipped",
    "support_weight_range": [0.25, 4.0]
  }
}
```

---

## 🎯 **Usage Options**

### **Option 1: Full Training Pipeline (Recommended)**
Runs complete 3-fold CV + final model training + OOF evaluation:

```bash
uv run python scripts/run_polar_cv.py \
  --kfold-dir data/final_stratified_kfold_splits_adaptive_oof \
  --output-dir runs/polr_final_complete \
  --seed 42
```

### **Option 2: Skip Hyperparameter Search**
Uses default hyperparameters, faster execution:

```bash
uv run python scripts/run_polar_cv.py \
  --kfold-dir data/final_stratified_kfold_splits_adaptive_oof \
  --output-dir runs/polr_no_hypersearch \
  --skip-hyperparam-search \
  --seed 42
```

### **Option 3: CV Only (No Final Model)**
Just runs 3-fold CV for hyperparameter search:

```bash
uv run python scripts/run_polar_cv.py \
  --kfold-dir data/final_stratified_kfold_splits_adaptive_oof \
  --output-dir runs/polr_cv_only \
  --skip-final \
  --seed 42
```

### **Option 4: Custom Dev Set Sizes**
Adjust development set parameters (rarely needed):

```bash
uv run python scripts/run_polar_cv.py \
  --kfold-dir data/final_stratified_kfold_splits_adaptive_oof \
  --output-dir runs/polr_custom_dev \
  --dev-tail-frac 0.25 \
  --min-dev-quotes 200 \
  --seed 42
```

---

## ⚡ **Quick Start Commands**

### **🎯 Production Run (Full Pipeline)**
```bash
# Complete training with all features
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_production_$(date +%Y%m%d_%H%M) \
  --seed 42 \
  --n-jobs -1
```

### **🧪 Fast Development Run**
```bash
# Quick test with minimal hyperparameter search
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_dev_test \
  --skip-hyperparam-search \
  --seed 42
```

---

## 📋 **Complete Parameter Reference**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kfold-dir` | `data/final_stratified_kfold_splits_authoritative` | Directory with k-fold splits |
| `--output-dir` | `runs/polar_experiment` | Output directory for results |
| `--compute-tertiles` | `False` | **DO NOT USE** - Always inherit pre-computed |
| `--continuous-target` | `final_judgement_real` | Field for continuous target (if computing) |
| `--seed` | `42` | Random seed for reproducibility |
| `--n-jobs` | `-1` | Parallel jobs (-1 = all cores) |
| `--skip-final` | `False` | Skip final model training (CV only) |
| `--skip-hyperparam-search` | `False` | Use default hyperparameters |
| `--dev-tail-frac` | `0.20` | Starting DEV fraction |
| `--min-dev-cases` | `3` | Minimum DEV cases |
| `--min-dev-quotes` | `150` | Minimum DEV quotes |
| `--require-all-classes` | `False` | Require all 3 classes in DEV |

---

## 🔍 **Output Structure**

After running, your `--output-dir` will contain:

```
runs/polr_production_YYYYMMDD_HHMM/
├── cv_results.json                 # Cross-validation metrics
├── final_model_metadata.json       # Final model performance
├── final_polar_model.joblib        # Trained model
├── final_preprocessor.joblib       # Feature preprocessor
├── final_calibrators.joblib        # Calibration models
├── oof_predictions.jsonl           # OOF test predictions (polr_ prefix)
├── dev_predictions.jsonl           # DEV predictions
└── paper_assets/                   # Auto-generated paper materials
    ├── latex/                      # T1-T10 LaTeX tables
    ├── figures/                    # F1-F10 PDF figures
    └── PAPER_SUMMARY.md            # Complete asset summary
```

---

## 🏷️ **Output Prediction Format**

All predictions use the **`polr_` prefix** as requested:

```json
{
  "case_id": "1:21-cv-01238_ded",
  "text": "The company believes this will improve results...",

  "polr_pred_bucket": "medium",
  "polr_pred_class": 1,
  "polr_confidence": 0.67,
  "polr_class_probs": {
    "low": 0.15,
    "medium": 0.67,
    "high": 0.18
  },
  "polr_prob_low": 0.15,
  "polr_prob_medium": 0.67,
  "polr_prob_high": 0.18,
  "polr_scores": [0.34, 0.78],
  "polr_model_threshold": 0.5,
  "polr_model_buckets": ["low", "medium", "high"],

  "weights": {...},
  "fold": 3,
  "split": "test",
  "model": "polar",
  "hyperparams": {...},
  "calibration": {"method": "isotonic_cumulative", "version": "v1.0"}
}
```

---

## 📈 **Automatic Paper Asset Generation**

The pipeline automatically generates **complete academic paper assets**:

### **10 LaTeX Tables (T1-T10)**
- T1: Dataset health & composition
- T2: Feature dictionary
- T3: Summary statistics
- T4: Per-bucket descriptives
- T5: Ordered logit associations (**with proportional-odds check**)
- T6: Multicollinearity analysis
- T7: Temporal stability (**with year correlations & PSI**)
- T8: Jurisdiction probe (**court extraction from case IDs**)
- T9: Size-bias probe
- T10: Calibration metrics (**ECE/MCE/Brier**)

### **10 Publication Figures (F1-F10)**
- F1: Outcome distribution
- F2: Class priors over time
- F3: Correlation heatmap
- F4: Per-bucket violins
- F5: Calibration curves
- F6: Coefficient plot
- F7: Log-odds word-shift
- F8: Qualitative exemplars
- F9: Drift assessment
- F10: OOF performance

---

## ✅ **Validation & Quality Checks**

The pipeline includes comprehensive validation:

1. **Metadata Inheritance**: Verifies all tertile boundaries and weights are loaded from files
2. **Column Governance**: Uses only approved interpretable features (10 final features)
3. **Temporal Purity**: Maintains strict temporal ordering
4. **Leakage Prevention**: OOF test set completely isolated
5. **Calibration**: Isotonic calibration on development set
6. **Robustness**: Handles edge cases (small dev sets, missing classes)

---

## 🚨 **Important Notes**

### **✅ DO:**
- Always use `--seed 42` for reproducibility
- Let the pipeline inherit all metadata (default behavior)
- Use the default `--kfold-dir` path
- Keep `--compute-tertiles False` (default)

### **❌ DON'T:**
- Set `--compute-tertiles True` (breaks metadata inheritance)
- Modify tertile boundaries manually
- Use different case ID formats
- Skip the final model training for production runs

---

## 🔬 **Advanced Usage**

### **Custom Hyperparameter Grid**
Edit `polar_pipeline.py` to modify the hyperparameter search space:

```python
param_grid = {
    'model__alpha': [0.001, 0.01, 0.1, 1.0],
    'model__max_iter': [1000, 2000],
    # Add custom parameters
}
```

### **Custom Paper Asset Generation**
Generate assets separately after training:

```bash
uv run python scripts/final_paper_assets.py \
  --model-dir runs/polr_production_YYYYMMDD_HHMM \
  --output-dir docs/final_paper_assets
```

---

## 🎯 **Ready Commands for Different Scenarios**

### **📊 Full Academic Paper Run**
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_paper_$(date +%Y%m%d_%H%M) \
  --seed 42 \
  --n-jobs -1
# Generates complete paper package with all tables & figures
```

### **⚡ Quick Validation Run**
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_validation \
  --skip-hyperparam-search \
  --seed 42
# Fast run to validate pipeline and feature set
```

### **🔍 Hyperparameter-Only Run**
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_hyperparam_search \
  --skip-final \
  --seed 42
# Just find best hyperparameters via 3-fold CV
```

---

## 🎉 **Expected Runtime**

- **Quick validation**: ~2-5 minutes
- **Full pipeline**: ~15-30 minutes (depending on hyperparameter search)
- **Paper asset generation**: +5 minutes

---

This pipeline is now **production-ready** with complete metadata inheritance, proper `polr_` prefixing, and automatic generation of all required academic paper assets! 🚀
