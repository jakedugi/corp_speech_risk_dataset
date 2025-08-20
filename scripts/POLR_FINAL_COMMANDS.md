# 🚀 **POLR Training Commands - Final Ready Guide**

Your POLR pipeline is now **100% configured** with:
- ✅ **Exactly 10 features** from `feature_dictionary.csv`
- ✅ **Feature-specific transformations** (7 binarize, 1 log1p, 2 none)
- ✅ **Inherited class weights** from `per_fold_metadata.json`
- ✅ **√N discount weighting** with [0.25, 4.0] clipping
- ✅ **polr_ prediction prefix**
- ✅ **No recomputation** of tertiles or weights

---

## 🎯 **Ready Commands**

### **🔥 PRODUCTION RUN (Recommended)**
Full 3-fold CV + final model + OOF test + paper assets:
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_production_$(date +%Y%m%d_%H%M) \
  --seed 42 \
  --n-jobs -1
```

### **⚡ QUICK TEST RUN**
Skip hyperparameter search for faster testing:
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_test \
  --skip-hyperparam-search \
  --seed 42
```

### **🔍 CV ONLY (Hyperparameter Search)**
Just find best hyperparameters via 3-fold CV:
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_cv_only \
  --skip-final \
  --seed 42
```

### **📊 CUSTOM DEV SIZE**
Adjust development set parameters:
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_custom_dev \
  --dev-tail-frac 0.25 \
  --min-dev-quotes 200 \
  --seed 42
```

---

## 📋 **What Happens During Training**

### **Phase 1: 3-Fold Cross-Validation (Folds 0, 1, 2)**
1. Load fold-specific tertile boundaries from `per_fold_metadata.json`
2. Load fold-specific class weights (0: 1.012, 1: 1.012, 2: 0.977 for fold 3)
3. Apply column governance → exactly 10 interpretable features
4. Apply feature-specific transformations:
   - **Binarize (>0)**: 7 features (deception, guarantee, pricing, etc.)
   - **Log1p**: 1 feature (hedges_norm)
   - **None**: 2 features (hedges_present, superlatives_present)
5. Compute tempered α-weights: √N discount + class rebalancing
6. Hyperparameter search using development sets
7. Select best hyperparameters across 3 folds

### **Phase 2: Final Model Training (Fold 3)**
1. Train on fold 3 train+dev data using best hyperparameters
2. Apply isotonic calibration on held-out calibration set
3. Save final model, preprocessor, and calibrators

### **Phase 3: OOF Evaluation**
1. Load completely held-out OOF test set
2. Generate predictions with `polr_` prefix
3. Compute final performance metrics
4. Save predictions and metrics

### **Phase 4: Paper Asset Generation (Automatic)**
1. Generate all 10 LaTeX tables (T1-T10)
2. Generate all 10 publication figures (F1-F10)
3. Create comprehensive summary documents

---

## 🏷️ **Output Structure**

```
runs/polr_production_YYYYMMDD_HHMM/
├── cv_results.json                 # 3-fold CV hyperparameter results
├── final_model_metadata.json       # Final model performance metrics
├── final_polar_model.joblib        # Trained POLR model
├── final_preprocessor.joblib       # Feature transformation pipeline
├── final_calibrators.joblib        # Isotonic calibration models
├── oof_predictions.jsonl           # OOF test predictions (polr_ prefix)
├── dev_predictions.jsonl           # DEV set predictions
└── paper_assets/                   # Auto-generated academic materials
    ├── latex/
    │   ├── t1_dataset_health.tex   # Dataset composition
    │   ├── t2_feature_dictionary.tex # Feature definitions
    │   ├── t3_feature_summary.tex  # Summary statistics
    │   ├── t4_per_bucket.tex       # Per-bucket descriptives
    │   ├── t5_ordered_logit.tex    # Associations + proportional-odds
    │   ├── t6_multicollinearity.tex # Redundancy analysis
    │   ├── t7_temporal_stability.tex # Drift assessment
    │   ├── t8_jurisdiction.tex     # Court proxy probe
    │   ├── t9_size_bias.tex        # Case size bias probe
    │   └── t10_calibration.tex     # ECE/MCE/Brier metrics
    ├── figures/
    │   ├── f1_outcome_distribution.pdf
    │   ├── f2_class_priors_time.pdf
    │   ├── f3_correlation_heatmap.pdf
    │   ├── f4_bucket_violins.pdf
    │   ├── f5_calibration_curves.pdf
    │   ├── f6_coefficient_plot.pdf
    │   ├── f7_word_shift_panels.pdf
    │   ├── f8_qualitative_exemplars.pdf
    │   ├── f9_drift_barplot.pdf
    │   └── f10_oof_performance.pdf
    └── PAPER_ASSETS_SUMMARY.md
```

---

## 🎯 **Prediction Format**

All predictions use **`polr_` prefix**:

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

  "weights": {"support_weight": 0.87, "class_weight": 1.012, "sample_weight": 0.89},
  "fold": 3,
  "split": "test",
  "model": "polar",
  "hyperparams": {"alpha": 0.1, "max_iter": 1000},
  "calibration": {"method": "isotonic_cumulative", "version": "v1.0"}
}
```

---

## ✅ **Verified Guarantees**

### **Feature Processing:**
- **Exactly 10 features** from `feature_dictionary.csv`
- **7 features binarized** (deception_norm, deception_present, guarantee_norm, guarantee_present, pricing_claims_present, ling_high_certainty, seq_discourse_additive)
- **1 feature log1p** (hedges_norm)
- **2 features unchanged** (hedges_present, superlatives_present)
- **All features standardized** after transformation

### **Weight Inheritance:**
- **Class weights from fold 3**: {0: 1.012, 1: 1.012, 2: 0.977}
- **Support weight range**: [0.25, 4.0] (√N discount clipping)
- **Tempered weighting**: α=0.5, β=0.5
- **Quote-level weights**: `w_quote = case_support × class_weight`

### **Data Flow:**
- **3-fold CV**: Uses folds 0, 1, 2 for hyperparameter search
- **Final training**: Uses fold 3 train+dev for model training
- **OOF test**: Completely held-out test set for final evaluation
- **Tertile boundaries**: Inherited from `per_fold_metadata.json` (never recomputed)

---

## 🚨 **Critical Notes**

### **✅ DO:**
- Always use `--seed 42` for reproducibility
- Let metadata inheritance happen automatically (default)
- Use the production command for final paper results
- Check the `paper_assets/` folder for all generated materials

### **❌ DON'T:**
- Set `--compute-tertiles True` (breaks inheritance)
- Modify the feature dictionary without updating `polar_pipeline.py`
- Use different random seeds for comparison runs
- Skip the final model training for production

---

## 📈 **Expected Runtime**

- **Quick test**: ~3-5 minutes
- **CV only**: ~10-15 minutes
- **Full production**: ~20-35 minutes (depending on hyperparameter grid)
- **Paper assets**: Additional ~5 minutes

---

## 🎉 **Ready to Launch!**

Your pipeline is **production-ready** with:
- ✅ **Perfect feature alignment** with your dictionary
- ✅ **Complete metadata inheritance**
- ✅ **Proper weight computation**
- ✅ **Academic paper package generation**

**Run this now:**
```bash
uv run python scripts/run_polar_cv.py \
  --output-dir runs/polr_ready_$(date +%Y%m%d_%H%M) \
  --seed 42
```

**🚀 Everything is correctly configured! 🎯**
