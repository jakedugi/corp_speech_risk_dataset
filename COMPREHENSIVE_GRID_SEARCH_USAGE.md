# Comprehensive Grid Search Implementation

## ✅ **ALL 13 Model Variants Implemented**

### **Block A: 3 Validated Features Only (7 variants)**
1. `lr_l2_3VF` - L2 Logistic Regression (lbfgs)
2. `lr_l1_3VF` - L1 Logistic Regression (saga)
3. `lr_elasticnet_3VF` - ElasticNet Logistic Regression (saga)
4. `svm_linear_3VF` - Linear SVM (squared_hinge + l2)
5. `mlr_enhanced_3VF` - MLR Enhanced (ElasticNet wrapper)
6. `mlr_balanced_3VF` - MLR Balanced (ElasticNet wrapper)
7. `polr_3VF` - POLR (if mord available)

### **Block B: Embedding Variants (6 variants)**
8. `lr_l2_E` - L2 LR on Legal-BERT embeddings only
9. `lr_l1_E` - L1 LR on Legal-BERT embeddings only
10. `lr_elasticnet_E` - ElasticNet LR on Legal-BERT embeddings only
11. `lr_l2_E+3` - L2 LR on Legal-BERT + 3 validated scalars
12. `lr_l1_E+3` - L1 LR on Legal-BERT + 3 validated scalars
13. `lr_elasticnet_E+3` - ElasticNet LR on Legal-BERT + 3 validated scalars

---

## 🔍 **Scout Pass vs Full Pass Strategy**

### **Scout Pass (--scout-pass)**
**Purpose:** Initial screening with reduced grids
**Grids:** C ∈ {0.1, 1, 10}, l1_ratio ∈ {0.1, 0.9}
**Config Count:** ~24 configs (before folds)

### **Full Pass (--grid-search)**
**Purpose:** Complete optimization for promoted models
**Grids:** C ∈ {0.01, 0.1, 1, 10}, l1_ratio ∈ {0.1, 0.5, 0.9}
**Config Count:** ~80 configs (before folds)

---

## 🎯 **Global Settings (Fixed Across All Runs)**

```python
# CV Strategy
cv_strategy = GroupKFold(n_splits=3)  # Case-level splits
scoring = 'roc_auc'  # Grid search scoring (MCC optimized post-hoc)

# Fixed Parameters
max_iter = 2000          # 5000 for SVM
tol = 1e-4
class_weight = 'balanced'  # Always balanced (not gridded)
random_state = 42

# Calibration
method = 'sigmoid'       # Platt scaling post-training
cv_calibration = 3       # Inner CV for calibration
```

---

## 📊 **Domain-Standard Evaluation Suite**

### **Primary Metrics (Ranked by Importance):**
1. **MCC (Primary)** - Matthews Correlation Coefficient
2. **AUROC (Secondary)** - Area Under ROC Curve
3. **PR-AUC** - Precision-Recall AUC (imbalance focus)
4. **Calibration Quality:**
   - Brier Score (mean squared error of probabilities)
   - ECE (Expected Calibration Error) - target ≤ 0.08

### **Operating Point Metrics (at MCC-optimal threshold):**
- Precision, Recall, Specificity
- Confusion Matrix (TP/FP/TN/FN)

### **Generalization Check (Identity Suppression):**
- Δ(raw - suppressed) ≤ 0.02 for MCC
- Case/era-level centering for embeddings and scalars

---

## 🚀 **Usage Commands**

### **1. Scout Pass (Initial Screening)**
```bash
uv run python scripts/train_validated_features_with_embeddings.py \
    --data-dir data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage \
    --output-dir results/scout_pass_screening \
    --fold 4 \
    --scout-pass
```

### **2. Full Grid Search (Complete Optimization)**
```bash
uv run python scripts/train_validated_features_with_embeddings.py \
    --data-dir data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage \
    --output-dir results/full_grid_optimization \
    --fold 4 \
    --grid-search
```

### **3. Quick Test (Default Params)**
```bash
uv run python scripts/train_validated_features_with_embeddings.py \
    --data-dir data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage \
    --output-dir results/quick_test \
    --fold 4
```

---

## 📋 **Selection Criteria (Automatic)**

Models are automatically evaluated against:

1. **Identity-Robust MCC:** Primary ranking metric
2. **Generalization Check:** Δ(raw - suppressed) ≤ 0.02
3. **Calibration Quality:** ECE ≤ 0.08
4. **Stability:** Consistent performance across CV folds

### **Selection Rule:**
```
Simplest model that:
✅ Maximizes identity-suppressed MCC
✅ Keeps Δ(raw - suppressed) ≤ 0.02
✅ Achieves ECE ≤ 0.08 post-calibration
✅ Shows stable CV performance
```

---

## 📊 **Output Format**

### **Results Table (sorted by test_mcc):**
| Model | Feature Config | CV MCC | Test MCC | AUROC | PR-AUC | Brier | ECE | Δ(MCC) | Criteria |
|-------|----------------|--------|----------|-------|--------|-------|-----|--------|----------|
| lr_l1_E+3 | E+3 | 0.6234 | 0.6456 | 0.8012 | 0.7123 | 0.1234 | 0.0654 | 0.015 | ✅ PASS |

### **Console Output:**
```
✅ lr_l1_E+3 evaluation completed
   CV MCC: 0.6234 (PRIMARY)
   CV AUC: 0.7891 ± 0.0234
   CV PR-AUC: 0.7123
   Test MCC: 0.6456
   Test AUC: 0.8012
   Test ECE: 0.0654
   Optimal threshold: 0.347
   Δ(MCC): 0.015 (PASS ≤ 0.02)
   ECE: 0.065 (PASS ≤ 0.08)
```

---

## 🔧 **Implementation Features**

### **✅ High-ROI Grid Strategy:**
- **Tight C range:** {0.01, 0.1, 1, 10} covers under/over-regularization
- **Minimal l1_ratio:** {0.1, 0.5, 0.9} spans L2 → L1 spectrum
- **Fixed solvers:** lbfgs/saga as appropriate (no solver grids)
- **Fixed class_weight:** Always 'balanced' (no class-weight grids)
- **Lean SVM:** squared_hinge + l2 only (no loss/penalty grids)

### **✅ Scout → Full Promotion Logic:**
1. **Scout Pass:** Run reduced grids on all 13 variants
2. **Promotion:** Identify models within 1% MCC of best performer
3. **Full Pass:** Expand to complete grids only for promoted models
4. **Selection:** Apply criteria (Δ≤0.02, ECE≤0.08) for final choice

### **✅ Identity-Suppressed Evaluation:**
- **Embeddings:** Case-level centering (subtract case mean)
- **Scalars:** Case-level centering (except binary disclaimers_present)
- **Generalization Check:** Δ(raw - suppressed) ≤ 0.02

---

## 💡 **Paper Narrative (2-line summary):**

> "We evaluated 13 linear model variants across interpretable features and Legal-BERT embeddings using stratified case-held-out cross-validation. Model selection prioritized identity-robust MCC (Δ≤0.02), calibration quality (ECE≤0.08), and domain-standard metrics (AUROC, PR-AUC), ensuring generalizable risk assessment beyond case-specific legal style."

**Ready for high-impact legal NLP research! 🚀**
