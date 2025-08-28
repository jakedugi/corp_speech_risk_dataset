# Court-Based Identity Suppression Implementation

## ✅ **Complete Implementation of Court-Based Identity Suppression**

### **Key Strategy Implemented:**
- **PRIMARY GROUP:** `court_id` for identity suppression
- **FALLBACK:** State code extracted from `court_id` (letters only)
- **FINAL FALLBACK:** Global mean for unseen courts/states
- **SELECTION:** Models selected based on **suppressed MCC**, not raw MCC
- **TRAINING:** Always on raw features (never train on suppressed)
- **EVALUATION:** Both raw and suppressed metrics reported side-by-side

---

## 🎯 **Core Implementation Features**

### **1. Court/State Extraction Logic**
```python
def extract_state_from_court_id(self, court_ids: np.ndarray) -> np.ndarray:
    """Extract state codes from court_id (letters in court_id)."""
    states = []
    for court_id in court_ids:
        # Extract letters from court_id (e.g., 'ca9' -> 'ca', 'nyed' -> 'nyed')
        state_code = ''.join([c for c in str(court_id).lower() if c.isalpha()])
        states.append(state_code if state_code else 'UNK')
    return np.array(states)
```

### **2. Hierarchical Suppression Strategy**
```python
def compute_suppression_means(self, X_train: np.ndarray, court_ids: np.ndarray):
    # Court-level means (minimum 10 samples per court)
    self.court_means = {}
    for court in np.unique(court_ids):
        court_mask = (court_ids == court)
        if court_mask.sum() >= 10:  # Minimum threshold
            self.court_means[court] = X_train[court_mask].mean(axis=0)

    # State-level means (minimum 25 samples per state)
    self.state_means = {}
    for state in np.unique(states):
        state_mask = (states == state)
        if state_mask.sum() >= 25:  # Higher threshold for state fallback
            self.state_means[state] = X_train[state_mask].mean(axis=0)

    # Global mean as final fallback
    self.global_mean = X_train.mean(axis=0)
```

### **3. Suppression Application (Evaluation Only)**
```python
def apply_identity_suppression(self, X: np.ndarray, court_ids: np.ndarray) -> np.ndarray:
    X_suppressed = X.copy()
    states = self.extract_state_from_court_id(court_ids)

    for i, (court_id, state) in enumerate(zip(court_ids, states)):
        # Try court-level suppression first
        if court_id in self.court_means:
            X_suppressed[i] -= self.court_means[court_id]
        # Fallback to state-level suppression
        elif state in self.state_means:
            X_suppressed[i] -= self.state_means[state]
        # Final fallback to global mean
        else:
            X_suppressed[i] -= self.global_mean

    return X_suppressed
```

---

## 🔄 **Nested CV with Suppressed Selection**

### **Inner Grid Search:**
1. **Fit model** on raw inner-train features
2. **Compute suppression means** on inner-train only
3. **Score candidate** on inner-val **suppressed** features
4. **Select hyperparameters** that maximize **suppressed MCC** (not raw)

### **Outer Evaluation:**
1. **Refit model** on raw outer-train with chosen hyperparameters
2. **Calibrate** using Platt scaling on raw training data
3. **Compute RAW metrics** on outer-val
4. **Compute SUPPRESSED metrics** by applying suppression to outer-val
5. **Record ΔMCC** = raw - suppressed

### **Selection Gates:**
- **ΔMCC ≤ 0.02** (generalization guardrail)
- **ECE ≤ 0.08** (calibration quality on suppressed view)

---

## 📊 **Dual Evaluation Metrics**

### **CV Results (Both Views):**
```python
# RAW metrics (for reporting)
'oof_mcc_raw': max_mcc_raw,
'oof_auc_raw': oof_auc_raw,
'brier_score_raw': brier_score_raw,
'ece_raw': ece_raw,

# SUPPRESSED metrics (PRIMARY for selection)
'oof_mcc_suppressed': max_mcc_suppressed,  # PRIMARY SELECTION METRIC
'oof_auc_suppressed': oof_auc_suppressed,
'brier_score_suppressed': brier_score_suppressed,
'ece_suppressed': ece_suppressed,

# Delta metrics
'delta_mcc': delta_mcc,
'delta_auc': delta_auc
```

### **Test Results (Both Views):**
```python
# RAW test metrics
'test_mcc_raw': test_operating_metrics_raw['mcc'],
'test_auc_raw': test_auc_raw,

# SUPPRESSED test metrics
'test_mcc_suppressed': test_operating_metrics_suppressed['mcc'],
'test_auc_suppressed': test_auc_suppressed,

# Delta and comparison metrics
'delta_mcc_test': delta_mcc_test,
'delta_auc_test': delta_auc_test
```

---

## 🚀 **Usage Commands**

### **Scout Pass (Court-Based Selection):**
```bash
uv run python scripts/train_validated_features_with_embeddings.py \
    --data-dir data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage \
    --output-dir results/court_suppressed_scout \
    --fold 4 \
    --scout-pass
```

### **Full Grid Search (Court-Based Selection):**
```bash
uv run python scripts/train_validated_features_with_embeddings.py \
    --data-dir data/final_stratified_kfold_splits_binary_quote_balanced_with_graphsage \
    --output-dir results/court_suppressed_comprehensive \
    --fold 4 \
    --grid-search
```

---

## 📋 **Selection Logic**

### **Primary Selection Metric:**
**CV Suppressed MCC** - Models ranked by performance on court-suppressed features

### **Gates Applied:**
1. **CV Δ(MCC) ≤ 0.02** - Generalization check
2. **CV ECE (suppressed) ≤ 0.08** - Calibration quality
3. **Model simplicity** - Tie-break by number of features

### **Final Reporting:**
Both **RAW** and **SUPPRESSED** performance side-by-side:
```
✅ lr_l1_E+3 evaluation completed
   CV MCC (SUPPRESSED): 0.6234 (PRIMARY SELECTION)
   CV AUC (RAW): 0.7891 ± 0.0234
   CV AUC (SUPPRESSED): 0.7654 ± 0.0198
   Test MCC (RAW): 0.6456
   Test MCC (SUPPRESSED): 0.6398
   CV Δ(MCC): 0.015 (PASS ≤ 0.02)
   CV ECE (suppressed): 0.065 (PASS ≤ 0.08)
```

---

## 🔬 **Why Court (Not Era) Suppression**

### **Rationale:**
1. **Temporal splits already handle era drift** → era suppression adds little value
2. **Court style persists across time** → subtracts the identity signal we want to remove
3. **Court overlap between train/val** → can always compute meaningful court means
4. **Era suppression risks empty means** → if outer-val era doesn't appear in train

### **Paper-Ready Narrative:**
> "Because folds are time-ordered, we suppress court (not era) style by subtracting train-only court means at evaluation; this tests identity-robust performance and calibration without refitting the model."

---

## 🎯 **Implementation Guardrails**

### **✅ Never Train on Suppressed Features:**
- Models always fitted on **raw features**
- Suppression applied **only at evaluation time**
- Means computed on **training fold only**

### **✅ Fold-Local Suppression:**
- No global precomputation (prevents leakage)
- Court means computed **per CV fold**
- Test suppression uses **final training set means**

### **✅ Hierarchical Fallback:**
1. **Court-level** (≥10 samples)
2. **State-level** (≥25 samples)
3. **Global mean** (final fallback)

### **✅ Selection Transparency:**
- **CV suppressed MCC** drives hyperparameter selection
- **Both views reported** for transparency
- **Delta metrics** quantify style dependence

**Ready for identity-robust legal NLP evaluation! 🏛️**
