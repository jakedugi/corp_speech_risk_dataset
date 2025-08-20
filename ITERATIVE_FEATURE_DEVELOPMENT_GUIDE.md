# Complete Guide: Iterative Feature Development for Class Discrimination

## 🎯 Goal
Rapidly develop interpretable features that can discriminate between all risk classes, especially the challenging **Class 0 (low-risk)** cases, using our unified pipeline with automatic testing and pruning.

## 🚀 Quick Start: Test New Features

### 1. **Add Features to `features.py`**

Add your new interpretable features in the appropriate extraction methods:

```python
# In src/corp_speech_risk_dataset/fully_interpretable/features.py

# Add new lexicons to RISK_LEXICONS
RISK_LEXICONS.update({
    "your_new_category": {
        "term1", "term2", "term3"
    }
})

# Or add features in _extract_low_risk_signals() method
def _extract_low_risk_signals(self, text, tokens, quote_tokens, base_features):
    # ... existing code ...

    # YOUR NEW FEATURE:
    your_pattern = re.compile(r"your_regex_pattern", re.I)
    your_matches = your_pattern.findall(text)
    features["your_feature_name"] = len(your_matches)
    features["your_feature_norm"] = len(your_matches) / max(len(tokens), 1)

    return features
```

### 2. **Quick Test (Single File)**

```bash
# Test your new features on a small sample
python scripts/iterative_feature_development.py \
    --iteration test_v1 \
    --input "data/raw/doc_1000001_text_stage15.jsonl" \
    --sample-size 1000 \
    --quick-test \
    --test-class-discrimination
```

### 3. **Check Results**

```bash
# View the results
cat docs/feature_development/iteration_test_v1/iteration_test_v1_summary.md

# Check class 0 discriminators
cat docs/feature_development/iteration_test_v1/class_0_discrimination.csv | head -20
```

## 📊 Full Iteration Workflow

### Phase 1: Extract and Analyze Features

```bash
# Run full analysis on larger sample
python scripts/iterative_feature_development.py \
    --iteration 1 \
    --input "data/raw/doc_*_text_stage15.jsonl" \
    --sample-size 25000 \
    --test-class-discrimination \
    --auto-governance-update
```

This will:
- Extract all 142+ features from your data
- Test discriminative power (MI, Kruskal-Wallis)
- Check class 0 separation specifically
- Test for size bias and leakage
- Generate governance updates for failed features

### Phase 2: Review and Refine

```python
# Analyze the results
import pandas as pd

# Load discrimination results
class0_df = pd.read_csv("docs/feature_development/iteration_1/class_0_discrimination.csv")

# Find features that help with class 0
good_class0 = class0_df[
    (class0_df['class_0_separation'] > 0.1) &
    (class0_df['class_0_significance'] < 0.05) &
    (class0_df['class_0_auc'] > 0.6)
]

print("🎯 Features that discriminate Class 0:")
print(good_class0[['feature', 'class_0_auc', 'class_0_separation']].head(10))

# Check overall discriminative power
disc_df = pd.read_csv("docs/feature_development/iteration_1/discriminative_power.csv")
top_features = disc_df[
    (disc_df['mutual_info'] > 0.01) &
    (disc_df['kw_pvalue'] < 0.05)
].sort_values('mutual_info', ascending=False)

print("\n📊 Top discriminative features:")
print(top_features[['feature', 'mutual_info', 'kw_pvalue']].head(10))
```

### Phase 3: Update Column Governance

```bash
# Apply governance updates (failed features)
cat docs/feature_development/iteration_1/governance_update_iteration_1.txt

# Manually add the blocked patterns to column_governance.py BLOCKLIST_PATTERNS
```

## 🔍 Feature Quality Thresholds

### ✅ **Passing Criteria** (feature keeps)
- **Discriminative Power**: MI > 0.005, KW p-value < 0.1
- **Class 0 Discrimination**: AUC > 0.55, separation > 0.1, p < 0.05
- **No Size Bias**: |correlation with case_size| < 0.3
- **No Leakage**: No court encoding, outcome correlation < 0.8
- **Data Quality**: Zero% < 95%, Missing% < 20%

### ❌ **Failing Criteria** (feature drops)
- Weak discrimination: MI ≤ 0.005 OR KW p ≥ 0.1
- High sparsity: Zero% ≥ 95%
- High missingness: Missing% ≥ 20%
- Size bias: Strong correlation with case_size
- Court/venue leakage: High variance across courts

## 🎯 Strategies for Class 0 (Low-Risk) Features

The new features specifically target Class 0 discrimination:

### **Safe Harbor & Compliance Features**
```python
# These features should score HIGH for Class 0:
- lr_safe_harbor_norm          # "forward-looking statements", "cautionary"
- lr_non_advice_norm           # "not investment advice", "informational only"
- lr_evidential_norm           # "according to", URLs, SEC refs
- lex_compliance_norm          # "compliant", "ethical", "transparent"
```

### **Protective Language Patterns**
```python
# These indicate cautious, liability-limiting language:
- lr_negated_guarantee_norm    # "we do NOT guarantee"
- lr_hedge_near_guarantee_norm # "may" near "guarantee"
- lr_conditional_sentence_share # "if", "unless", "provided that"
- lr_scope_limiter_norm        # "only", "limited to", "except"
```

### **Composite Ratios for Class 0**
```python
# Higher ratios = lower risk:
- ratio_disclaimer_vs_guarantee   # disclaimers / guarantees
- ratio_compliance_vs_deception   # compliance / deception terms
- ratio_scope_vs_claims          # scope limiters / strong claims
- ratio_future_cautioned         # cautioned futures / bold futures
```

## 🔄 Rapid Iteration Commands

### Test Single New Feature
```bash
# Modify features.py, then:
python scripts/iterative_feature_development.py \
    --iteration quick_test \
    --input "data/raw/doc_1000001_text_stage15.jsonl" \
    --sample-size 500 \
    --quick-test
```

### Full Iteration with Analysis
```bash
# After adding multiple features:
python scripts/iterative_feature_development.py \
    --iteration 2 \
    --input "data/raw/doc_*_text_stage15.jsonl" \
    --sample-size 50000 \
    --test-class-discrimination \
    --auto-governance-update
```

### Compare Iterations
```python
# Track improvement across iterations
def compare_iterations(iter1, iter2):
    df1 = pd.read_csv(f"docs/feature_development/iteration_{iter1}/discriminative_power.csv")
    df2 = pd.read_csv(f"docs/feature_development/iteration_{iter2}/discriminative_power.csv")

    # Features that improved
    merged = df1.merge(df2, on='feature', suffixes=('_old', '_new'))
    improved = merged[merged['mutual_info_new'] > merged['mutual_info_old']]

    print(f"Features improved from {iter1} to {iter2}:")
    print(improved[['feature', 'mutual_info_old', 'mutual_info_new']].head(10))

compare_iterations("1", "2")
```

## 🎪 Production Pipeline Usage

Once you've developed good features through iterations:

### 1. **Extract Features on Full Dataset**
```bash
# Use the unified pipeline for full extraction
python scripts/extract_and_prune_features_pipeline.py \
    --input "data/raw/doc_*_text_stage15.jsonl" \
    --output-dir data/enhanced_final \
    --analysis-output-dir docs/feature_analysis_final \
    --batch-size 1000 \
    --sample-size 100000 \
    --run-pruning \
    --mi-threshold 0.005 \
    --p-threshold 0.1
```

### 2. **Split Enhanced Data**
```bash
# Create k-fold splits with enhanced features
python scripts/stratified_kfold_case_split.py \
    --input "data/enhanced_final/doc_*_text_stage15.jsonl" \
    --output-dir data/final_stratified_kfold_splits_enhanced \
    --n-splits 4 \
    --oof-test-fraction 0.15 \
    --random-seed 42
```

### 3. **Train Models**
```bash
# Train with enhanced feature set
python scripts/run_polr_comprehensive.py \
    --kfold-dir data/final_stratified_kfold_splits_enhanced \
    --run-name enhanced_features_final
```

## 📈 Current Enhanced Feature Set

After adding the new features, we now have **142 total features**:

### **Original Features** (83)
- Base lexicons, linguistic, sequence, structural features
- Original derived ratios and interactions

### **NEW Low-Risk Features** (59)
- **Compliance lexicons**: 18 features
- **Safe harbor patterns**: 9 features
- **Grammar/tense patterns**: 12 features
- **Social media signals**: 8 features
- **Proximity features**: 6 features (negated guarantees, hedge-near-guarantee)
- **Composite ratios**: 6 features for class 0 discrimination

## 🚨 Troubleshooting Class 0 Issues

If Class 0 is still hard to predict after iterations:

### Diagnostic Commands
```python
# Check class 0 specifically
def diagnose_class_0(results_dir):
    class0_df = pd.read_csv(f"{results_dir}/class_0_discrimination.csv")

    # Features with good class 0 AUC
    good_auc = class0_df[class0_df['class_0_auc'] > 0.6]
    print(f"Features with AUC > 0.6 for class 0: {len(good_auc)}")

    # Features with significant separation
    sig_sep = class0_df[
        (class0_df['class_0_significance'] < 0.05) &
        (class0_df['class_0_separation'] > 0.1)
    ]
    print(f"Features with significant separation: {len(sig_sep)}")

    if len(sig_sep) == 0:
        print("🚨 NO FEATURES CAN DISCRIMINATE CLASS 0!")
        print("   Add more features targeting:")
        print("   - Compliance/caution language")
        print("   - Negative/absence indicators")
        print("   - Formal/procedural language")

diagnose_class_0("docs/feature_development/iteration_1")
```

### Emergency Feature Ideas for Class 0
```python
# Add to _extract_low_risk_signals() if still struggling:

# Absence of risky patterns (what's NOT there)
risky_lexicons = ["guarantee", "superlatives", "deception"]
absent_risk_count = sum(
    1 for lex in risky_lexicons
    if base_features.get(f"lex_{lex}_present", 0) == 0
)
features["lr_absent_risk_signals"] = absent_risk_count / len(risky_lexicons)

# Conservative language density
conservative_terms = {"prudent", "conservative", "careful", "measured", "gradual"}
conservative_count = sum(1 for token in tokens if token in conservative_terms)
features["lr_conservative_norm"] = conservative_count / n_tokens

# Procedural/formal language (common in low-risk routine cases)
procedural_terms = {"pursuant", "hereby", "whereas", "aforementioned", "aforelisted"}
procedural_count = sum(1 for token in tokens if token in procedural_terms)
features["lr_procedural_norm"] = procedural_count / n_tokens
```

## 📋 Success Checklist

- [x] **Pipeline extracts 142+ features consistently**
- [x] **New low-risk features added (safe harbor, compliance, etc.)**
- [x] **Automatic testing and pruning integrated**
- [x] **Column governance update automation**
- [ ] **Test on real data and validate class 0 discrimination**
- [ ] **Apply governance updates**
- [ ] **Run full production pipeline**

Your enhanced pipeline is now ready for rapid, authoritative feature development! 🚀
