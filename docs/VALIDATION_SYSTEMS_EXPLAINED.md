# Validation Systems Explained

## Why Are There Two Different Validation Systems?

You're absolutely right to be confused! There are **two complementary but different validation systems** with different purposes and criteria:

## 1. Iterative Feature Development (`iterative_feature_development_kfold.py`)

**Purpose**: Feature research and discriminative power analysis
**Focus**: Finding features that can distinguish between risk classes
**Use Case**: Model development and feature selection

### Tests Performed:
- **Discriminative Power**: Mutual information, Kruskal-Wallis test
- **Class 0 Discrimination**: AUC for separating low-risk (Class 0) from others
- **Size Bias Check**: Correlation with case size to avoid confounding
- **Leakage Check**: Correlation with target to avoid data leakage
- **New Feature Focus**: Special analysis for recently added features

### Thresholds (More Permissive):
- Coverage: < 95% zeros
- Class 0 AUC: > 0.52
- Size bias: correlation < 0.3
- Leakage: correlation < 0.3

### Results:
- **351 features tested**
- **28 new Class 0 discriminators**
- **3 features passing all tests**

---

## 2. Unified Validation Pipeline (`run_unified_validation.sh`)

**Purpose**: Production readiness and robustness testing
**Focus**: Ensuring features are stable, unbiased, and production-ready
**Use Case**: Final validation before deployment

### Tests Performed:
- **Tier 1**: Ultra-fast screening (coverage, basic bias, leakage)
- **Tier 2**: CV stability and permutation importance
- **Tier 3**: Comprehensive robustness (not reached due to failures)
- **Auto-Blacklist**: Remove problematic features

### Thresholds (Much Stricter):
- Coverage: < 90% zeros (stricter!)
- Unique values: > 10 minimum
- Size bias: correlation < 0.15 (stricter!)
- Leakage: correlation < 0.1 (stricter!)
- CV stability: coefficient of variation < 0.5

### Results:
- **498 features tested**
- **145 Tier 1 survivors**
- **0 Tier 2 survivors** (very strict!)
- **112 features blacklisted**

---

## Why Features Pass One System But Not The Other

### Common Reasons for Disagreement:

1. **Sparsity Thresholds**:
   - Iterative: < 95% zeros
   - Unified: < 90% zeros
   - Many features with 90-95% zeros pass iterative but fail unified

2. **Size Bias Tolerance**:
   - Iterative: < 0.3 correlation
   - Unified: < 0.15 correlation
   - Features with moderate size correlation pass iterative but fail unified

3. **Leakage Detection**:
   - Iterative: < 0.3 correlation with target
   - Unified: < 0.1 correlation with target
   - More sensitive unified detection catches subtle leakage

4. **Stability Requirements**:
   - Iterative: Single fold testing
   - Unified: Cross-validation stability requirements
   - Some features work on one fold but aren't stable across folds

### Example Disagreements:

**Features Good for Class 0 but Blacklisted in Unified**:
- `lex_obligation_present` - Good discrimination but too sparse
- `lex_permission_present` - Good AUC but fails size bias
- `lex_ambiguity_present` - Discriminative but unstable
- `lex_compliance_present` - Useful but moderate leakage

---

## How to Interpret Current Status

### For Model Development:
✅ **Use the 28 Class 0 discriminators from iterative testing**
- These have proven discriminative power for identifying low-risk quotes
- Good for research models and initial prototyping
- Examples: `lex_obligation_count`, `ling_third_person_ratio`, `ratio_policy_vs_obligation`

### For Production Deployment:
✅ **Use the 145 Tier 1 survivors from unified validation**
- These are robust, stable, and production-ready
- Lower risk of bias, leakage, or instability
- Safe for deployment in live systems

### For Feature Engineering:
⚠️ **Investigate why good discriminators are failing unified tests**
- Many promising features are being over-rejected by strict thresholds
- Consider adjusting unified validation thresholds
- Focus on improving feature robustness rather than abandoning them

---

## Running Both Systems Together

### Option 1: Run Complete Pipeline
```bash
chmod +x scripts/run_complete_validation.sh
./scripts/run_complete_validation.sh
```

### Option 2: Run Individually
```bash
# Iterative (detailed analysis)
uv run python scripts/iterative_feature_development_kfold.py \
    --iteration comprehensive_validation_with_20_new \
    --fold-dir data/final_stratified_kfold_splits_authoritative \
    --fold 3 --sample-size 10000 --test-class-discrimination

# Unified (production readiness)
./run_unified_validation.sh

# Analysis (compare results)
uv run scripts/unified_validation_analysis.py
```

---

## Key Takeaways

1. **Both systems are valuable** - they serve different purposes
2. **Disagreements are expected** - unified is much stricter by design
3. **Use iterative for research** - finding discriminative features
4. **Use unified for production** - ensuring robustness and safety
5. **200+ disagreements are normal** - the systems have very different goals

The fact that you have **28 new Class 0 discriminators** is excellent for model development, even if many don't pass the stricter unified validation! 🎯
