# Documentation Updates Summary

## Overview
Updated the academic paper documentation to accurately reflect the corrected pipeline design and methodology after fixing the boundary labeling issues.

## Key Changes Made

### 1. Cross-Validation Methodology Section
**BEFORE**: "Stratified group k-fold cross-validation"
**AFTER**: "Rolling-origin temporal cross-validation"

- Updated section title from "Stratified K-Fold Cross-Validation Analysis" to "Rolling-Origin Temporal Cross-Validation Analysis"
- Replaced composite stratification language with temporal purity emphasis
- Added correct boundary convention: Low: y < e₁; Medium: e₁ ≤ y ≤ e₂; High: y > e₂

### 2. Key Methodological Features
**Updated to reflect**:
- ✅ Train-only tertiles computed per fold
- ✅ Temporal purity with rolling-origin design
- ✅ Support weighting via inverse-√N instead of stratification
- ✅ Correct split naming: Train/VAL within folds, DEV for final tuning, OOF Test for holdout

### 3. Class Weight Analysis
**Added important clarification**:
- Quote distribution appears highly imbalanced **by design**
- Case-level balancing (33/33/33) achieved during binning
- Quote-level imbalance handled via combined weighting scheme

### 4. Distribution Shift Documentation
**Added new section**: "Distribution Shift and Temporal Evolution"
- Documents expected OOF test shift: Medium +35%, High -38%, Low +1%
- Explains this as realistic temporal evolution, not an error
- Validates temporal methodology's ability to detect distribution drift

### 5. Cross-Validation Validation Updates
**Replaced stratification language with temporal focus**:
- Data leakage prevention ✅
- **Temporal purity** (new emphasis)
- **Per-fold tertile consistency** (new)
- **Distribution shift detection** (new)

### 6. Figure Captions and Titles
**Updated throughout to reflect**:
- "3-fold rolling-origin temporal CV" instead of "5-fold cross-validation"
- "Train-only tertiles" emphasis
- "Temporal CV splits" instead of generic "K-fold"

### 7. Statistics Tables
**Updated fold statistics table**:
- Added Final Training fold row
- Added OOF Test row
- Clarified representative numbers
- Added explanatory note about temporal design

### 8. Monitoring and Limitations
**Added new section**:
- PSI monitoring protocols (PSI > 0.25 triggers recalibration)
- Causality vs. association disclaimer
- Temporal context for interpretation

## Code Updates

### scripts/generate_dataset_paper_figures.py
- Updated figure titles and captions
- Corrected terminology throughout
- Enhanced temporal holdouts visualization title

### Verification Status
- ✅ All label verification still passes (0 errors)
- ✅ Boundary handling correct (250 boundary cases fixed)
- ✅ OOF test has all 3 classes (306/978/432 distribution)
- ✅ Temporal purity maintained
- ✅ Support weighting verified

## Final Status

The documentation now accurately reflects:

1. **Corrected Methodology**: Rolling-origin temporal CV with train-only tertiles
2. **Fixed Boundary Logic**: Proper inclusive/exclusive boundary handling
3. **Realistic Distribution Shift**: OOF temporal evolution documented as expected
4. **Proper Weighting**: Case support + class weighting instead of stratification
5. **Monitoring Protocols**: PSI thresholds and recalibration procedures

**Result**: The paper documentation is now aligned with the implemented and verified pipeline, ready for POLR training and academic submission.
