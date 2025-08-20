# Interpretable Features Analysis Summary

**Analysis Date:** 2025-08-18 16:29:39
**Total Features Analyzed:** 11
**Feature Categories:** 3

## Overview

This analysis covers the interpretable features used for training the POLAR model.
All features have been filtered through column governance to ensure they are:
- Interpretable and auditable
- Free from data leakage
- Legally and ethically appropriate

## Feature Categories

## Statistical Summary

| Category | Features | Avg Mean | Avg Std | Avg Zero% | Avg Skew |
|----------|----------|----------|---------|-----------|----------|
| Lex | 8 | 0.134 | 0.231 | 79.1% | 4.09 |
| Ling | 1 | 0.631 | 1.263 | 69.7% | 2.94 |
| Seq | 2 | 0.669 | 0.484 | 37.5% | 3.39 |

## Key Findings

- **Sparse Features:** 5 features have >80% zero values
- **Highly Skewed:** 8 features show high skewness (|skew| > 2)
- **Most Informative Category:** Ling

## Files Generated

- `latex/feature_summary.tex` - Main summary table
- `latex/feature_explanations.tex` - Feature definitions
- `latex/*_details.tex` - Detailed tables by category
- `figures/feature_distributions.pdf` - Distribution overview
- `figures/category_distributions.pdf` - Category-specific distributions
- `figures/feature_correlations.pdf` - Correlation heatmap
