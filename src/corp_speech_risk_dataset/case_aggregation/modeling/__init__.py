"""Academically rigorous case-level outcome prediction from quote-level risks.

This module aggregates quote-level risk predictions into interpretable case-level
features and trains transparent models to predict litigation outcomes. Designed
for publication-quality analysis with statistical rigor.

Key Features:
- Multiple temporal thresholds (complete case, docket-based, token-based)
- Cross-validation with hyperparameter tuning
- Bootstrap confidence intervals
- Feature importance and selection
- Statistical significance testing
- Fairness and bias analysis
- Publication-ready visualizations and LaTeX tables

Modules:
- dataset: Flexible aggregation with configurable features and thresholds
- models: Interpretable ML with cross-validation, tuning, and ensemble methods
- reporting: Academic-quality figures and statistical analysis
- utils: Efficient data loading and preprocessing
- cli: Comprehensive experiment runner with all options

Example:
    uv run python -m corp_speech_risk_dataset.case_aggregation.modeling.cli \
        --quotes-dir /path/to/predictions \
        --output-dir /path/to/results \
        --use-pred-class \
        --enable-cv \
        --enable-tuning \
        --enable-stats
"""

from __future__ import annotations

__all__ = [
    "utils",
    "dataset",
    "models",
]
