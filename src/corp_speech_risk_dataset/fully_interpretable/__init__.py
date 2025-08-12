"""
fully_interpretable
===================

Enhanced interpretable models for legal risk classification with state-of-the-art
performance and publication-ready outputs. Provides fully transparent alternatives
to black-box models while maintaining competitive accuracy.

Key Components:
- Advanced Models: POLR, EBM, calibrated classifiers, transparent ensembles
- Rich Features: Risk lexicons, sequence modeling, linguistic analysis
- Interpretability: Feature importance, local explanations, validation experiments
- Publication Support: LaTeX tables, calibration plots, forest plots

Entry Points:
- `fully_interpretable.cli`: Enhanced CLI for training and prediction
- `fully_interpretable.pipeline`: Core training and evaluation pipeline
- `fully_interpretable.models`: Advanced interpretable model implementations
- `fully_interpretable.features`: Sophisticated feature engineering
- `fully_interpretable.interpretation`: Publication-ready visualizations
- `fully_interpretable.validation`: Comprehensive validation experiments
"""

from __future__ import annotations

from .pipeline import (
    InterpretableConfig,
    train_and_eval,
    save_model,
    load_model,
    predict_directory,
    build_dataset,
    make_model,
)

from .models import (
    ProportionalOddsLogisticRegression,
    CalibratedInterpretableClassifier,
    TransparentEnsemble,
    create_ebm_classifier,
)

from .features import (
    InterpretableFeatureExtractor,
    create_feature_matrix,
    RISK_LEXICONS,
    DISCOURSE_MARKERS,
)

from .interpretation import InterpretabilityReport
from .validation import ValidationExperiments

__version__ = "2.0.0"

__all__ = [
    # Core pipeline
    "InterpretableConfig",
    "train_and_eval",
    "save_model",
    "load_model",
    "predict_directory",
    "build_dataset",
    "make_model",
    # Models
    "ProportionalOddsLogisticRegression",
    "CalibratedInterpretableClassifier",
    "TransparentEnsemble",
    "create_ebm_classifier",
    # Features
    "InterpretableFeatureExtractor",
    "create_feature_matrix",
    "RISK_LEXICONS",
    "DISCOURSE_MARKERS",
    # Analysis
    "InterpretabilityReport",
    "ValidationExperiments",
]
