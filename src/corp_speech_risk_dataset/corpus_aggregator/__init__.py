"""
Corpus Aggregator Module

This module provides case-level aggregation from quote-level features and outcomes.
It takes QuoteFeatures and Outcomes as input and produces CaseVector JSONL files.

Main components:
- Case-level feature aggregation (mean, quantiles, density statistics)
- Position-based feature extraction
- Modeling and prediction at case level
- CLI interface for case aggregation workflows
"""

from .modeling.dataset import (
    ThresholdSpec,
    FeatureConfig,
    DEFAULT_THRESHOLDS,
)
from .modeling.models import (
    build_classification_models,
    evaluate_models_cv,
)
from .modeling import reporting
from .aggregation.run_minimal_case_prediction_from_mirror import (
    main as run_minimal_prediction,
)
from .aggregation.run_case_prediction_with_existing_infra import (
    main as run_prediction_with_infra,
)

__all__ = [
    "ThresholdSpec",
    "FeatureConfig",
    "DEFAULT_THRESHOLDS",
    "build_classification_models",
    "evaluate_models_cv",
    "reporting",
    "run_minimal_prediction",
    "run_prediction_with_infra",
]
