"""
Corpus Temporal CV Module

This module provides temporal cross-validation and model training functionality.
It takes QuoteFeatures and Outcomes as input and produces model artifacts and OOF metrics.

Main components:
- Temporal CV splitting with leakage prevention
- Multiple model trainers (MLP, LR, CORAL, etc.)
- Metrics calculation and reporting
- CLI interface for running CV experiments
"""

from .splits import main as run_cv_splits
from .trainers.frozen_mlp_trainer import PyTorchMLPEvaluator
from .trainers.optimized_features_trainer import OptimizedModelEvaluator
from .trainers.final_lr_trainer import main as run_final_lr_training
from .temporal_cv import TemporalCVConfig, _ensure_dev_tail

__all__ = [
    "run_cv_splits",
    "PyTorchMLPEvaluator",
    "OptimizedModelEvaluator",
    "run_final_lr_training",
    "TemporalCVConfig",
    "_ensure_dev_tail",
]
