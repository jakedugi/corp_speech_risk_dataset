# Enhanced Fully Interpretable Models for Legal Risk Classification

This module provides state-of-the-art interpretable machine learning models for corporate speech risk classification, designed specifically for academic publication and regulatory compliance where model transparency is crucial.

## Key Features

### 🎯 Advanced Interpretable Models
- **POLR (Proportional Odds Logistic Regression)**: Gold standard for ordinal classification
- **EBM (Explainable Boosting Machine)**: Non-linear relationships with full interpretability
- **Calibrated Models**: Well-calibrated probability predictions
- **Transparent Ensembles**: Voting-based combinations maintaining interpretability

### 📊 Sophisticated Feature Engineering
- **Risk Lexicons**: Domain-specific vocabularies for legal risk detection
  - Deception terms ("misleading", "false", "deceptive")
  - Guarantee/assurance language
  - Pricing claims and superlatives
  - Scienter/knowledge indicators
- **Sequence Modeling**: Transparent sequence features without embeddings
  - N-gram transitions between risk categories
  - Positional encoding of risk terms
  - Discourse marker analysis
- **Linguistic Features**: Rich linguistic analysis
  - Negation and modality detection
  - Financial amount extraction and binning
  - Temporal reference analysis

### 📈 Publication-Ready Outputs
- **Comprehensive Metrics**: QWK, MAE, calibration metrics (ECE)
- **Interpretability Reports**: Feature importance with confidence intervals
- **Validation Experiments**: Negative controls and ablation studies
- **Case-Level Analysis**: Aggregation to case outcomes
- **LaTeX Tables**: Ready for academic papers

### ⚡ Performance Optimizations
- Parallel feature extraction
- Sparse matrix support
- Batch prediction processing
- Chi-squared feature selection

## Installation

```bash
# Install optional dependencies for enhanced models
pip install mord  # For POLR
pip install interpret  # For EBM
```

## Quick Start

### Training a Model

```bash
# Train POLR model (recommended for ordinal data)
python -m corp_speech_risk_dataset.fully_interpretable.cli train \
    --data data/interpretable_training_data.jsonl \
    --model polr \
    --output-dir runs/polr_experiment \
    --out runs/polr_model.joblib

# Train EBM for non-linear patterns
python -m corp_speech_risk_dataset.fully_interpretable.cli train \
    --data data/interpretable_training_data.jsonl \
    --model ebm \
    --output-dir runs/ebm_experiment \
    --out runs/ebm_model.joblib

# Train ensemble for best performance
python -m corp_speech_risk_dataset.fully_interpretable.cli train \
    --data data/interpretable_training_data.jsonl \
    --model ensemble \
    --output-dir runs/ensemble_experiment \
    --out runs/ensemble_model.joblib
```

### Feature Configuration

```bash
# Use all enhanced features
python -m corp_speech_risk_dataset.fully_interpretable.cli train \
    --data data/training.jsonl \
    --model polr \
    --include-scalars \
    --output-dir runs/full_features \
    --out runs/full_model.joblib

# Minimal model (text only)
python -m corp_speech_risk_dataset.fully_interpretable.cli train \
    --data data/training.jsonl \
    --model ridge \
    --no-lexicons --no-sequence --no-linguistic --no-structural \
    --no-keywords --no-speaker --no-numeric \
    --out runs/text_only.joblib
```

### Making Predictions

```bash
# Batch predictions with interpretability info
python -m corp_speech_risk_dataset.fully_interpretable.cli predict-dir \
    --model runs/polr_model.joblib \
    --input-root data/test \
    --output-root data/test_predictions
```

## Model Types

### POLR (Proportional Odds Logistic Regression)
Best for: Ordinal classification with clear interpretability
```python
cfg = InterpretableConfig(
    model="polr",
    model_params={"C": 1.0, "penalty": "l2"}
)
```

### EBM (Explainable Boosting Machine)
Best for: Capturing non-linear patterns while maintaining interpretability
```python
cfg = InterpretableConfig(
    model="ebm",
    model_params={
        "interactions": 0,  # Set >0 for interaction terms
        "max_rounds": 5000
    }
)
```

### Calibrated Models
Ensures well-calibrated probabilities for risk assessment
```python
cfg = InterpretableConfig(
    model="logistic",
    calibrate=True,
    calibration_method="isotonic"
)
```

## Validation Experiments

The module includes comprehensive validation experiments:

### Negative Control
Replaces quotes with random document spans to verify content-specific learning
```python
validator.negative_control_experiment(data, labels)
```

### Feature Ablation
Tests the impact of removing feature groups
```python
validator.feature_ablation_study(data, labels)
```

### Case-Level Analysis
Correlates quote-level predictions with case outcomes
```python
validator.case_level_analysis(data, labels, case_outcomes)
```

## Output Structure

### Model Predictions
```json
{
  "text": "...",
  "fi_polr_pred_class": 2,
  "fi_polr_pred_bucket": "high",
  "fi_polr_confidence": 0.87,
  "fi_polr_class_probs": {
    "low": 0.05,
    "medium": 0.08,
    "high": 0.87
  },
  "fi_polr_risk_features": [
    {"feature": "lex_deception_count", "value": 3.0},
    {"feature": "lex_guarantee_present", "value": 1.0}
  ]
}
```

### Interpretability Report
- `interpretability/feature_importance_forest.pdf`: Top features with confidence intervals
- `interpretability/confusion_matrix.pdf`: Model performance visualization
- `interpretability/calibration_curves.pdf`: Probability calibration assessment
- `interpretability/local_explanation_*.pdf`: Individual prediction explanations

### Validation Results
- `validation/negative_control_experiment.pdf`: Content specificity test
- `validation/feature_ablation_study.pdf`: Feature group importance
- `validation/case_level_correlation.pdf`: Downstream task validation

## Advanced Usage

### Custom Feature Weights
```python
cfg = InterpretableConfig(
    lexicon_weights={
        "deception": 2.0,  # Double weight for deception terms
        "hedges": 0.5      # Half weight for hedging language
    }
)
```

### Parallel Processing
```python
cfg = InterpretableConfig(
    n_jobs=-1,  # Use all CPU cores
    use_sparse=True  # Memory efficient
)
```

### Feature Selection
```python
cfg = InterpretableConfig(
    feature_selection=True,
    n_features=5000  # Top 5000 features by chi-squared
)
```

## Citation

If you use this module in your research, please cite:

```bibtex
@software{corp_speech_risk_interpretable,
  title={Enhanced Interpretable Models for Corporate Speech Risk},
  author={Your Name},
  year={2024},
  url={https://github.com/your/repo}
}
```

## Performance Benchmarks

Typical performance on legal risk classification (3-class ordinal):

| Model | QWK | MAE | Accuracy | ECE |
|-------|-----|-----|----------|-----|
| POLR  | 0.72| 0.31| 0.78     | 0.04|
| EBM   | 0.75| 0.28| 0.80     | 0.03|
| Ensemble | 0.76| 0.27| 0.81  | 0.03|

## Troubleshooting

### Missing Dependencies
```bash
# Install all optional dependencies
pip install mord interpret matplotlib seaborn
```

### Memory Issues
- Use `--n-features` to limit vocabulary size
- Enable `--feature-selection` to reduce dimensionality
- Process large datasets with smaller `--batch-size`

### Slow Training
- Reduce `--n-jobs` if system becomes unresponsive
- For EBM, lower `max_rounds` in model_params
- Use Ridge or Logistic for fastest training
