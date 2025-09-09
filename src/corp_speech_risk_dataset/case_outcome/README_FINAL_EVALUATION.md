# Final Evaluation - Case Outcome Imputation

This directory contains the final evaluation script for the case outcome imputation pipeline using hyperparameters optimized through Bayesian optimization.

## Quick Start

### Option 1: Using the Python script directly
```bash
# From project root
uv run python src/corp_speech_risk_dataset/case_outcome/final_evaluate.py \
    --annotations data/gold_standard/case_outcome_amounts_hand_annotated.csv \
    --extracted-root data/extracted
```

### Option 2: Using the convenience shell script
```bash
# From project root
./scripts/run_final_evaluation.sh
```

### Option 3: With custom environment variables
```bash
export ANNOTATIONS=data/gold_standard/case_outcome_amounts_hand_annotated.csv
export EXTRACTED_ROOT=data/extracted
./scripts/run_final_evaluation.sh
```

## Tuned Hyperparameters

The evaluation uses these exact parameters found through fine-tuning:

- **min_amount**: 29309.97970771781
- **context_chars**: 561
- **min_features**: 15
- **case_pos**: 0.5423630428751168
- **docket_pos**: 0.7947200838693315

## Disabled Logic

To focus purely on amount extraction, the following logic is disabled by setting extreme thresholds:

- **dismissal_ratio_threshold**: 200 (vs normal ~0.5)
- **strict_dismissal_threshold**: 200 (vs normal ~0.8)
- **dismissal_document_type_weight**: 0 (vs normal 2.0-3.0)
- **strict_dismissal_document_type_weight**: 0 (vs normal 2.0-3.0)
- **bankruptcy_ratio_threshold**: 6e22 (vs normal ~50)
- **patent_ratio_threshold**: 6e22 (vs normal ~50)

## Output Metrics

The evaluation reports:

### Per-Case Metrics
- True amount vs predicted amount
- Prediction error and squared error
- Processing status (SUCCESS, DISMISSED, MISSING_DATA, etc.)
- Candidate counts (raw vs filtered)
- Coverage (whether true amount appears in candidates)

### Overall Metrics
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Root Mean Squared Error (RMSE)**: Square root of mean squared error
- **Precision/Recall/F1**: Binary classification performance (award vs zero)
- **Exact Accuracy**: Percentage of cases with perfect amount matches
- **Coverage**: Percentage where true amount appears in candidates

### Status Breakdown
Count and percentage of cases by processing status:
- SUCCESS: Normal processing completed
- DISMISSED: Case flagged as dismissed (should be rare with disabled logic)
- MISSING_DATA: Case directory not found
- NO_STAGE1_FILES: No extracted text files found
- PATENT: Patent case flagged (should be rare with disabled logic)

## Implementation Details

The script:

1. Loads hand-annotated ground truth from CSV
2. Processes each case using the `evaluate_case()` function from the test harness
3. Collects per-case results and computes aggregate metrics
4. Provides detailed output for analysis

The evaluation uses the same core logic as the hyperparameter tuning but with fixed, optimal parameters instead of searching the parameter space.

## Files

- `final_evaluate.py`: Main evaluation script
- `test_case_outcome_imputer.py`: Test harness with `evaluate_case()` function
- `case_outcome_imputer.py`: Core imputation logic
- `extract_cash_amounts_stage1.py`: Amount extraction and voting logic
- `../../scripts/run_final_evaluation.sh`: Convenience runner script

## Dependencies

The script requires the same dependencies as the test harness:
- pandas, numpy
- scikit-learn (for metrics)
- The case outcome imputation modules

All dependencies are managed via `uv` and specified in `pyproject.toml`.
