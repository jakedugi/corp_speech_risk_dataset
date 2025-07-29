# Case Outcome Hyperparameter Optimization

This directory contains scripts for optimizing the hyperparameters of the case outcome imputer using Leave-One-Out Cross-Validation (LOOCV) on the gold standard dataset.

## Files

- `grid_search_optimizer.py`: Core optimization engine that performs LOOCV grid search
- `run_optimization.py`: Wrapper script with proper logging and error handling
- `README_optimization.md`: This documentation file

## Overview

The optimization process:

1. **Loads the gold standard dataset** containing 20 hand-annotated cases with known outcomes
2. **Performs LOOCV**: For each case, trains on 19 cases and tests on 1 case
3. **Tests hyperparameter combinations** across a predefined grid
4. **Optimizes on multiple metrics**:
   - MSE Loss: How close predicted amounts are to actual amounts
   - Precision: Percentage of exact matches among predictions
   - Recall: Percentage of actual amounts that were exactly predicted
   - F1 Score: Harmonic mean of precision and recall

## Hyperparameters Optimized

### Core Extraction Parameters
- `min_amount`: Minimum dollar amount to consider (default: $10,000)
- `context_chars`: Number of characters of context around amounts (default: 100)
- `min_features`: Minimum feature votes required to pass filter (default: 2)
- `header_chars`: Number of characters to consider as header for document titles (default: 2000)

### Position Thresholds
- `case_position_threshold`: Threshold for case position voting (default: 0.5)
- `docket_position_threshold`: Threshold for docket position voting (default: 0.5)

### Case Flag Thresholds
- `fee_shifting_ratio_threshold`: Fee-shifting detection threshold (default: 1.0)
- `patent_ratio_threshold`: Patent reference detection threshold (default: 20.0)
- `dismissal_ratio_threshold`: Dismissal language detection threshold (default: 0.5)
- `bankruptcy_ratio_threshold`: Bankruptcy court detection threshold (default: 0.5)

### Voting Weights
- `proximity_pattern_weight`: Weight for monetary context words (default: 1.0)
- `judgment_verbs_weight`: Weight for legal action verbs (default: 1.0)
- `case_position_weight`: Weight for chronological position within case (default: 1.0)
- `docket_position_weight`: Weight for chronological position within docket (default: 1.0)
- `all_caps_titles_weight`: Weight for ALL CAPS section titles (default: 1.0)
- `document_titles_weight`: Weight for document titles (default: 1.0)

## Usage

### Quick Start (Reduced Grid)
```bash
cd src/corp_speech_risk_dataset/case_outcome
python run_optimization.py
```

### Full Grid Search (Comprehensive)
```bash
python run_optimization.py --full-grid --max-workers 4
```

### Custom Parameters
```bash
python run_optimization.py \
    --gold-standard data/gold_standard/case_outcome_amounts_hand_annotated.csv \
    --extracted-data-root data/extracted \
    --output my_optimization_results.json \
    --max-workers 2 \
    --log-level DEBUG
```

## Output

The optimization produces:

1. **Console output** showing progress and top results
2. **JSON results file** with all hyperparameter combinations and their metrics
3. **Log file** with detailed execution logs

### Example Output
```
🏆 Top 10 Results:
================================================================================

1. MSE Loss: 1.23e+12
   Precision: 0.850
   Recall: 0.800
   F1 Score: 0.824
   Exact Matches: 16/20
   Hyperparameters:
     min_amount: 10000
     context_chars: 200
     min_features: 2
     ...
```

## Gold Standard Dataset

The optimization uses `data/gold_standard/case_outcome_amounts_hand_annotated.csv` which contains:

- `case_id`: Case identifier (e.g., "data/extracted/courtlistener/09-11435_nysb/")
- `final_amount`: Hand-annotated actual outcome amount
- Additional metadata about each case

### Dataset Statistics
- **20 cases** with known outcomes
- **Amount range**: $1,750,000 - $5,000,000,000
- **Case types**: Class actions, patent cases, data breaches, etc.
- **Special cases**: Bankruptcy courts (null outcomes), dismissed cases (zero outcomes)

## LOOCV Process

For each hyperparameter combination:

1. **For each of the 20 cases**:
   - Use the other 19 cases as "training" (though no actual training occurs)
   - Test on the held-out case
   - Record prediction vs actual outcome

2. **Calculate metrics**:
   - MSE Loss across all 20 test cases
   - Precision/Recall for exact matches
   - F1 Score

3. **Rank combinations** by MSE Loss (primary) and F1 Score (secondary)

## Performance Considerations

- **Reduced grid**: ~2,000 combinations (faster testing)
- **Full grid**: ~50,000+ combinations (comprehensive but slow)
- **Parallel processing**: Use `--max-workers` to speed up evaluation
- **Memory usage**: Each combination requires processing all case data

## Interpreting Results

### Best Practices
1. **Start with reduced grid** to get a baseline
2. **Run full grid** for production optimization
3. **Focus on MSE Loss** as primary metric
4. **Consider F1 Score** for balanced precision/recall
5. **Validate top results** manually on a few cases

### Common Patterns
- **Higher `min_amount`**: Often better for avoiding noise
- **Higher `context_chars`**: Better for capturing full context
- **Balanced voting weights**: Usually outperform extreme values
- **Conservative thresholds**: Better for avoiding false positives

## Troubleshooting

### Common Issues
1. **Missing case data**: Ensure extracted data exists for all gold standard cases
2. **Memory errors**: Reduce `--max-workers` or use reduced grid
3. **Long runtime**: Use reduced grid for initial testing
4. **Poor results**: Check gold standard data quality and case coverage

### Debug Mode
```bash
python run_optimization.py --log-level DEBUG
```

This will show detailed information about each prediction and evaluation step.

## Integration with Pipeline

The best hyperparameters from optimization can be applied to the main pipeline:

```python
# Use optimized hyperparameters in case_outcome_imputer.py
voting_weights = VotingWeights(
    proximity_pattern_weight=1.5,  # From optimization
    judgment_verbs_weight=2.0,     # From optimization
    # ... other optimized weights
)
```

## Future Improvements

1. **Bayesian optimization**: More efficient than grid search
2. **Feature importance**: Analyze which hyperparameters matter most
3. **Cross-validation**: Use k-fold instead of LOOCV for larger datasets
4. **Ensemble methods**: Combine multiple hyperparameter sets
5. **Online learning**: Update hyperparameters as new cases arrive
