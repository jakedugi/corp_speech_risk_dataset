# Unified Feature Extraction and Pruning Pipeline

## Overview

The unified feature extraction and pruning pipeline (`extract_and_prune_features_pipeline.py`) is the authoritative tool for preparing interpretable features for the Corporate Speech Risk dataset. It combines all feature extraction, testing, and pruning steps into a single comprehensive pipeline.

## Key Features

1. **Comprehensive Feature Extraction**
   - Base lexicon features (deception, guarantee, hedges, etc.)
   - Linguistic features (certainty, negation, etc.)
   - Sequence features (discourse markers, transitions)
   - Structural features (punctuation, capitalization, etc.)
   - **NEW: Derived features** (ratios and interactions)

2. **Automatic Feature Analysis**
   - Redundancy detection within concept groups
   - Discriminative power analysis (mutual information, Kruskal-Wallis)
   - Multicollinearity checks (VIF)
   - Zero-inflation and missing value analysis

3. **Intelligent Feature Pruning**
   - Removes redundant feature scales (keeps norm + presence)
   - Drops features with weak discriminative power
   - Applies approved feature list filter
   - Generates pruning reasons for transparency

## Derived Features

The pipeline now automatically computes these derived features:

### Ratios (directional, unit-free)
- `ratio_guarantee_vs_hedge`: (guarantee_norm + ε) / (hedges_norm + ε)
- `ratio_deception_vs_hedge`: (deception_norm + ε) / (hedges_norm + ε)
- `ratio_guarantee_vs_superlative`: (guarantee_norm + ε) / (superlatives_present + ε)

### Interactions (hypothesized nonlinearities)
- `interact_guarantee_x_cert`: guarantee_norm × high_certainty
- `interact_superlative_x_cert`: superlatives_present × high_certainty
- `interact_hedge_x_guarantee`: hedges_norm × guarantee_norm

## Usage

### Basic Feature Extraction Only
```bash
python scripts/extract_and_prune_features_pipeline.py \
    --input "data/raw/*.jsonl" \
    --output-dir data/enhanced \
    --text-field text \
    --context-field context \
    --batch-size 1000
```

### Full Pipeline with Analysis and Pruning
```bash
python scripts/extract_and_prune_features_pipeline.py \
    --input "data/raw/*.jsonl" \
    --output-dir data/enhanced \
    --analysis-output-dir docs/feature_analysis \
    --text-field text \
    --context-field context \
    --batch-size 1000 \
    --sample-size 50000 \
    --run-pruning \
    --mi-threshold 0.005 \
    --p-threshold 0.1
```

### Using the Shell Script
```bash
bash scripts/run_unified_feature_pipeline.sh
```

## Output Structure

### Enhanced Data Files
```
data/enhanced/
├── doc_1000001_text_stage15.jsonl
├── doc_1000002_text_stage15.jsonl
└── ...
```

Each record includes:
- Original fields (text, context, case_id, etc.)
- All interpretable features with prefix (e.g., `interpretable_lex_deception_norm`)
- Metadata fields (feature_count, text_length, context_length)

### Analysis Results (when --run-pruning is used)
```
docs/feature_analysis/
├── final_feature_set/
│   ├── final_kept_features.txt      # List of features to use for training
│   ├── pruned_features.csv          # Dropped features with reasons
│   ├── final_feature_card.csv       # Statistics for kept features
│   └── discriminative_power_analysis.csv  # Full analysis results
```

## Final Approved Features (16 total)

### Lexicon Features (8)
- `interpretable_lex_deception_norm` - Normalized deception terms
- `interpretable_lex_deception_present` - Binary deception presence
- `interpretable_lex_guarantee_norm` - Normalized guarantee terms
- `interpretable_lex_guarantee_present` - Binary guarantee presence
- `interpretable_lex_hedges_norm` - Normalized hedging terms
- `interpretable_lex_hedges_present` - Binary hedging presence
- `interpretable_lex_pricing_claims_present` - Binary pricing claims
- `interpretable_lex_superlatives_present` - Binary superlatives

### Linguistic Features (1)
- `interpretable_ling_high_certainty` - High certainty modal count

### Sequence Features (1)
- `interpretable_seq_discourse_additive` - Additive discourse markers

### Derived Features (6)
- `interpretable_ratio_guarantee_vs_hedge` - Guarantee/hedge ratio
- `interpretable_ratio_deception_vs_hedge` - Deception/hedge ratio
- `interpretable_ratio_guarantee_vs_superlative` - Guarantee/superlative ratio
- `interpretable_interact_guarantee_x_cert` - Guarantee × certainty
- `interpretable_interact_superlative_x_cert` - Superlative × certainty
- `interpretable_interact_hedge_x_guarantee` - Hedge × guarantee

## Integration with Training Pipeline

The POLAR/MLR training pipeline (`polar_pipeline.py`) now expects these features to already be present in the data. Feature engineering has been removed from the training step to ensure consistency.

### Feature Transformations Applied During Training
- **asinh_rate**: For normalized rates (deception_norm, guarantee_norm, hedges_norm)
- **log1p_robust**: For counts and derived features (interactions, ratios)
- **none**: For binary presence features

## Next Steps

After running the unified pipeline:

1. **Balanced Case Split**:
   ```bash
   python scripts/balanced_case_split.py \
       --input "data/enhanced/doc_*_text_stage15.jsonl" \
       --output-dir data/balanced_case_splits
   ```

2. **Stratified K-Fold Split**:
   ```bash
   python scripts/stratified_kfold_case_split.py \
       --input "data/enhanced/doc_*_text_stage15.jsonl" \
       --output-dir data/final_stratified_kfold_splits
   ```

3. **Train Models**:
   ```bash
   python scripts/run_polr_comprehensive.py \
       --kfold-dir data/final_stratified_kfold_splits \
       --run-name final_model
   ```

## Performance Considerations

- **Batch Processing**: Uses batches to handle large files efficiently
- **Memory Usage**: Sample size for analysis can be adjusted with `--sample-size`
- **Parallel Processing**: File processing is sequential but can be parallelized externally
- **Progress Tracking**: Uses tqdm for visual progress indicators

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `--batch-size` or `--sample-size`
2. **Missing Features**: Ensure text/context fields are correctly specified
3. **Pruning Too Aggressive**: Adjust `--mi-threshold` and `--p-threshold`

### Validation

To verify features were extracted correctly:
```python
import pandas as pd

# Load a sample
df = pd.read_json("data/enhanced/doc_1000001_text_stage15.jsonl", lines=True)

# Check feature columns
feature_cols = [c for c in df.columns if c.startswith("interpretable_")]
print(f"Found {len(feature_cols)} features")

# Check for derived features
derived = [c for c in feature_cols if "ratio" in c or "interact" in c]
print(f"Derived features: {derived}")
```

## Citation

If using this pipeline, please cite:
```
Corporate Speech Risk Dataset Pipeline
Author: Jake Dugan
Institution: University of Edinburgh
```
