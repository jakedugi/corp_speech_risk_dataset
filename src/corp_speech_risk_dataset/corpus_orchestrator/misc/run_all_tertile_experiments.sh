#!/bin/bash
#
# Run all tertile experiments on the complete dataset
# This script runs feature validation and all model types
#

set -e  # Exit on error

# Base directories
DATA_DIR="data/final_stratified_kfold_splits_authoritative_complete"
RESULTS_BASE="results/tertile_comprehensive_$(date +%Y%m%d_%H%M%S)"

echo "=== TERTILE COMPREHENSIVE EXPERIMENTS ==="
echo "Data directory: $DATA_DIR"
echo "Results base: $RESULTS_BASE"
echo ""

# Step 1: Run feature validation
echo "Step 1: Running tertile feature validation..."
echo "----------------------------------------"
uv run python scripts/unified_tertile_feature_pipeline.py \
    --data-dir "$DATA_DIR" \
    --output-dir "${RESULTS_BASE}/feature_validation" \
    --fold 3 \
    --sample-size 20000 \
    --iterations 1

echo ""
echo "Feature validation complete!"
echo ""

# Step 2: Run all model types
echo "Step 2: Training all tertile models..."
echo "----------------------------------------"

# Note: POLR and MLR might already be trained, but we'll rerun for consistency
MODEL_TYPES=("polr" "mlr" "elasticnet" "l1" "l2" "svm")

for model in "${MODEL_TYPES[@]}"; do
    echo ""
    echo "Training $model..."
    echo "=================="

    output_dir="${RESULTS_BASE}/models/${model}_full"

    uv run python scripts/run_tertile_comprehensive_models.py \
        --model-type "$model" \
        --data-dir "$DATA_DIR" \
        --output-dir "$output_dir"

    echo "$model training complete!"
done

echo ""
echo "=== ALL TERTILE EXPERIMENTS COMPLETE ==="
echo ""

# Step 3: Generate comparison table
echo "Step 3: Generating results comparison..."
echo "----------------------------------------"

# Create summary comparison
uv run python -c "
import json
import pandas as pd
from pathlib import Path

results_dir = Path('${RESULTS_BASE}')
model_results = {}

# Load results from each model
for model_dir in (results_dir / 'models').glob('*_full'):
    model_name = model_dir.name.replace('_full', '')
    results_file = model_dir / 'training_results.json'

    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            oof = data['oof_metrics']
            model_results[model_name] = {
                'QWK': oof['qwk'],
                'Macro F1': oof['f1_macro'],
                'Accuracy': oof['accuracy'],
                'Per-Class F1': oof['f1_per_class']
            }

# Create comparison table
df = pd.DataFrame(model_results).T
df = df.round(4)
print('\n=== TERTILE MODEL COMPARISON ===\n')
print(df.to_string())

# Save to file
summary_path = results_dir / 'TERTILE_COMPARISON_SUMMARY.md'
with open(summary_path, 'w') as f:
    f.write('# Tertile Model Comparison Summary\n\n')
    f.write('## Performance Metrics (OOF)\n\n')
    f.write(df.to_markdown())
    f.write('\n\n')

    # Add feature validation summary
    val_results = results_dir / 'feature_validation' / 'validation_results_iter_1.json'
    if val_results.exists():
        with open(val_results) as vf:
            val_data = json.load(vf)
            summary = val_data['summary']
            f.write('## Feature Validation Summary\n\n')
            f.write(f\"- Total Features Tested: {summary['total_features']}\n\")
            f.write(f\"- Features Passed: {summary['passed_features']}\n\")
            f.write(f\"- Pass Rate: {summary['passed_features']/summary['total_features']*100:.1f}%\n\")

print(f'\nResults saved to: {summary_path}')
"

echo ""
echo "All results saved to: $RESULTS_BASE"
echo "Done!"
