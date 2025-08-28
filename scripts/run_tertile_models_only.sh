#!/bin/bash
#
# Run just the tertile model training (since feature validation completed)
#

set -e  # Exit on error

# Use existing results directory
RESULTS_BASE="results/tertile_comprehensive_20250826_191647"

echo "=== TERTILE MODEL TRAINING ==="
echo "Results base: $RESULTS_BASE"
echo ""

# Run all model types
MODEL_TYPES=("polr" "mlr" "elasticnet" "l1" "l2" "svm")

for model in "${MODEL_TYPES[@]}"; do
    echo ""
    echo "Training $model..."
    echo "=================="

    output_dir="${RESULTS_BASE}/models/${model}_full"

    uv run python scripts/run_tertile_comprehensive_models.py \
        --model-type "$model" \
        --data-dir "data/final_stratified_kfold_splits_authoritative_complete" \
        --output-dir "$output_dir"

    echo "$model training complete!"
done

echo ""
echo "=== MODEL TRAINING COMPLETE ==="
echo ""

# Generate comparison table
echo "Generating results comparison..."
echo "----------------------------------------"

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
        with open(results_file, 'rb') as f:
            import orjson
            data = orjson.loads(f.read())
            oof = data['oof_metrics']
            model_results[model_name] = {
                'QWK': oof['qwk'],
                'Macro F1': oof['f1_macro'],
                'Accuracy': oof['accuracy'],
                'Per-Class F1': oof['f1_per_class']
            }

# Create comparison table
if model_results:
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
            import orjson
            with open(val_results, 'rb') as vf:
                val_data = orjson.loads(vf.read())
                summary = val_data['summary']
                f.write('## Feature Validation Summary\n\n')
                f.write(f\"- Total Features Tested: {summary['total_features']}\n\")
                f.write(f\"- Features Passed: {summary['passed_features']}\n\")
                f.write(f\"- Pass Rate: {summary['passed_features']/summary['total_features']*100:.1f}%\n\")

    print(f'\nResults saved to: {summary_path}')
else:
    print('No model results found yet.')
"

echo ""
echo "Done!"
