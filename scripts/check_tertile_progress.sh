#!/bin/bash
#
# Quick script to check tertile experiment progress
#

echo "=== TERTILE EXPERIMENT PROGRESS ==="
echo ""

# Find latest results directory
LATEST_DIR=$(ls -td results/tertile_comprehensive_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No tertile experiment results found yet."
    exit 1
fi

echo "Checking: $LATEST_DIR"
echo ""

# Check feature validation
if [ -f "$LATEST_DIR/feature_validation/validation_results_iter_1.json" ]; then
    echo "✅ Feature Validation: COMPLETE"
    PASSED=$(uv run python -c "import orjson; data=orjson.loads(open('$LATEST_DIR/feature_validation/validation_results_iter_1.json', 'rb').read()); print(f\"   Passed: {data['summary']['passed_features']}/{data['summary']['total_features']} features\")")
    echo "$PASSED"
else
    echo "⏳ Feature Validation: In progress..."
fi

echo ""

# Check models
echo "📊 Model Training Status:"
for model in polr mlr elasticnet l1 l2 svm; do
    if [ -f "$LATEST_DIR/models/${model}_full/training_results.json" ]; then
        echo "  ✅ $model: COMPLETE"
        # Extract QWK score
        QWK=$(uv run python -c "import orjson; data=orjson.loads(open('$LATEST_DIR/models/${model}_full/training_results.json', 'rb').read()); print(f\"     QWK: {data['oof_metrics']['qwk']:.4f}\")" 2>/dev/null || echo "     Error reading results")
        echo "$QWK"
    elif [ -d "$LATEST_DIR/models/${model}_full" ]; then
        echo "  ⏳ $model: Training..."
    else
        echo "  ⏸️  $model: Not started"
    fi
done

echo ""

# Check for final summary
if [ -f "$LATEST_DIR/TERTILE_COMPARISON_SUMMARY.md" ]; then
    echo "✅ ALL EXPERIMENTS COMPLETE!"
    echo "   Summary: $LATEST_DIR/TERTILE_COMPARISON_SUMMARY.md"
else
    echo "⏳ Experiments still running..."
fi

echo ""

# Check log file
if [ -f "tertile_experiments.log" ]; then
    echo "📄 Log file: tertile_experiments.log"
    echo "   Last update: $(tail -1 tertile_experiments.log | head -c 50)..."
fi
