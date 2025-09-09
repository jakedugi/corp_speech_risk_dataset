#!/bin/bash
# Run End-to-End Binary Fusion Model for a Single Fold
# Useful for debugging or running specific folds

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <fold_number> [output_dir]"
    echo "Example: $0 0"
    echo "Example: $0 2 results/my_test"
    exit 1
fi

FOLD=$1
DATA_DIR="data/final_stratified_kfold_splits_binary_quote_balanced"
OUTPUT_DIR=${2:-"results/binary_fusion_fold${FOLD}_$(date +%Y%m%d_%H%M%S)"}

echo "============================================================"
echo "End-to-End Binary Fusion - Single Fold Training"
echo "============================================================"
echo "Fold: $FOLD"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training for single fold
echo "Training fold $FOLD..."
uv run python scripts/train_binary_fusion_end2end.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --folds $FOLD \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --patience 5 \
    --pretrain-graphsage \
    --pretrain-fusion \
    --device auto

# If this is the last fold (fold 3), optionally run OOF inference
if [ "$FOLD" = "3" ]; then
    echo ""
    echo "Running OOF inference with fold 3 model..."
    uv run python scripts/train_binary_fusion_end2end.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --folds 3 \
        --epochs 1 \
        --run-oof-inference \
        --device auto
fi

echo ""
echo "Training complete! Results saved to: $OUTPUT_DIR"

# Display results
if [ -f "$OUTPUT_DIR/fold_${FOLD}_results.json" ]; then
    echo ""
    echo "Fold $FOLD Results:"
    echo "=================="
    uv run python -c "
import json
with open('$OUTPUT_DIR/fold_${FOLD}_results.json', 'r') as f:
    results = json.load(f)
    final = results['final_metrics']
    print(f\"AUC: {final['auc']:.4f}\")
    print(f\"AP: {final['avg_precision']:.4f}\")
    print(f\"F1: {final['f1']:.4f}\")
    print(f\"Best Epoch: {results['best_epoch'] + 1}\")
"
fi
