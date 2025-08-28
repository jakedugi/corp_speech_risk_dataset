#!/bin/bash
# Run Binary Fusion Model with Cross-Validation
# Quick execution script for binary classification with cross-modal fusion

set -e  # Exit on error

# Configuration
DATA_DIR="data/final_stratified_kfold_splits_binary_quote_balanced"
OUTPUT_DIR="results/binary_fusion_cv_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10  # Quick training for 1-day deadline
BATCH_SIZE=32  # M1 Mac optimized
LR=1e-4
PATIENCE=5

echo "=================================================="
echo "Binary Fusion Model - Cross-Validation Run"
echo "=================================================="
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save run configuration
cat > "$OUTPUT_DIR/run_config.txt" << EOF
Binary Fusion CV Run Configuration
==================================
Date: $(date)
Data Directory: $DATA_DIR
Output Directory: $OUTPUT_DIR
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Learning Rate: $LR
Patience: $PATIENCE

Architecture:
- Cross-Modal Fusion: Legal-BERT (768D) + GraphSAGE (256D)
- Backbone: 3-layer residual MLP (768 → 512 → 256)
- Head: Binary MLP (256 → 128 → 1)
- Loss: Binary Cross-Entropy with class weights

Notes:
- Pure MLP head instead of CORAL for binary task
- Retains cross-modal fusion novelty
- Optimized for M1 Mac performance
EOF

# Run training
echo "Starting cross-validation training..."
uv run python scripts/train_binary_fusion_model.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LR" \
    --patience "$PATIENCE" \
    --device auto

echo ""
echo "Training complete! Results saved to: $OUTPUT_DIR"
echo ""

# Display summary if available
if [ -f "$OUTPUT_DIR/cv_summary.json" ]; then
    echo "Cross-Validation Summary:"
    echo "========================="
    uv run python -c "
import json
with open('$OUTPUT_DIR/cv_summary.json', 'r') as f:
    summary = json.load(f)
    print(f\"Mean AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}\")
    print(f\"Mean AP: {summary['mean_avg_precision']:.4f} ± {summary['std_avg_precision']:.4f}\")
    print(f\"Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}\")
"
fi
