#!/bin/bash
# Run End-to-End Binary Fusion Model Training
# Trains Legal-BERT, GraphSAGE, CrossModalFusion, and Binary MLP per fold

set -e  # Exit on error

# Configuration
DATA_DIR="data/final_stratified_kfold_splits_binary_quote_balanced"
OUTPUT_DIR="results/binary_fusion_end2end_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10  # Quick training for 1-day deadline
BATCH_SIZE=16  # Reduced for end-to-end memory requirements
LR=1e-4
PATIENCE=5

echo "============================================================"
echo "End-to-End Binary Fusion Model Training"
echo "============================================================"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (reduced for end-to-end)"
echo "Learning Rate: $LR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save run configuration
cat > "$OUTPUT_DIR/run_config.txt" << EOF
End-to-End Binary Fusion Training Configuration
==============================================
Date: $(date)
Data Directory: $DATA_DIR
Output Directory: $OUTPUT_DIR
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Learning Rate: $LR
Patience: $PATIENCE

Architecture:
- Legal-BERT: Trained per fold on training texts
- GraphSAGE: Pretrained per fold on dependency graphs
- CrossModalFusion: Pretrained per fold with contrastive learning
- Binary MLP: Fine-tuned per fold for classification

Key Features:
- Full end-to-end training (no pre-computed embeddings)
- Per-fold model training as requested
- OOF inference using last fold's model
- Optimized for M1 Mac performance

Training Strategy:
1. Pretrain GraphSAGE on graph reconstruction (10 epochs)
2. Pretrain CrossModalFusion with InfoNCE loss (5 epochs)
3. Fine-tune Binary MLP classifier (10 epochs)
EOF

# Run training with all folds
echo "Starting end-to-end training..."
uv run python scripts/train_binary_fusion_end2end.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LR" \
    --patience "$PATIENCE" \
    --pretrain-graphsage \
    --pretrain-fusion \
    --device auto

# Run OOF inference using last fold
echo ""
echo "Running OOF inference..."
uv run python scripts/train_binary_fusion_end2end.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --folds 3 \
    --epochs 1 \
    --run-oof-inference \
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

if [ -f "$OUTPUT_DIR/oof_results.json" ]; then
    echo ""
    echo "OOF Test Results:"
    echo "================="
    uv run python -c "
import json
with open('$OUTPUT_DIR/oof_results.json', 'r') as f:
    oof = json.load(f)
    print(f\"OOF AUC: {oof['auc']:.4f}\")
    print(f\"OOF AP: {oof['avg_precision']:.4f}\")
    print(f\"OOF F1: {oof['f1']:.4f}\")
"
fi
