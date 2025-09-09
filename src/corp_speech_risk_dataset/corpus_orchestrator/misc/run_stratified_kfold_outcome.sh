#!/bin/bash
"""
Run stratified k-fold cross-validation using final_judgement_real with 3 equal buckets.

This creates k-fold splits stratified by outcome amounts divided into 3 equal buckets:
- Bucket 1: Bottom 33% of outcomes (low amounts)
- Bucket 2: Middle 33% of outcomes (medium amounts)
- Bucket 3: Top 33% of outcomes (high amounts)

This ensures each fold has a balanced distribution of low/medium/high outcome cases.
"""

# Configuration
INPUT_DATA="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/balanced_case_splits_with_features/train.jsonl"
OUTPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/kfold_splits_outcome"
K_FOLDS=5
TARGET_FIELD="final_judgement_real"
STRATIFY_TYPE="regression"
N_BINS=3  # 3 equal buckets (33% each)
RANDOM_SEED=42

echo "============================================"
echo "Stratified K-Fold with Outcome Buckets"
echo "============================================"
echo "Input data: $INPUT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "K-folds: $K_FOLDS"
echo "Target field: $TARGET_FIELD"
echo "Stratify type: $STRATIFY_TYPE"
echo "Number of buckets: $N_BINS (33% each)"
echo "Random seed: $RANDOM_SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run stratified k-fold splitting
echo "Creating stratified k-fold splits with outcome buckets..."
uv run python scripts/stratified_kfold_case_split.py \
  --input "$INPUT_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --k-folds $K_FOLDS \
  --target-field "$TARGET_FIELD" \
  --stratify-type "$STRATIFY_TYPE" \
  --case-id-field "case_id" \
  --n-bins $N_BINS \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --random-seed $RANDOM_SEED

echo ""
echo "K-fold splits with outcome buckets completed!"
echo ""

# Show output summary
echo "Generated fold structure:"
ls -la "$OUTPUT_DIR/"

echo ""
echo "============================================"
echo "Usage for Model Training"
echo "============================================"
echo ""
echo "This creates folds you use BEFORE training. Typical workflow:"
echo ""
echo "1. Run this script once to create folds"
echo "2. For each fold, train your model:"
echo ""
echo "   for fold in {0..4}; do"
echo "     echo \"Training on fold \$fold...\""
echo "     # Train model using:"
echo "     #   Training data: $OUTPUT_DIR/fold_\$fold/train.jsonl"
echo "     #   Validation data: $OUTPUT_DIR/fold_\$fold/val.jsonl"
echo "   done"
echo ""
echo "3. Average results across all folds for final performance"
