#!/bin/bash

# Improved Stratified K-Fold Cross-Validation with Label Balancing
# This script creates k-fold splits with proper stratification and missing label handling

set -e

# Configuration
INPUT_DATA="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/enhanced_combined/final_clean_dataset.jsonl"
OUTPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_stratified_kfold_splits"
K_FOLDS=5
TARGET_FIELD="final_judgement_real"
STRATIFY_TYPE="regression"
N_BINS=3
RANDOM_SEED=42

echo "============================================"
echo "Improved Stratified K-Fold Cross-Validation"
echo "============================================"
echo ""
echo "Key improvements:"
echo "- Uses StratifiedGroupKFold for proper label balancing"
echo "- Option to drop missing labels for stable training"
echo "- Computes class weights for balanced training"
echo "- Better validation and error handling"
echo ""
echo "Input data: $INPUT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "K-folds: $K_FOLDS"
echo "Target field: $TARGET_FIELD"
echo "Stratify type: $STRATIFY_TYPE"
echo "Number of buckets: $N_BINS (33% each)"
echo "Drop missing labels: true"
echo "Compute class weights: true"
echo "Random seed: $RANDOM_SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run improved stratified k-fold splitting
echo "Creating improved stratified k-fold splits..."
uv run python scripts/stratified_kfold_case_split.py \
  --input "$INPUT_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --k-folds $K_FOLDS \
  --target-field "$TARGET_FIELD" \
  --stratify-type "$STRATIFY_TYPE" \
  --case-id-field "_src" \
  --n-bins $N_BINS \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --compute-weights \
  --random-seed $RANDOM_SEED

echo ""
echo "Improved k-fold splits completed!"
echo ""

# Show generated structure
echo "Generated fold structure:"
ls -la "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "Key Improvements Made"
echo "============================================"
echo ""
echo "1. LABEL BALANCING:"
echo "   - Uses StratifiedGroupKFold to ensure balanced label distribution"
echo "   - Each fold has similar proportions of low/medium/high outcome bins"
echo "   - Maintains case-level integrity (no data leakage)"
echo ""
echo "2. MISSING LABEL HANDLING:"
echo "   - Drops cases with 'missing' labels from training"
echo "   - Focuses model on actual risk bins with monetary outcomes"
echo "   - Prevents model from learning to predict 'missing' class"
echo ""
echo "3. CLASS WEIGHTS:"
echo "   - Computed balanced class weights for training"
echo "   - Saved to: $OUTPUT_DIR/class_weights.json"
echo "   - Use these weights in your model loss function"
echo ""
echo "4. VALIDATION:"
echo "   - Validates minimum support per label across folds"
echo "   - Provides detailed statistics and warnings"
echo "   - Handles edge cases gracefully"
echo ""
echo "============================================"
echo "Usage for Model Training"
echo "============================================"
echo ""
echo "This creates balanced folds you use BEFORE training. Typical workflow:"
echo ""
echo "1. Run this script once to create folds"
echo "2. For each fold, train your model using the class weights:"
echo ""
echo "   for fold in {0..4}; do"
echo "     echo \"Training on fold \$fold...\""
echo "     # Use class weights from: $OUTPUT_DIR/class_weights.json"
echo "     # Training data: $OUTPUT_DIR/fold_\$fold/train.jsonl"
echo "     # Validation data: $OUTPUT_DIR/fold_\$fold/val.jsonl"
echo "     # Test data: $OUTPUT_DIR/fold_\$fold/test.jsonl"
echo "   done"
echo ""
echo "3. Average results across all folds for final performance"
echo ""
