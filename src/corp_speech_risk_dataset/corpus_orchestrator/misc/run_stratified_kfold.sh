#!/bin/bash
"""
Run stratified k-fold cross-validation with case-level splits.

This script creates k-fold cross-validation splits that maintain:
- Case-level integrity (no data leakage)
- Stratified label distribution
- Balanced support across folds

The resulting folds can be used directly for model training and evaluation.
"""

# Configuration
INPUT_DATA="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/balanced_case_splits_with_features/train.jsonl"
OUTPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/kfold_splits"
K_FOLDS=5
TARGET_FIELD="coral_pred_class"
STRATIFY_TYPE="classification"
RANDOM_SEED=42

echo "============================================"
echo "Stratified K-Fold Cross-Validation Pipeline"
echo "============================================"
echo "Input data: $INPUT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "K-folds: $K_FOLDS"
echo "Target field: $TARGET_FIELD"
echo "Stratify type: $STRATIFY_TYPE"
echo "Random seed: $RANDOM_SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run stratified k-fold splitting
echo "Creating stratified k-fold splits..."
uv run python scripts/stratified_kfold_case_split.py \
  --input "$INPUT_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --k-folds $K_FOLDS \
  --target-field "$TARGET_FIELD" \
  --stratify-type "$STRATIFY_TYPE" \
  --case-id-field "case_id" \
  --random-seed $RANDOM_SEED

echo ""
echo "K-fold splits completed!"
echo ""

# Show output summary
echo "Generated fold structure:"
ls -la "$OUTPUT_DIR/"

echo ""
echo "Sample fold directory contents:"
if [ -d "$OUTPUT_DIR/fold_0" ]; then
    ls -la "$OUTPUT_DIR/fold_0/"
fi

echo ""
echo "============================================"
echo "Ready for model training and evaluation!"
echo "============================================"
echo ""
echo "Usage examples:"
echo ""
echo "# Train model on fold 0:"
echo "# Train on: $OUTPUT_DIR/fold_0/train.jsonl"
echo "# Validate on: $OUTPUT_DIR/fold_0/val.jsonl"
echo ""
echo "# Access case IDs:"
echo "# cat $OUTPUT_DIR/fold_0/case_ids.json"
echo ""
echo "# View fold statistics:"
echo "# cat $OUTPUT_DIR/fold_statistics.json"
