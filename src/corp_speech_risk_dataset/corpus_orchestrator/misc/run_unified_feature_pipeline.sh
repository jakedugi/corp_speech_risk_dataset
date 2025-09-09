#!/bin/bash
"""
Unified Feature Extraction and Pruning Pipeline

This script runs the comprehensive feature extraction and pruning pipeline that:
1. Extracts all interpretable features (base + derived)
2. Tests features for discriminative power
3. Analyzes redundancy and multicollinearity
4. Prunes features based on established criteria
5. Generates final feature set for training

The enhanced data can then be used for balanced case splitting.
"""

# Configuration
RAW_DATA_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred"
OUTPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_features"
ANALYSIS_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/docs/feature_analysis"
BATCH_SIZE=500
SAMPLE_SIZE=50000

echo "============================================================"
echo "UNIFIED FEATURE EXTRACTION AND PRUNING PIPELINE"
echo "============================================================"
echo "Input directory: $RAW_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Analysis directory: $ANALYSIS_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Sample size for analysis: $SAMPLE_SIZE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ANALYSIS_DIR"

# Run unified pipeline with pruning
echo "Starting unified feature extraction and pruning pipeline..."
uv run python scripts/extract_and_prune_features_pipeline.py \
  --input "$RAW_DATA_DIR/doc_*_text_stage15.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --analysis-output-dir "$ANALYSIS_DIR" \
  --text-field "text" \
  --context-field "context" \
  --feature-prefix "interpretable" \
  --batch-size $BATCH_SIZE \
  --sample-size $SAMPLE_SIZE \
  --run-pruning \
  --mi-threshold 0.005 \
  --p-threshold 0.1

echo ""
echo "Pipeline completed!"
echo ""

# Show output summary
echo "============================================================"
echo "OUTPUT SUMMARY"
echo "============================================================"
echo ""
echo "Enhanced files created:"
ls -la "$OUTPUT_DIR/" | head -10

echo ""
echo "Analysis results:"
ls -la "$ANALYSIS_DIR/final_feature_set/"

echo ""
echo "Final feature count:"
if [ -f "$ANALYSIS_DIR/final_feature_set/final_kept_features.txt" ]; then
    FEATURE_COUNT=$(grep -v "^#" "$ANALYSIS_DIR/final_feature_set/final_kept_features.txt" | wc -l)
    echo "  Kept features: $FEATURE_COUNT"
    echo ""
    echo "Feature list:"
    grep -v "^#" "$ANALYSIS_DIR/final_feature_set/final_kept_features.txt" | head -20
fi

echo ""
echo "Sample of enhanced data (first record):"
if ls "$OUTPUT_DIR"/doc_*_text_stage15.jsonl 1> /dev/null 2>&1; then
    head -1 "$OUTPUT_DIR"/doc_*_text_stage15.jsonl | python -m json.tool | head -30
fi

echo ""
echo "============================================================"
echo "READY FOR BALANCED CASE SPLITTING!"
echo "============================================================"
echo ""
echo "Next step: Run balanced case split with enhanced data:"
echo ""
echo "python scripts/balanced_case_split.py \\"
echo "  --input \"$OUTPUT_DIR/doc_*_text_stage15.jsonl\" \\"
echo "  --output-dir data/balanced_case_splits_with_features \\"
echo "  --outlier-threshold 5000000000 \\"
echo "  --exclude-speakers \"Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA\" \\"
echo "  --train-ratio 0.7 \\"
echo "  --val-ratio 0.15 \\"
echo "  --test-ratio 0.15 \\"
echo "  --random-seed 42"
echo ""
echo "Or use the stratified k-fold split for cross-validation:"
echo ""
echo "python scripts/stratified_kfold_case_split.py \\"
echo "  --input \"$OUTPUT_DIR/doc_*_text_stage15.jsonl\" \\"
echo "  --output-dir data/final_stratified_kfold_splits \\"
echo "  --n-splits 4 \\"
echo "  --oof-test-fraction 0.15 \\"
echo "  --outlier-threshold 5000000000 \\"
echo "  --exclude-speakers \"Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA\" \\"
echo "  --random-seed 42"
echo ""
