#!/bin/bash
"""
Enhance raw data with interpretable features before splitting.

This script processes the raw JSONL files and adds interpretable features
as new fields, then the enhanced data can be used for balanced case splitting.
"""

# Configuration
RAW_DATA_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred"
OUTPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_features"
BATCH_SIZE=500

echo "============================================"
echo "Raw Data Feature Enhancement Pipeline"
echo "============================================"
echo "Input directory: $RAW_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run feature enhancement
echo "Starting feature enhancement..."
uv run python scripts/enhance_raw_data_with_features.py \
  --input "$RAW_DATA_DIR/doc_*_text_stage15.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --text-field "text" \
  --context-field "context" \
  --feature-prefix "interpretable" \
  --batch-size $BATCH_SIZE

echo ""
echo "Feature enhancement completed!"
echo ""

# Show output summary
echo "Output files created:"
ls -la "$OUTPUT_DIR/"

echo ""
echo "Sample of enhanced data (first record):"
head -1 "$OUTPUT_DIR/doc_"*"_text_stage15.jsonl" | head -1 | python -m json.tool | head -20

echo ""
echo "============================================"
echo "Ready for balanced case splitting!"
echo "============================================"
echo ""
echo "Next step: Run balanced case split with enhanced data:"
echo "python scripts/balanced_case_split.py \\"
echo "  --input \"$OUTPUT_DIR/doc_*_text_stage15.jsonl\" \\"
echo "  --output-dir data/balanced_case_splits_with_features \\"
echo "  --outlier-threshold 5000000000 \\"
echo "  --exclude-speakers \"Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA\" \\"
echo "  --train-ratio 0.7 \\"
echo "  --val-ratio 0.15 \\"
echo "  --test-ratio 0.15 \\"
echo "  --random-seed 42"
