#!/bin/bash

# Run Binary Feature Development Pipeline
# This script runs the unified binary feature development pipeline with sensible defaults

set -e  # Exit on error

# Configuration
DATA_DIR="${1:-data/final_stratified_kfold_splits_binary_quote_balanced}"
OUTPUT_DIR="${2:-results/binary_feature_development_$(date +%Y%m%d_%H%M%S)}"
FOLD="${3:-4}"
SAMPLE_SIZE="${4:-50000}"
ITERATIONS="${5:-3}"

echo "========================================="
echo "Binary Feature Development Pipeline"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Fold: $FOLD"
echo "Sample size: $SAMPLE_SIZE"
echo "Iterations: $ITERATIONS"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the pipeline
echo "Starting pipeline..."
uv run python scripts/unified_binary_feature_pipeline.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --fold "$FOLD" \
    --sample-size "$SAMPLE_SIZE" \
    --iterations "$ITERATIONS" \
    --auto-update-governance 2>&1 | tee "$OUTPUT_DIR/pipeline.log"

# Generate summary
echo ""
echo "========================================="
echo "Pipeline completed!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "- Final report: $OUTPUT_DIR/FINAL_REPORT.md"
echo "- Figures: $OUTPUT_DIR/figures/"
echo "- Reports: $OUTPUT_DIR/reports/"
echo "- Log: $OUTPUT_DIR/pipeline.log"
echo ""

# Display final feature count
if [ -f "$OUTPUT_DIR/reports/final_feature_importance.csv" ]; then
    FEATURE_COUNT=$(tail -n +2 "$OUTPUT_DIR/reports/final_feature_importance.csv" | wc -l)
    echo "Final approved features: $FEATURE_COUNT"
fi

echo "========================================="
