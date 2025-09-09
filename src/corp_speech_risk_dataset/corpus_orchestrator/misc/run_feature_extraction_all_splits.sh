#!/bin/bash
"""
Run interpretable feature extraction on all balanced splits.

This script processes all split files (train, val, test, outliers) and creates
enhanced versions with interpretable features appended.
"""

# Configuration
SPLITS_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/balanced_case_splits"
OUTPUT_DIR="$SPLITS_DIR/with_features"
BATCH_SIZE=500

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Interpretable Feature Extraction Pipeline"
echo "============================================"
echo "Input directory: $SPLITS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo ""

# Function to extract features for a split
extract_features() {
    local split_name="$1"
    local input_file="$SPLITS_DIR/${split_name}.jsonl"
    local output_file="$OUTPUT_DIR/${split_name}_with_features.jsonl"

    if [[ ! -f "$input_file" ]]; then
        echo "⚠️  Skipping $split_name: input file not found"
        return 1
    fi

    echo "🔄 Processing $split_name split..."
    echo "   Input: $input_file"
    echo "   Output: $output_file"

    uv run python scripts/extract_interpretable_features.py \
        --input "$input_file" \
        --output "$output_file" \
        --text-field "text" \
        --context-field "context" \
        --feature-prefix "interpretable" \
        --batch-size "$BATCH_SIZE" \
        --log-level INFO

    if [[ $? -eq 0 ]]; then
        echo "✅ Successfully processed $split_name"

        # Show file size comparison
        local input_size=$(du -h "$input_file" | cut -f1)
        local output_size=$(du -h "$output_file" | cut -f1)
        echo "   File size: $input_size → $output_size"

        # Count records
        local record_count=$(wc -l < "$input_file")
        echo "   Records: $record_count"
        echo ""

        return 0
    else
        echo "❌ Failed to process $split_name"
        return 1
    fi
}

# Process each split
SPLITS=("train" "val" "test" "outliers")
SUCCESS_COUNT=0
TOTAL_COUNT=${#SPLITS[@]}

for split in "${SPLITS[@]}"; do
    if extract_features "$split"; then
        ((SUCCESS_COUNT++))
    fi
done

echo "============================================"
echo "Feature Extraction Summary"
echo "============================================"
echo "Successfully processed: $SUCCESS_COUNT/$TOTAL_COUNT splits"

if [[ $SUCCESS_COUNT -eq $TOTAL_COUNT ]]; then
    echo "🎉 All splits processed successfully!"
    echo ""
    echo "Enhanced files with interpretable features:"
    ls -lh "$OUTPUT_DIR"/*.jsonl
else
    echo "⚠️  Some splits failed to process"
    exit 1
fi
