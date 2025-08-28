#!/bin/bash
"""
Extract interpretable features from the raw dataset and create enhanced version.

This script applies all the new interpretable features from the features.py module
to the raw data and saves an enhanced dataset ready for binary classification.
"""

set -e  # Exit on any error

# Default paths
INPUT_FILE="data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl"
OUTPUT_FILE="data/enhanced_combined/final_clean_dataset_with_interpretable_features.jsonl"
BATCH_SIZE=5000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--input INPUT_FILE] [--output OUTPUT_FILE] [--batch-size BATCH_SIZE]"
            echo ""
            echo "Extract interpretable features from raw dataset"
            echo ""
            echo "Options:"
            echo "  --input         Input JSONL file (default: $INPUT_FILE)"
            echo "  --output        Output JSONL file (default: $OUTPUT_FILE)"
            echo "  --batch-size    Batch size for processing (default: $BATCH_SIZE)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting interpretable feature extraction..."
echo "📁 Input: $INPUT_FILE"
echo "📁 Output: $OUTPUT_FILE"
echo "📦 Batch size: $BATCH_SIZE"

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Run feature extraction
echo "🔧 Extracting features..."
python scripts/extract_interpretable_features.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --batch-size "$BATCH_SIZE" \
    --include-lexicons \
    --include-sequence \
    --include-linguistic \
    --include-structural

# Check if successful
if [[ $? -eq 0 ]]; then
    echo "✅ Feature extraction completed successfully!"
    echo "📄 Enhanced dataset saved to: $OUTPUT_FILE"

    # Show basic stats
    if command -v jq &> /dev/null; then
        echo ""
        echo "📊 Quick stats:"
        echo "Total records: $(wc -l < "$OUTPUT_FILE")"
        echo "Feature count: $(head -1 "$OUTPUT_FILE" | jq 'keys | map(select(startswith("feat_"))) | length')"
        echo "Sample feature categories:"
        head -1 "$OUTPUT_FILE" | jq -r 'keys | map(select(startswith("feat_"))) | map(split("_")[1]) | unique | sort | join(", ")'
    else
        echo "📊 Install jq for detailed stats: brew install jq"
    fi
else
    echo "❌ Feature extraction failed!"
    exit 1
fi
