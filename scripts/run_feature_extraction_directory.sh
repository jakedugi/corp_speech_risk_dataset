#!/bin/bash
"""
Extract interpretable features from all files in the courtlistener_v6_fused_raw_coral_pred_with_features directory.

This script processes all 3,288 individual JSONL files in the specified directory
and adds comprehensive interpretable features to each record.
"""

set -e  # Exit on any error

# Target directories
INPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_features"
OUTPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_interpretable_features"

# Configuration
TEXT_FIELD="text"
CONTEXT_FIELD="context"
WORKERS=6  # Adjust based on your CPU cores

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Extract interpretable features from all JSONL files in courtlistener directory"
            echo ""
            echo "Options:"
            echo "  --input-dir     Input directory (default: $INPUT_DIR)"
            echo "  --output-dir    Output directory (default: $OUTPUT_DIR)"
            echo "  --workers       Number of parallel workers (default: $WORKERS)"
            echo "  --dry-run       Show what would be processed without processing"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "This processes all ~3,288 individual JSONL files and adds comprehensive"
            echo "interpretable features including lexicons, sequence, linguistic, and structural features."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting batch interpretable feature extraction..."
echo "📁 Input directory: $INPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "👥 Workers: $WORKERS"
echo "📝 Text field: $TEXT_FIELD"
echo "📝 Context field: $CONTEXT_FIELD"

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "❌ Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Count files to process
FILE_COUNT=$(find "$INPUT_DIR" -name "*.jsonl" | wc -l)
echo "📊 Found $FILE_COUNT JSONL files to process"

if [[ "$FILE_COUNT" -eq 0 ]]; then
    echo "❌ Error: No JSONL files found in input directory"
    exit 1
fi

# Show estimated processing time
echo "⏱️ Estimated processing time: $((FILE_COUNT / 60)) - $((FILE_COUNT / 30)) minutes (depending on file sizes)"

# Run batch feature extraction
echo "🔧 Starting parallel feature extraction..."
python scripts/batch_add_features_to_directory.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --text-field "$TEXT_FIELD" \
    --context-field "$CONTEXT_FIELD" \
    --workers "$WORKERS" \
    --include-lexicons \
    --include-sequence \
    --include-linguistic \
    --include-structural \
    ${DRY_RUN}

# Check if successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Batch feature extraction completed successfully!"
    echo "📂 Enhanced files saved to: $OUTPUT_DIR"

    # Show basic stats if not dry run
    if [[ -z "$DRY_RUN" ]]; then
        echo ""
        echo "📊 Results summary:"
        OUTPUT_FILE_COUNT=$(find "$OUTPUT_DIR" -name "*.jsonl" | wc -l)
        echo "Input files: $FILE_COUNT"
        echo "Output files: $OUTPUT_FILE_COUNT"

        if [[ "$OUTPUT_FILE_COUNT" -eq "$FILE_COUNT" ]]; then
            echo "✅ All files processed successfully!"
        else
            echo "⚠️ Some files may have failed processing"
        fi

        # Check sample file for features if jq is available
        if command -v jq &> /dev/null; then
            SAMPLE_FILE=$(find "$OUTPUT_DIR" -name "*.jsonl" | head -1)
            if [[ -n "$SAMPLE_FILE" ]]; then
                echo ""
                echo "📋 Feature analysis from sample file:"
                FEATURE_COUNT=$(head -1 "$SAMPLE_FILE" | jq 'keys | map(select(startswith("feat_"))) | length')
                echo "Features added per record: $FEATURE_COUNT"

                echo "Feature categories:"
                head -1 "$SAMPLE_FILE" | jq -r 'keys | map(select(startswith("feat_"))) | map(split("_")[1]) | group_by(.) | map("\(.[0]): \(length)") | join(", ")'
            fi
        else
            echo "💡 Install jq for detailed feature analysis: brew install jq"
        fi

        echo ""
        echo "🎯 Ready for binary classification K-fold splitting!"
        echo "💡 Next step: Run binary K-fold split with:"
        echo "   ./scripts/run_binary_kfold_split.sh --input-dir \"$OUTPUT_DIR\""
    fi
else
    echo "❌ Batch feature extraction failed!"
    exit 1
fi
