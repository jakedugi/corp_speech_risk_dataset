#!/bin/bash
"""
Create binary classification stratified K-fold splits.

This script creates leakage-safe temporal K-fold cross-validation splits using
binary classification (lower/higher) instead of tertiles, with all the same
temporal CV, DNT policy, and weighting logic as the original.
"""

set -e  # Exit on any error

# Default parameters
INPUT_FILE="data/enhanced_combined/final_clean_dataset_with_interpretable_features.jsonl"
OUTPUT_DIR="data/final_stratified_kfold_splits_binary"
K_FOLDS=4
TARGET_FIELD="final_judgement_real"
OOF_RATIO=0.15
RANDOM_SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --k-folds)
            K_FOLDS="$2"
            shift 2
            ;;
        --target-field)
            TARGET_FIELD="$2"
            shift 2
            ;;
        --oof-ratio)
            OOF_RATIO="$2"
            shift 2
            ;;
        --seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Create binary classification stratified K-fold splits"
            echo ""
            echo "Options:"
            echo "  --input         Input JSONL file (default: $INPUT_FILE)"
            echo "  --output-dir    Output directory (default: $OUTPUT_DIR)"
            echo "  --k-folds       Number of folds (default: $K_FOLDS)"
            echo "  --target-field  Target field name (default: $TARGET_FIELD)"
            echo "  --oof-ratio     Out-of-fold test ratio (default: $OOF_RATIO)"
            echo "  --seed          Random seed (default: $RANDOM_SEED)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "This creates binary classification (lower/higher) instead of tertiles,"
            echo "using a single interpolated midpoint to divide each fold into case-level halves."
            echo "All temporal CV, DNT policy, and weighting logic remains exactly the same."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting binary classification K-fold splitting..."
echo "📁 Input: $INPUT_FILE"
echo "📁 Output: $OUTPUT_DIR"
echo "🔢 K-folds: $K_FOLDS"
echo "🎯 Target field: $TARGET_FIELD"
echo "📊 OOF ratio: $OOF_RATIO"
echo "🎲 Random seed: $RANDOM_SEED"

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Error: Input file does not exist: $INPUT_FILE"
    echo "💡 Tip: Run feature extraction first with run_feature_extraction.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run binary k-fold splitting
echo "🔧 Creating binary classification splits..."
python scripts/stratified_kfold_binary_split.py \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --k-folds "$K_FOLDS" \
    --target-field "$TARGET_FIELD" \
    --use-temporal-cv \
    --oof-test-ratio "$OOF_RATIO" \
    --oof-min-ratio 0.15 \
    --oof-max-ratio 0.40 \
    --oof-step 0.05 \
    --oof-min-class-cases 5 \
    --oof-min-class-quotes 50 \
    --oof-class-criterion both \
    --random-seed "$RANDOM_SEED"

# Check if successful
if [[ $? -eq 0 ]]; then
    echo "✅ Binary K-fold splitting completed successfully!"
    echo "📂 Results saved to: $OUTPUT_DIR"

    # Show structure
    echo ""
    echo "📊 Split structure:"
    echo "Folds created:"
    for fold_dir in "$OUTPUT_DIR"/fold_*; do
        if [[ -d "$fold_dir" ]]; then
            fold_name=$(basename "$fold_dir")
            echo "  $fold_name:"

            # Count records in each split
            if [[ -f "$fold_dir/train.jsonl" ]]; then
                train_count=$(wc -l < "$fold_dir/train.jsonl" 2>/dev/null || echo "0")
                echo "    train: $train_count records"
            fi

            if [[ -f "$fold_dir/val.jsonl" ]]; then
                val_count=$(wc -l < "$fold_dir/val.jsonl" 2>/dev/null || echo "0")
                echo "    val: $val_count records"
            fi

            if [[ -f "$fold_dir/test.jsonl" ]]; then
                test_count=$(wc -l < "$fold_dir/test.jsonl" 2>/dev/null || echo "0")
                echo "    test: $test_count records"
            fi

            if [[ -f "$fold_dir/dev.jsonl" ]]; then
                dev_count=$(wc -l < "$fold_dir/dev.jsonl" 2>/dev/null || echo "0")
                echo "    dev: $dev_count records (final training fold)"
            fi
        fi
    done

    # Check for OOF test
    if [[ -d "$OUTPUT_DIR/oof_test" ]]; then
        oof_count=$(wc -l < "$OUTPUT_DIR/oof_test/test.jsonl" 2>/dev/null || echo "0")
        echo "  oof_test: $oof_count records"
    fi

    echo ""
    echo "📋 Key files created:"
    echo "  • fold_statistics.json - Overall statistics"
    echo "  • per_fold_metadata.json - Per-fold edges and weights"
    echo "  • dnt_manifest.json - Do-not-train columns"
    echo ""
    echo "🎯 Classification type: BINARY (lower/higher)"
    echo "⚡ Ready for binary classification model training!"

else
    echo "❌ Binary K-fold splitting failed!"
    exit 1
fi
