#!/bin/bash
"""
Complete pipeline: Extract features and create binary K-fold splits from directory.

This script runs the full pipeline:
1. Extract interpretable features from all files in the directory
2. Create binary classification K-fold splits
"""

set -e  # Exit on any error

# Default directories
INPUT_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_features"
ENHANCED_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_interpretable_features"
SPLITS_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/final_stratified_kfold_splits_binary_from_directory"

# Configuration
K_FOLDS=4
TARGET_FIELD="final_judgement_real"
OOF_RATIO=0.15
RANDOM_SEED=42
WORKERS=6

# Flags
SKIP_FEATURES=false
SKIP_SPLITS=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --enhanced-dir)
            ENHANCED_DIR="$2"
            shift 2
            ;;
        --splits-dir)
            SPLITS_DIR="$2"
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
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --skip-features)
            SKIP_FEATURES=true
            shift
            ;;
        --skip-splits)
            SKIP_SPLITS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Complete pipeline: extract features and create binary K-fold splits"
            echo ""
            echo "Options:"
            echo "  --input-dir      Input directory (default: $INPUT_DIR)"
            echo "  --enhanced-dir   Enhanced files directory (default: $ENHANCED_DIR)"
            echo "  --splits-dir     K-fold splits directory (default: $SPLITS_DIR)"
            echo "  --k-folds        Number of folds (default: $K_FOLDS)"
            echo "  --target-field   Target field name (default: $TARGET_FIELD)"
            echo "  --workers        Parallel workers (default: $WORKERS)"
            echo "  --skip-features  Skip feature extraction step"
            echo "  --skip-splits    Skip K-fold splitting step"
            echo "  --dry-run        Show what would be done without doing it"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Pipeline steps:"
            echo "1. Extract interpretable features from ~3,288 JSONL files"
            echo "2. Combine all files and create binary classification K-fold splits"
            echo "3. Apply temporal CV, DNT policy, and case-level balancing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting complete pipeline for directory processing..."
echo "📁 Input directory: $INPUT_DIR"
echo "📁 Enhanced directory: $ENHANCED_DIR"
echo "📁 Splits directory: $SPLITS_DIR"
echo "🔢 K-folds: $K_FOLDS"
echo "🎯 Target field: $TARGET_FIELD"
echo "👥 Workers: $WORKERS"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 DRY RUN MODE - showing what would be done"
fi

# Step 1: Feature extraction
if [[ "$SKIP_FEATURES" == "false" ]]; then
    echo ""
    echo "📋 STEP 1: Extracting interpretable features..."
    echo "=========================================="

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would extract features from: $INPUT_DIR"
        echo "Would save enhanced files to: $ENHANCED_DIR"
        echo "Would use $WORKERS parallel workers"

        # Count files
        FILE_COUNT=$(find "$INPUT_DIR" -name "*.jsonl" 2>/dev/null | wc -l || echo "0")
        echo "Files to process: $FILE_COUNT"
    else
        # Check if input directory exists
        if [[ ! -d "$INPUT_DIR" ]]; then
            echo "❌ Error: Input directory does not exist: $INPUT_DIR"
            exit 1
        fi

        # Check if enhanced directory already exists
        if [[ -d "$ENHANCED_DIR" ]]; then
            echo "⚠️ Enhanced directory already exists: $ENHANCED_DIR"
            read -p "Do you want to overwrite it? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping feature extraction..."
                SKIP_FEATURES=true
            else
                echo "Removing existing enhanced directory..."
                rm -rf "$ENHANCED_DIR"
            fi
        fi

        if [[ "$SKIP_FEATURES" == "false" ]]; then
            # Run feature extraction
            python scripts/batch_add_features_to_directory.py \
                --input-dir "$INPUT_DIR" \
                --output-dir "$ENHANCED_DIR" \
                --workers "$WORKERS" \
                --include-lexicons \
                --include-sequence \
                --include-linguistic \
                --include-structural

            if [[ $? -ne 0 ]]; then
                echo "❌ Feature extraction failed!"
                exit 1
            fi

            echo "✅ Feature extraction completed!"
        fi
    fi
else
    echo "📋 STEP 1: Skipping feature extraction (--skip-features)"
fi

# Step 2: Binary K-fold splitting
if [[ "$SKIP_SPLITS" == "false" ]]; then
    echo ""
    echo "📋 STEP 2: Creating binary classification K-fold splits..."
    echo "======================================================"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would combine files from: $ENHANCED_DIR"
        echo "Would create K-fold splits in: $SPLITS_DIR"
        echo "Would use binary classification (lower/higher)"
        echo "Would create $K_FOLDS CV folds + 1 final training fold + OOF test"
    else
        # Check if enhanced directory exists
        if [[ ! -d "$ENHANCED_DIR" ]]; then
            echo "❌ Error: Enhanced directory does not exist: $ENHANCED_DIR"
            echo "💡 Run feature extraction first or use --skip-features if already done"
            exit 1
        fi

        # Check if splits directory already exists
        if [[ -d "$SPLITS_DIR" ]]; then
            echo "⚠️ Splits directory already exists: $SPLITS_DIR"
            read -p "Do you want to overwrite it? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping K-fold splitting..."
                SKIP_SPLITS=true
            else
                echo "Removing existing splits directory..."
                rm -rf "$SPLITS_DIR"
            fi
        fi

        if [[ "$SKIP_SPLITS" == "false" ]]; then
            # Create intermediate combined file path
            INTERMEDIATE_FILE="$SPLITS_DIR/combined_dataset.jsonl"

            # Run binary K-fold splitting
            python scripts/run_binary_kfold_on_directory.py \
                --input-dir "$ENHANCED_DIR" \
                --output-dir "$SPLITS_DIR" \
                --k-folds "$K_FOLDS" \
                --target-field "$TARGET_FIELD" \
                --oof-test-ratio "$OOF_RATIO" \
                --random-seed "$RANDOM_SEED" \
                --intermediate-file "$INTERMEDIATE_FILE"

            if [[ $? -ne 0 ]]; then
                echo "❌ Binary K-fold splitting failed!"
                exit 1
            fi

            echo "✅ Binary K-fold splitting completed!"
        fi
    fi
else
    echo "📋 STEP 2: Skipping K-fold splitting (--skip-splits)"
fi

# Final summary
echo ""
echo "🎉 PIPELINE COMPLETE!"
echo "===================="

if [[ "$DRY_RUN" == "false" ]]; then
    # Show results summary
    if [[ "$SKIP_FEATURES" == "false" && -d "$ENHANCED_DIR" ]]; then
        ENHANCED_COUNT=$(find "$ENHANCED_DIR" -name "*.jsonl" 2>/dev/null | wc -l || echo "0")
        echo "✅ Enhanced files: $ENHANCED_COUNT files in $ENHANCED_DIR"
    fi

    if [[ "$SKIP_SPLITS" == "false" && -d "$SPLITS_DIR" ]]; then
        FOLD_COUNT=$(find "$SPLITS_DIR" -name "fold_*" -type d 2>/dev/null | wc -l || echo "0")
        echo "✅ K-fold splits: $FOLD_COUNT folds in $SPLITS_DIR"

        # Check for key files
        if [[ -f "$SPLITS_DIR/fold_statistics.json" ]]; then
            echo "✅ Metadata files: fold_statistics.json, per_fold_metadata.json, dnt_manifest.json"
        fi

        if [[ -d "$SPLITS_DIR/oof_test" ]]; then
            echo "✅ Out-of-fold test set: available in oof_test/"
        fi
    fi

    echo ""
    echo "🎯 Binary classification ready!"
    echo "📊 Classification type: BINARY (lower/higher outcomes)"
    echo "⚡ Temporal CV with DNT policy applied"
    echo "🔒 Leakage-safe case-level splits"
    echo ""
    echo "📂 Next steps:"
    echo "  • Train models on individual folds"
    echo "  • Use final training fold for production model"
    echo "  • Evaluate on OOF test set for final performance"

else
    echo "🔍 Dry run completed - no files were modified"
fi
