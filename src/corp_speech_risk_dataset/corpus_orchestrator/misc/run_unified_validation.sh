#!/bin/bash
# Unified Feature Validation Pipeline - Optimized for Mac M1
# Combines all validation logic into one efficient pipeline

set -e

echo "🚀 Starting Unified Feature Validation Pipeline"
echo "==============================================="

# Set optimal environment for Mac M1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Run the unified validation pipeline
uv run python scripts/unified_feature_validation_pipeline.py \
    --fold-dir data/final_stratified_kfold_splits_authoritative \
    --fold 3 \
    --sample-size 8000 \
    --output-dir docs/unified_validation_results \
    --auto-update-governance

echo "✅ Unified validation pipeline complete!"
echo "📁 Results available in: docs/unified_validation_results/"
echo "📊 Progress reports: docs/unified_validation_results/progress_*.json"
echo "📋 Full results: docs/unified_validation_results/unified_validation_results.json"
