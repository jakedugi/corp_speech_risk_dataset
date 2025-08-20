#!/bin/bash
"""
Complete Validation Pipeline - Run Both Systems Together

This script runs both validation approaches and provides unified interpretation:
1. Iterative Feature Development (detailed feature analysis)
2. Unified Validation Pipeline (production readiness)
3. Comparative Analysis (interpret differences)
"""

set -e  # Exit on any error

# Configuration
ITERATION="comprehensive_validation_with_20_new"
FOLD_DIR="data/final_stratified_kfold_splits_authoritative"
FOLD=3
SAMPLE_SIZE=10000

echo "🚀 COMPLETE VALIDATION PIPELINE"
echo "================================"
echo "Iteration: $ITERATION"
echo "Fold: $FOLD"
echo "Sample Size: $SAMPLE_SIZE"
echo ""

# Step 1: Run Iterative Feature Development
echo "📊 STEP 1: Iterative Feature Development Analysis"
echo "------------------------------------------------"
echo "This analyzes discriminative power, Class 0 AUC, size bias, and leakage..."
echo ""

uv run python scripts/iterative_feature_development_kfold.py \
    --iteration "$ITERATION" \
    --fold-dir "$FOLD_DIR" \
    --fold "$FOLD" \
    --sample-size "$SAMPLE_SIZE" \
    --test-class-discrimination

echo ""
echo "✅ Iterative analysis complete!"

# Step 2: Run Unified Validation Pipeline
echo ""
echo "🔧 STEP 2: Unified Validation Pipeline"
echo "--------------------------------------"
echo "This tests production readiness with stricter thresholds..."
echo ""

./run_unified_validation.sh

echo ""
echo "✅ Unified validation complete!"

# Step 3: Generate Feature Performance Report
echo ""
echo "📈 STEP 3: Feature Performance Report"
echo "------------------------------------"
echo "Generating comprehensive performance analysis..."
echo ""

uv run python scripts/generate_feature_performance_report.py \
    --validation-dir "docs/feature_development_kfold/iteration_$ITERATION" \
    --output-file docs/feature_performance_comprehensive_report.json

echo ""
echo "✅ Performance report generated!"

# Step 4: Compare and Interpret Results
echo ""
echo "🔍 STEP 4: Unified Analysis & Interpretation"
echo "--------------------------------------------"
echo "Comparing both systems and interpreting differences..."
echo ""

uv run scripts/unified_validation_analysis.py

echo ""
echo "🎉 COMPLETE VALIDATION PIPELINE FINISHED!"
echo ""
echo "📁 RESULTS LOCATIONS:"
echo "   • Iterative Results: docs/feature_development_kfold/iteration_$ITERATION/"
echo "   • Unified Results: docs/unified_validation_results/"
echo "   • Performance Report: docs/feature_performance_comprehensive_report.json"
echo "   • Comparative Analysis: docs/unified_validation_analysis.json"
echo ""
echo "📋 NEXT STEPS:"
echo "   1. Review unified_validation_analysis.json for interpretation"
echo "   2. Use Class 0 discriminators for model building"
echo "   3. Use Tier 1 survivors for production features"
echo "   4. Consider threshold adjustments if needed"
