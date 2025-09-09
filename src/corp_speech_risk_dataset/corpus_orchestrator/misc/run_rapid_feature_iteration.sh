#!/bin/bash
"""
Complete Rapid Feature Iteration Example

This script demonstrates the complete workflow for rapidly developing and testing
interpretable features that can discriminate all classes, especially class 0.

Run this after adding new features to features.py
"""

ITERATION=${1:-"test_1"}
INPUT_PATTERN="data/final_destination/courtlistener_v6_fused_raw_coral_pred/doc_*_text_stage15.jsonl"
SAMPLE_SIZE=10000

echo "🚀 RAPID FEATURE ITERATION: $ITERATION"
echo "=" * 60
echo "Sample size: $SAMPLE_SIZE"
echo ""

# Step 1: Test feature extraction quickly
echo "STEP 1: Quick Feature Test"
echo "-" * 30
python scripts/iterative_feature_development.py \
    --iteration "$ITERATION" \
    --input "$INPUT_PATTERN" \
    --sample-size $SAMPLE_SIZE \
    --test-class-discrimination \
    --auto-governance-update

if [ $? -ne 0 ]; then
    echo "❌ Feature extraction failed!"
    exit 1
fi

echo ""
echo "STEP 2: Analyze Results"
echo "-" * 30

# Check if we have results
RESULTS_DIR="docs/feature_development/iteration_$ITERATION"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Show summary
if [ -f "$RESULTS_DIR/iteration_${ITERATION}_summary.md" ]; then
    echo "📊 ITERATION SUMMARY:"
    cat "$RESULTS_DIR/iteration_${ITERATION}_summary.md"
fi

echo ""
echo "STEP 3: Class 0 Discrimination Check"
echo "-" * 30

# Check class 0 discriminators
if [ -f "$RESULTS_DIR/class_0_discrimination.csv" ]; then
    echo "🎯 Features that help discriminate Class 0:"
    python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('$RESULTS_DIR/class_0_discrimination.csv')

# Filter to good class 0 discriminators
good_class0 = df[
    (df.get('class_0_separation', 0) > 0.1) &
    (df.get('class_0_significance', 1) < 0.05) &
    (df.get('class_0_auc', 0) > 0.55)
].sort_values('class_0_auc', ascending=False) if 'class_0_auc' in df.columns else pd.DataFrame()

if len(good_class0) > 0:
    print(f'✅ Found {len(good_class0)} features that discriminate Class 0:')
    for _, row in good_class0.head(10).iterrows():
        feature = row['feature'].replace('interpretable_', '')
        auc = row.get('class_0_auc', 0)
        sep = row.get('class_0_separation', 0)
        print(f'   {feature:<40} AUC: {auc:.3f}, Sep: {sep:.3f}')
else:
    print('❌ NO FEATURES DISCRIMINATE CLASS 0 WELL!')
    print('   Need to add more class 0 specific features')
    print('   Try: compliance language, absence indicators, procedural terms')
"
fi

echo ""
echo "STEP 4: Overall Feature Quality"
echo "-" * 30

if [ -f "$RESULTS_DIR/discriminative_power.csv" ]; then
    echo "📈 Top discriminative features overall:"
    python3 -c "
import pandas as pd

df = pd.read_csv('$RESULTS_DIR/discriminative_power.csv')

# Top features by mutual information
top_mi = df[
    (df.get('mutual_info', 0) > 0.005) &
    (df.get('kw_pvalue', 1) < 0.1)
].sort_values('mutual_info', ascending=False) if 'mutual_info' in df.columns else pd.DataFrame()

if len(top_mi) > 0:
    print(f'✅ {len(top_mi)} features pass quality thresholds:')
    for _, row in top_mi.head(15).iterrows():
        feature = row['feature'].replace('interpretable_', '')
        mi = row.get('mutual_info', 0)
        p = row.get('kw_pvalue', 1)
        print(f'   {feature:<40} MI: {mi:.4f}, p: {p:.4f}')
else:
    print('❌ NO FEATURES PASS QUALITY THRESHOLDS!')
    print('   All features failed - need better feature engineering')
"
fi

echo ""
echo "STEP 5: Governance Update (if needed)"
echo "-" * 30

# Check if governance update was generated
GOVERNANCE_UPDATE="$RESULTS_DIR/governance_update_iteration_${ITERATION}.txt"
if [ -f "$GOVERNANCE_UPDATE" ]; then
    echo "📝 Governance update generated:"
    echo "   File: $GOVERNANCE_UPDATE"
    echo "   Failed features to block:"
    grep '^    r"' "$GOVERNANCE_UPDATE" | wc -l | xargs echo "   Count:"

    echo ""
    echo "💡 To apply governance update:"
    echo "   1. Review: cat $GOVERNANCE_UPDATE"
    echo "   2. Apply: python scripts/auto_update_column_governance.py \\"
    echo "              --test-results $RESULTS_DIR/discriminative_power.csv \\"
    echo "              --iteration $ITERATION \\"
    echo "              --apply-update"
else
    echo "✅ No governance update needed - all features passed!"
fi

echo ""
echo "=" * 60
echo "✅ RAPID ITERATION $ITERATION COMPLETE!"
echo "=" * 60

echo ""
echo "📊 NEXT STEPS:"
echo "1. Review results in: $RESULTS_DIR/"
echo "2. If class 0 discrimination is poor, add more low-risk features"
echo "3. Apply governance updates to block failed features"
echo "4. Run next iteration with improved features"
echo "5. Once satisfied, run full pipeline for production"

echo ""
echo "🎯 PRODUCTION PIPELINE (when ready):"
echo "   bash scripts/run_unified_feature_pipeline.sh"

echo ""
echo "🔄 NEXT ITERATION:"
echo "   # Add new features to features.py, then:"
echo "   bash scripts/run_rapid_feature_iteration.sh $((${ITERATION//[!0-9]/} + 1))"
