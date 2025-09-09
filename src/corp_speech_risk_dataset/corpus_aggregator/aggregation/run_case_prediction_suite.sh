#!/bin/zsh
set -euo pipefail

# Case Prediction Suite - Multiple approaches for using mirrored MLP predictions
# for case-level judicial outcome prediction

MIRROR_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/results/corrected_dnt_validation_FINAL/mirror_with_predictions"
OUT_BASE="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/results/case_prediction_suite"

echo "=== CASE PREDICTION SUITE ==="
echo "Mirror dir: $MIRROR_DIR"
echo "Output base: $OUT_BASE"
echo

mkdir -p "$OUT_BASE"

# Option 1: Minimal KISS approach (pure recipe implementation)
echo ">>> Running MINIMAL KISS approach..."
KISS_OUT="$OUT_BASE/minimal_kiss"
mkdir -p "$KISS_OUT"

uv run python scripts/minimal_kiss_case_predictor.py \
  --mirror-path "$MIRROR_DIR" \
  --output-path "$KISS_OUT/results.json"

echo ">>> KISS approach complete: $KISS_OUT/results.json"
echo

# Option 2: Enhanced minimal with full evaluation protocol
echo ">>> Running ENHANCED MINIMAL approach..."
ENHANCED_OUT="$OUT_BASE/enhanced_minimal"
mkdir -p "$ENHANCED_OUT"

uv run python scripts/run_minimal_case_prediction_from_mirror.py \
  --mirror-dir "$MIRROR_DIR" \
  --output-dir "$ENHANCED_OUT" \
  --fold 4 \
  --data-dir "."

echo ">>> Enhanced minimal complete: $ENHANCED_OUT/"
echo

# Option 3: Using existing infrastructure (if you want to leverage thresholds)
echo ">>> Running WITH INFRASTRUCTURE approach..."
INFRA_OUT="$OUT_BASE/with_infrastructure"
mkdir -p "$INFRA_OUT"

uv run python scripts/run_case_prediction_with_existing_infra.py \
  --mirror-dir "$MIRROR_DIR" \
  --output-dir "$INFRA_OUT" \
  --thresholds token_2500 token_half token_third \
  --fold 4

echo ">>> Infrastructure approach complete: $INFRA_OUT/"
echo

# Generate comparative summary
echo ">>> Generating comparative summary..."
SUMMARY_OUT="$OUT_BASE/comparative_summary"
mkdir -p "$SUMMARY_OUT"

cat > "$SUMMARY_OUT/approach_comparison.md" << 'EOF'
# Case Prediction Approach Comparison

## 1. Minimal KISS
- **File**: `minimal_kiss/results.json`
- **Features**: Pure recipe implementation (~15-20 features)
- **Model**: Simple LR with L2, balanced class weights
- **Evaluation**: Basic CV MCC
- **Best for**: Quick prototyping, pure signal testing

## 2. Enhanced Minimal
- **File**: `enhanced_minimal/`
- **Features**: Same as KISS + full evaluation protocol
- **Model**: LR + Elastic Net with court suppression
- **Evaluation**: Court-suppressed MCC, calibration, multiple thresholds
- **Best for**: Production-ready baseline with robust evaluation

## 3. With Infrastructure
- **File**: `with_infrastructure/`
- **Features**: Leverages existing case_aggregation code
- **Model**: Supports token/docket threshold experiments
- **Evaluation**: Full threshold analysis + existing reporting
- **Best for**: Comprehensive threshold analysis, paper figures

## Key Metrics to Compare

For each approach, check:
- **CV MCC**: Cross-validation Matthews Correlation Coefficient
- **Feature count**: Number of case-level features extracted
- **Cases processed**: Total cases with predictions + outcomes
- **Top features**: Most important predictive features

## Recommended Workflow

1. **Start with Minimal KISS** for quick signal validation
2. **Use Enhanced Minimal** for production baseline
3. **Add Infrastructure** for comprehensive threshold analysis

EOF

echo ">>> Summary generated: $SUMMARY_OUT/approach_comparison.md"
echo

echo "=== CASE PREDICTION SUITE COMPLETE ==="
echo ""
echo "Results available in: $OUT_BASE/"
echo ""
echo "Quick comparison:"
echo "  1. KISS:        $OUT_BASE/minimal_kiss/results.json"
echo "  2. Enhanced:    $OUT_BASE/enhanced_minimal/model_summary.json"
echo "  3. Infra:       $OUT_BASE/with_infrastructure/threshold_results.json"
echo "  4. Summary:     $OUT_BASE/comparative_summary/approach_comparison.md"
echo ""
echo "All approaches use your mirrored MLP predictions for case-level outcome prediction."
echo "Choose the approach that best fits your current needs:"
echo "  - KISS: Quick validation of signal strength"
echo "  - Enhanced: Production-ready with court suppression"
echo "  - Infra: Full threshold analysis leveraging existing code"
