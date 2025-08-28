#!/bin/zsh
set -euo pipefail

# Minimal KISS case-level prediction experiments using mirrored MLP predictions
# Reuses infrastructure from run_case_aggregation_experiments but simplified

MIRROR_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/results/corrected_dnt_validation_FINAL/mirror_with_predictions"
OUT_BASE="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/results/minimal_case_prediction_experiments"
DATA_DIR="/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset"

# Test different fold configurations
FOLDS=(3 4)

echo "=== MINIMAL CASE-LEVEL PREDICTION EXPERIMENTS ==="
echo "Mirror dir: $MIRROR_DIR"
echo "Output base: $OUT_BASE"
echo "Data dir: $DATA_DIR"
echo

mkdir -p "$OUT_BASE"

for FOLD in "${FOLDS[@]}"; do
  FOLD_OUT="$OUT_BASE/fold_$FOLD"
  mkdir -p "$FOLD_OUT"

  echo ">>> Running minimal case prediction for fold $FOLD..."

  uv run python scripts/run_minimal_case_prediction_from_mirror.py \
    --mirror-dir "$MIRROR_DIR" \
    --output-dir "$FOLD_OUT" \
    --fold "$FOLD" \
    --data-dir "$DATA_DIR"

  echo ">>> Finished fold $FOLD"
  echo "Artifacts: $FOLD_OUT (case_features.csv, final_dataset.csv, model_summary.json, minimal_case_prediction.log)"
  echo
done

echo "=== GENERATING COMPARATIVE SUMMARY ==="

# Create a summary across all folds
SUMMARY_OUT="$OUT_BASE/comparative_summary"
mkdir -p "$SUMMARY_OUT"

cat > "$SUMMARY_OUT/summary_report.md" << 'EOF'
# Minimal Case-Level Prediction Results

## Experiment Overview
- **Approach**: Simple LR on ~15-20 case-level features from quote-level MLP predictions
- **Features**: Density, positional cutoffs, clustering, shape/calibration
- **Models**: Logistic Regression (L2) + Elastic Net with balanced class weights
- **Evaluation**: Court-suppressed MCC, temporal GroupKFold CV

## Results by Fold

EOF

for FOLD in "${FOLDS[@]}"; do
  FOLD_OUT="$OUT_BASE/fold_$FOLD"
  if [[ -f "$FOLD_OUT/model_summary.json" ]]; then
    echo "### Fold $FOLD" >> "$SUMMARY_OUT/summary_report.md"
    echo "" >> "$SUMMARY_OUT/summary_report.md"

    # Extract key metrics using Python
    python3 -c "
import json
import sys
try:
    with open('$FOLD_OUT/model_summary.json', 'r') as f:
        data = json.load(f)

    for model_name, results in data.items():
        cv_score = results.get('best_cv_score', 0)
        params = results.get('best_params', {})
        print(f'**{model_name}**:')
        print(f'- CV MCC: {cv_score:.4f}')
        print(f'- Best params: {params}')
        print()
except Exception as e:
    print(f'Error processing fold $FOLD: {e}')
    " >> "$SUMMARY_OUT/summary_report.md"

    echo "" >> "$SUMMARY_OUT/summary_report.md"
  fi
done

echo ">>> Summary report generated: $SUMMARY_OUT/summary_report.md"

echo ""
echo "=== EXPERIMENT COMPLETE ==="
echo "All results available in: $OUT_BASE"
echo ""
echo "Key files per fold:"
echo "  - case_features.csv: ~15-20 case-level features"
echo "  - final_dataset.csv: Features + outcomes"
echo "  - model_summary.json: Model performance metrics"
echo "  - minimal_case_prediction.log: Detailed execution log"
echo ""
echo "Comparative summary: $SUMMARY_OUT/summary_report.md"
