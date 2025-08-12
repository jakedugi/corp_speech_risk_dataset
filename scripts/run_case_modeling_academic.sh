#!/bin/bash
# Run comprehensive case-level modeling with academic rigor

# Set paths
QUOTES_DIR="${1:-data/final_destination/courtlistener_v6_fused_raw_coral_pred}"
OUTPUT_DIR="${2:-data/reports/case_modeling/academic_analysis}"

echo "Running academic case-level modeling analysis..."
echo "Quotes directory: $QUOTES_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run primary analysis using predicted classes only
echo "=== Analysis 1: Using predicted classes only ==="
uv run python -m corp_speech_risk_dataset.case_aggregation.modeling.cli \
    --quotes-dir "$QUOTES_DIR" \
    --output-dir "$OUTPUT_DIR/pred_class_only" \
    --use-pred-class \
    --enable-cv \
    --enable-tuning \
    --cv-folds 10 \
    --enable-stats \
    --enable-fairness \
    --generate-latex \
    --save-features

# Run secondary analysis using full probability features
echo "=== Analysis 2: Using probability features ==="
uv run python -m corp_speech_risk_dataset.case_aggregation.modeling.cli \
    --quotes-dir "$QUOTES_DIR" \
    --output-dir "$OUTPUT_DIR/probability_features" \
    --enable-cv \
    --enable-tuning \
    --cv-folds 10 \
    --enable-stats \
    --enable-fairness \
    --generate-latex \
    --save-features

# Run ablation study with feature selection
echo "=== Analysis 3: Feature selection (top 10 features) ==="
uv run python -m corp_speech_risk_dataset.case_aggregation.modeling.cli \
    --quotes-dir "$QUOTES_DIR" \
    --output-dir "$OUTPUT_DIR/feature_selection" \
    --feature-selection 10 \
    --enable-cv \
    --enable-tuning \
    --cv-folds 5 \
    --enable-stats

# Run focused analysis on specific thresholds
echo "=== Analysis 4: Complete case vs early signals ==="
uv run python -m corp_speech_risk_dataset.case_aggregation.modeling.cli \
    --quotes-dir "$QUOTES_DIR" \
    --output-dir "$OUTPUT_DIR/threshold_comparison" \
    --thresholds complete_case token_2500 docket_third \
    --enable-cv \
    --enable-tuning \
    --cv-folds 10 \
    --enable-stats \
    --generate-latex

echo "Analysis complete! Results saved to: $OUTPUT_DIR"
echo "Key outputs:"
echo "  - Executive summaries: */executive_summary.txt"
echo "  - Full results: */results_*.json"
echo "  - LaTeX tables: */table_*.tex"
echo "  - Figures: */figures_*/"
