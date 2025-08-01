#!/bin/bash

# Comprehensive Bayesian Optimization Script
# Optimizes all 40+ hyperparameters including high/low signal patterns

echo "🚀 COMPREHENSIVE BAYESIAN HYPERPARAMETER OPTIMIZATION"
echo "======================================================"
echo "📊 Optimizing 40+ hyperparameters:"
echo "   • Core extraction parameters (4)"
echo "   • Position thresholds (2)"
echo "   • Case flag thresholds (4)"
echo "   • VotingWeights parameters (30+)"
echo "   • High/Low signal pattern weights (5)"
echo "======================================================"

# Configuration
EVALUATIONS=${1:-100}  # Default to 100 evaluations
FAST_MODE=${2:-"--fast-mode"}  # Default to fast mode

echo "⚙️  Configuration:"
echo "   Max evaluations: $EVALUATIONS"
echo "   Fast mode: $([ "$FAST_MODE" = "--fast-mode" ] && echo "Enabled" || echo "Disabled")"
echo "   Output: bayesian_comprehensive_results.json"
echo ""

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run optimization
echo "🎯 Starting Bayesian optimization..."
uv run python src/corp_speech_risk_dataset/case_outcome/bayesian_optimizer.py \
    --gold-standard data/gold_standard/case_outcome_amounts_hand_annotated.csv \
    --extracted-data data/extracted/courtlistener \
    --max-evaluations "$EVALUATIONS" \
    $FAST_MODE \
    --output bayesian_comprehensive_results.json

echo ""
echo "✅ Optimization complete! Results saved to bayesian_comprehensive_results.json"
