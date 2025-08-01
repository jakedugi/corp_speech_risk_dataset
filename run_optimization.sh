#!/bin/bash

# run_optimization.sh
# Simple script to run the unified optimization system with proper environment setup

echo "🚀 Corporate Speech Risk Dataset - Unified Optimization Runner"
echo "=============================================================="

# Check if we're in the right directory
if [ ! -f "src/corp_speech_risk_dataset/case_outcome/unified_optimizer.py" ]; then
    echo "❌ Error: Run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: src/corp_speech_risk_dataset/case_outcome/unified_optimizer.py"
    exit 1
fi

# Set up Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Default optimization type
OPTIMIZATION_TYPE="${1:-bayesian}"
MAX_EVALUATIONS="${2:-100}"

echo "📊 Configuration:"
echo "   Optimization Type: $OPTIMIZATION_TYPE"
echo "   Max Evaluations: $MAX_EVALUATIONS"
echo "   Python Path: $PYTHONPATH"
echo ""

# Check for required dependencies
echo "🔍 Checking dependencies..."

python3 -c "import pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required dependencies: pandas, numpy"
    echo "   Install with: pip install pandas numpy"
    exit 1
fi

if [ "$OPTIMIZATION_TYPE" = "bayesian" ]; then
    python3 -c "import skopt" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "❌ Missing scikit-optimize for Bayesian optimization"
        echo "   Install with: pip install scikit-optimize"
        exit 1
    fi
fi

echo "✅ Dependencies check passed"
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p optimization_results

# Run the optimization
echo "🎯 Starting optimization..."
echo "   Command: python3 src/corp_speech_risk_dataset/case_outcome/unified_optimizer.py --type $OPTIMIZATION_TYPE --max-evaluations $MAX_EVALUATIONS"
echo ""

python3 src/corp_speech_risk_dataset/case_outcome/unified_optimizer.py \
    --type "$OPTIMIZATION_TYPE" \
    --max-evaluations "$MAX_EVALUATIONS" \
    --fast-mode \

EXIT_CODE=$?

echo ""
echo "=============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Optimization completed successfully!"
    echo "📁 Results saved to: optimization_results/"
    echo "📝 Logs saved to: logs/"
else
    echo "❌ Optimization failed with exit code: $EXIT_CODE"
fi

echo ""
echo "🔍 To monitor optimization in real-time (in another terminal):"
echo "   python3 src/corp_speech_risk_dataset/case_outcome/monitor_optimization.py"
echo ""
echo "📊 To view results:"
echo "   ls -la optimization_results/"
echo "   ls -la logs/"

exit $EXIT_CODE
