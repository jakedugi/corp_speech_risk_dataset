#!/bin/bash
"""
run_final_evaluation.sh

Convenience script to run the final evaluation with proper environment setup.
This script sets the necessary environment variables and executes the final
evaluation using the tuned hyperparameters.

Usage:
    ./scripts/run_final_evaluation.sh

Or with custom paths:
    ANNOTATIONS=/path/to/annotations.csv EXTRACTED_ROOT=/path/to/extracted ./scripts/run_final_evaluation.sh

Author: Jake Dugan <jake.dugan@ed.ac.uk>
"""

set -e  # Exit on any error

# Default paths (can be overridden by environment variables)
ANNOTATIONS=${ANNOTATIONS:-"data/gold_standard/case_outcome_amounts_hand_annotated.csv"}
EXTRACTED_ROOT=${EXTRACTED_ROOT:-"data/extracted"}

echo "Corporate Speech Risk Dataset - Final Evaluation Runner"
echo "======================================================"
echo "Using environment variables:"
echo "  ANNOTATIONS=${ANNOTATIONS}"
echo "  EXTRACTED_ROOT=${EXTRACTED_ROOT}"
echo ""

# Check if files exist
if [ ! -f "$ANNOTATIONS" ]; then
    echo "Error: Annotations file not found: $ANNOTATIONS"
    echo "Please set ANNOTATIONS environment variable or ensure file exists."
    exit 1
fi

if [ ! -d "$EXTRACTED_ROOT" ]; then
    echo "Error: Extracted root directory not found: $EXTRACTED_ROOT"
    echo "Please set EXTRACTED_ROOT environment variable or ensure directory exists."
    exit 1
fi

# Run the evaluation using uv (as per project requirements)
echo "Running final evaluation..."
echo "Command: uv run python src/corp_speech_risk_dataset/case_outcome/final_evaluate.py --annotations \"$ANNOTATIONS\" --extracted-root \"$EXTRACTED_ROOT\""
echo ""

uv run python src/corp_speech_risk_dataset/case_outcome/final_evaluate.py \
    --annotations "$ANNOTATIONS" \
    --extracted-root "$EXTRACTED_ROOT"

echo ""
echo "Final evaluation completed!"
