#!/bin/bash

echo "🔍 Monitoring temporal CV progress..."

# Wait for temporal CV to complete
while ps aux | grep -q "[s]tratified_kfold_case_split.py"; do
    echo "⏳ Temporal CV still running... ($(date))"
    sleep 10
done

echo "✅ Temporal CV completed! Checking output..."

# Verify output was created
if [ -d "data/final_stratified_kfold_splits_leakage_safe/fold_0" ]; then
    echo "📁 Found fold_0 directory - splits created successfully"

    echo "🧪 Starting comprehensive leakage audit..."
    uv run python scripts/comprehensive_leakage_audit.py

    echo "📊 Audit complete! Check results above."
else
    echo "❌ No fold directories found - temporal CV may have failed"
    ls -la data/final_stratified_kfold_splits_leakage_safe/
fi
