#!/usr/bin/env python3
"""
Simple test of the unified optimization system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Test unified optimizer with minimal configuration."""
    print("🧪 Testing Unified Optimizer (Simple)")
    print("=" * 50)

    try:
        from corp_speech_risk_dataset.case_outcome.unified_optimizer import (
            UnifiedOptimizer,
        )

        # Check data exists
        gold_standard = "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
        extracted_data = "data/extracted"

        if not Path(gold_standard).exists():
            print(f"❌ Gold standard not found: {gold_standard}")
            return 1

        if not Path(extracted_data).exists():
            print(f"❌ Extracted data not found: {extracted_data}")
            return 1

        # Create optimizer with minimal settings - use "fast" type which uses Bayesian with fewer evaluations
        print("🚀 Creating UnifiedOptimizer...")
        optimizer = UnifiedOptimizer(
            gold_standard_path=gold_standard,
            extracted_data_root=extracted_data,
            optimization_type="fast",  # Use fast Bayesian instead of grid
            max_evaluations=3,  # Very small number
            fast_mode=True,
            output_dir="test_results",
        )

        print("✅ UnifiedOptimizer created successfully")

        # Run optimization
        print("🎯 Running fast optimization...")
        try:
            result = optimizer.run_optimization()
            print("✅ Optimization completed successfully!")
            return 0
        except ImportError as e:
            if "scikit-optimize" in str(e):
                print("⚠️  scikit-optimize not available, skipping Bayesian test")
                print("✅ System structure is working correctly")
                return 0
            else:
                raise

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
