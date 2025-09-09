import json
import os
from pathlib import Path

print("\n📊 Test Results Summary")
print("=" * 50)

runs_dir = Path("runs")
for test_dir in sorted(runs_dir.glob("test*")):
    if test_dir.is_dir():
        # Try to find log files with metrics
        print(f"\n{test_dir.name}:")

        # Look for any .log or .json files
        for file in test_dir.iterdir():
            if file.suffix in [".log", ".json"]:
                print(f"  - Found: {file.name}")

        # Check if model was saved
        model_file = test_dir.parent / f"{test_dir.name}.joblib"
        if model_file.exists():
            print(f"  ✅ Model saved: {model_file.name}")
        else:
            print(f"  ❌ Model not found")
