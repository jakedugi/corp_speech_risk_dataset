#!/usr/bin/env python3
"""Test dismissal detection for specific cases."""

import json
import pathlib
from extract_cash_amounts_stage1 import count_dismissal_patterns, DISMISSAL_PATTERNS


def test_dismissal_detection():
    """Test dismissal detection for specific cases."""

    # Test case that should be dismissed
    case_path = pathlib.Path("../../data/extracted/courtlistener/1:16-cv-08364_nysd/")

    total_docs = 0
    dismissal_docs = 0

    print("🔍 Testing dismissal detection...")
    print(f"📁 Case path: {case_path}")
    print(f"📋 Dismissal patterns: {len(DISMISSAL_PATTERNS)}")

    for pattern in DISMISSAL_PATTERNS:
        print(f"   - {pattern.pattern}")

    for path in case_path.rglob("*_stage1.jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        total_docs += 1
                        dismissal_matches = count_dismissal_patterns(text)
                        if dismissal_matches > 0:
                            dismissal_docs += 1
                            print(f"\n✅ Found dismissal pattern in: {text[:300]}...")
                            print(f"   Matches: {dismissal_matches}")
                except json.JSONDecodeError:
                    continue

    print(f"\n📊 Results:")
    print(f"   Total documents: {total_docs}")
    print(f"   Documents with dismissal patterns: {dismissal_docs}")
    print(
        f"   Dismissal ratio: {dismissal_docs/total_docs if total_docs > 0 else 0:.2f}"
    )
    print(f"   Threshold: 0.5")
    print(
        f"   Would be dismissed: {dismissal_docs/total_docs >= 0.5 if total_docs > 0 else False}"
    )


if __name__ == "__main__":
    test_dismissal_detection()
