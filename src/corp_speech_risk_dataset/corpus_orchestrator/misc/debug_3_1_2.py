#!/usr/bin/env python3
"""Debug the specific issue with output_3_1_2.json"""

import json5
import re
from pathlib import Path

# Read the problematic file
json_file = Path("src/corp_speech_risk_dataset/disso_prompt/output_3_1_2.json")
content = json_file.read_text(encoding="utf-8")

print("Original file info:")
print(f"Length: {len(content)} chars")

# Find what's at column 484 of line 2
lines = content.split("\n")
if len(lines) >= 2:
    line2 = lines[1]
    print(f"\nLine 2 length: {len(line2)}")
    print(f"Line 2 preview: {repr(line2[:100])}")
    if len(line2) >= 484:
        print(f"Character at column 484: {repr(line2[483])}")
        print(f"Context around column 484: {repr(line2[470:500])}")

# Check for smart quotes
smart_quotes = ["\u201c", "\u201d", "\u2018", "\u2019"]
for i, line in enumerate(lines[:10]):  # Check first 10 lines
    for quote in smart_quotes:
        if quote in line:
            pos = line.find(quote)
            print(f"\nFound {repr(quote)} at line {i+1}, position {pos+1}")
            print(f"Context: {repr(line[max(0,pos-20):pos+20])}")

# Test if json5 can parse it directly
print("\n" + "=" * 60)
print("Testing json5 parsing:")
try:
    data = json5.loads(content)
    print("✅ json5 parsed successfully!")
except Exception as e:
    print(f"❌ json5 failed: {e}")

# Test with simple smart quote replacement
print("\n" + "=" * 60)
print("Testing with simple smart quote replacement:")
fixed = content.replace("\u201c", '"').replace("\u201d", '"')
fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")

try:
    data = json5.loads(fixed)
    print("✅ Simple replacement worked!")
except Exception as e:
    print(f"❌ Simple replacement failed: {e}")
    # Show where it failed
    if hasattr(e, "lineno") and hasattr(e, "colno"):
        error_line = (
            fixed.split("\n")[e.lineno - 1]
            if e.lineno <= len(fixed.split("\n"))
            else ""
        )
        print(f"Error at line {e.lineno}, column {e.colno}")
        if error_line and e.colno <= len(error_line):
            print(f"Context: {repr(error_line[max(0, e.colno-20):e.colno+20])}")
            print(f"Character at error: {repr(error_line[e.colno-1])}")
