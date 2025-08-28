#!/usr/bin/env python3
"""Check what was actually stored in the CSV"""

import csv
from pathlib import Path

csv_path = Path("src/corp_speech_risk_dataset/disso_prompt/disso_sections.csv")

# Read the CSV and find section 3.1.1
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

    # Find the row for section 3.1.1
    for row in rows:
        if row.get("outline_ref_json") == '{"ref": "3.1.1", "depth": 3}':
            print("Found section 3.1.1!")
            print("-" * 60)

            # Check the stored markdown
            md_content = row.get("final_output_markdown", "")
            print(f"final_output_markdown length: {len(md_content)} chars")
            if md_content:
                print(f"Preview: {repr(md_content[:200])}...")
            else:
                print("(empty)")

            # Check if JSON was stored
            json_content = row.get("final_output_json", "")
            print(f"\nfinal_output_json length: {len(json_content)} chars")
            if json_content:
                print(f"Preview: {repr(json_content[:200])}...")
            else:
                print("(empty)")

            # Check the LaTeX
            latex_content = row.get("final_output_latex", "")
            print(f"\nfinal_output_latex length: {len(latex_content)} chars")
            if latex_content:
                print(f"Preview: {repr(latex_content[:200])}...")
            else:
                print("(empty)")

            break
    else:
        print("Section 3.1.1 not found in CSV")
