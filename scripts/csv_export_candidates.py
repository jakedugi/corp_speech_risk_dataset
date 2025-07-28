#!/usr/bin/env python3
"""
CSV Export and Label Application for Cash Amount Candidates

This script provides CSV export for candidate annotation and applies labels back to JSONL.

Usage:
  # Export candidates to CSV
  python scripts/csv_export_candidates.py export --input candidates.jsonl --output candidates.csv --max-cases 50

  # Apply CSV annotations back to JSONL
  python scripts/csv_export_candidates.py apply --csv-labels candidates_annotated.csv --input candidates.jsonl --output candidates_labeled.jsonl

Author: Jake Dugan <jake.dugan@ed.ac.uk>
"""

import json
import csv
import re
import argparse
from pathlib import Path


def extract_case_id_from_path(file_path):
    """Extract case ID from file path."""
    # Assuming format like "data/extracted/courtlistener/CASE_ID/entries/..."
    parts = Path(file_path).parts
    for i, part in enumerate(parts):
        if part == "courtlistener" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def export_candidates_to_csv(jsonl_path, csv_output_path, max_cases):
    """Export candidates to CSV for annotation following the Tier 2 schema."""
    # Enhanced regex patterns
    JUDGMENT_VERBS = re.compile(
        r"\b(?:award(?:ed)?|order(?:ed)?|grant(?:ed)?|enter(?:ed)?|assess(?:ed)?|recover(?:y|ed)?)\b",
        re.IGNORECASE,
    )

    PROXIMITY_PATTERN = re.compile(
        r"\b(?:settlement|judgment|judgement|damages|award|penalty|fine|amount|paid|cost|price|fee|compensation|restitution|claim|relief|recover|value|sum)\b",
        re.IGNORECASE,
    )

    csv_rows = []
    cases_seen = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                candidate = json.loads(line)

                # Extract case_id
                case_id = candidate.get(
                    "case_id", extract_case_id_from_path(candidate.get("file", ""))
                )

                # Limit number of cases
                if len(cases_seen) >= max_cases:
                    break
                cases_seen.add(case_id)

                # Extract doc_id from file path
                file_name = candidate.get("file", "")
                doc_id = file_name.replace("_text_stage1.jsonl", "").replace("doc_", "")

                # Compute features
                context = candidate.get("context", "")
                has_verb = bool(JUDGMENT_VERBS.search(context))
                has_proximity = bool(PROXIMITY_PATTERN.search(context))

                # Parse amount value
                amount_text = candidate.get("amount", "")
                amount_val = 0.0
                try:
                    if "million" in amount_text.lower():
                        num_part = re.search(r"[\d.]+", amount_text)
                        if num_part:
                            amount_val = float(num_part.group()) * 1_000_000
                    elif "billion" in amount_text.lower():
                        num_part = re.search(r"[\d.]+", amount_text)
                        if num_part:
                            amount_val = float(num_part.group()) * 1_000_000_000
                    else:
                        # Remove currency symbols and parse
                        clean_amount = re.sub(r"[,$USD]", "", amount_text).strip()
                        if clean_amount:
                            amount_val = float(clean_amount)
                except (ValueError, AttributeError):
                    amount_val = 0.0

                csv_rows.append(
                    {
                        "case_id": case_id,
                        "doc_id": doc_id,
                        "page": "",  # Not available in current data
                        "candidate_text": context,
                        "amount_val": amount_val,
                        "has_verb": has_verb,
                        "has_proximity": has_proximity,
                        "rel_position": "",  # Not available in current data
                        "label": "",  # To be filled by annotator
                        "notes": "",  # To be filled by annotator
                        "simhash": candidate.get("simhash", ""),  # For matching back
                        "amount_text": amount_text,  # Original amount for reference
                    }
                )

            except json.JSONDecodeError:
                continue

    # Sort by priority: has_verb desc, has_proximity desc, amount_val desc
    csv_rows.sort(
        key=lambda x: (x["has_verb"], x["has_proximity"], x["amount_val"]), reverse=True
    )

    # Write CSV
    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "case_id",
            "doc_id",
            "page",
            "candidate_text",
            "amount_val",
            "has_verb",
            "has_proximity",
            "rel_position",
            "label",
            "notes",
            "simhash",
            "amount_text",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"✓ Exported {len(csv_rows)} candidates from {len(cases_seen)} cases to {csv_output_path}"
    )
    print("📝 Annotation workflow:")
    print("  1. Open CSV in Excel/Google Sheets")
    print("  2. Sort by has_verb desc, has_proximity desc, amount_val desc")
    print("  3. Fill 'label' column with: TRUE_FINAL, FALSE_MISC, or PROC_DOC")
    print("  4. Add notes as needed")
    print("  5. Save and use 'apply' command to apply back to JSONL")


def apply_csv_labels_to_jsonl(csv_path, jsonl_input_path, jsonl_output_path):
    """Apply CSV annotations back to JSONL output."""
    # Load CSV annotations
    labels_by_simhash = {}
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            simhash = row.get("simhash", "").strip()
            label = row.get("label", "").strip()
            notes = row.get("notes", "").strip()
            if simhash and label:
                labels_by_simhash[simhash] = {"label": label, "notes": notes}

    print(f"✓ Loaded {len(labels_by_simhash)} annotations from {csv_path}")

    # Apply labels to JSONL
    applied_count = 0
    with open(jsonl_input_path, "r", encoding="utf-8") as infile, open(
        jsonl_output_path, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            data = json.loads(line)
            simhash = str(data.get("simhash", ""))

            if simhash in labels_by_simhash:
                annotation = labels_by_simhash[simhash]
                data["annotation_label"] = annotation["label"]
                data["annotation_notes"] = annotation["notes"]
                applied_count += 1

            outfile.write(json.dumps(data) + "\n")

    print(f"✓ Applied {applied_count} annotations to {jsonl_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CSV export and label application for candidates"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export candidates to CSV")
    export_parser.add_argument(
        "--input", required=True, help="Input JSONL file with candidates"
    )
    export_parser.add_argument(
        "--output", required=True, help="Output CSV file for annotation"
    )
    export_parser.add_argument(
        "--max-cases", type=int, default=50, help="Max cases to export"
    )

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply CSV annotations to JSONL")
    apply_parser.add_argument("--csv-labels", required=True, help="Annotated CSV file")
    apply_parser.add_argument("--input", required=True, help="Original JSONL file")
    apply_parser.add_argument(
        "--output", required=True, help="Output JSONL with annotations"
    )

    args = parser.parse_args()

    if args.command == "export":
        export_candidates_to_csv(args.input, args.output, args.max_cases)
    elif args.command == "apply":
        apply_csv_labels_to_jsonl(args.csv_labels, args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
