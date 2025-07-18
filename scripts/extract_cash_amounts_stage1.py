#!/usr/bin/env python3
"""
extract_amounts_stage1.py

Recursively scans only the `_text_stage1.jsonl` files under the specified directory,
extracts dollar amounts and “million/billion” phrases with surrounding context,
deduplicates via an inline Simhash, groups by case (two levels up), and writes
results to JSONL.  At the end, prints both file‐level and case‐level hit summaries,
and counts total `_stage3.jsonl` entries in hit vs. no‐hit cases.
"""

import re
import json
import glob
import os
import hashlib
from collections import defaultdict

# Configuration
INPUT_PATTERN = "data/extracted/**/*_text_stage1.jsonl"
OUTPUT_FILE = "data/output_stage1_amounts.jsonl"
CONTEXT_CHARS = 100  # characters of context before/after match
BASE_DIR = "data/extracted"

# Match dollar amounts with thousands separators, or "X million/billion"
AMOUNT_REGEX = re.compile(
    r"\$[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?|"  # $1,000,000.00
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:million|billion)\b",  # 5 million
    re.IGNORECASE,
)

# Narrowed keywords for true settlement/judgment language
PROXIMITY_PATTERN = re.compile(
    r"\b(?:settlement|judgment)\b(?:\W+\w+){0,10}\W+", re.IGNORECASE
)


def compute_simhash(text: str) -> int:
    """
    Inline 64-bit Simhash implementation.
    """
    v = [0] * 64
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[-16:], 16)
        for i in range(64):
            bit = 1 << i
            v[i] += 1 if (h & bit) else -1
    fingerprint = 0
    for i, weight in enumerate(v):
        if weight > 0:
            fingerprint |= 1 << i
    return fingerprint


def main():
    seen = set()
    total_files = 0
    files_with_hits = 0

    # case-level tracking
    case_dirs = set()
    case_with_hits = set()

    # Stage-1 extraction
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for path in glob.glob(INPUT_PATTERN, recursive=True):
            total_files += 1
            # derive the case ID as the folder three levels above the .jsonl
            case_id = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(path)))
            )
            case_dirs.add(case_id)

            file_has_hit = False

            with open(path, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    text = record.get("text", "")

                    for m in AMOUNT_REGEX.finditer(text):
                        amt = m.group(0)
                        norm = amt.lower().replace(",", "").replace("$", "").strip()

                        # skip trivial values under 1M unless X million/billion
                        if "million" not in norm and "billion" not in norm:
                            try:
                                val = float(re.findall(r"[0-9.]+", norm)[0])
                                if val < 1_000_000:
                                    continue
                            except (IndexError, ValueError):
                                continue

                        start, end = m.span()
                        pre = text[max(0, start - CONTEXT_CHARS) : start]
                        post = text[end : end + CONTEXT_CHARS]
                        ctx = (pre + amt + post).replace("\n", " ").strip()

                        # exclude CAFA boilerplate
                        if re.search(r"amount in controversy", ctx, re.IGNORECASE):
                            continue

                        # require “settlement” or “judgment” within 10 words
                        if not PROXIMITY_PATTERN.search(ctx):
                            continue

                        sh = compute_simhash(ctx)
                        if sh in seen:
                            continue

                        seen.add(sh)
                        file_has_hit = True
                        case_with_hits.add(case_id)

                        out_record = {
                            "file": os.path.basename(path),
                            "amount": amt,
                            "context": ctx,
                            "simhash": sh,
                        }
                        out.write(json.dumps(out_record) + "\n")

            if file_has_hit:
                files_with_hits += 1

    files_without_hits = total_files - files_with_hits
    total_cases = len(case_dirs)
    cases_with_hits = len(case_with_hits)
    cases_without_hits = total_cases - cases_with_hits

    print(f"Extraction complete. See {OUTPUT_FILE}")
    print(
        f"Scanned {total_files} files: {files_with_hits} had ≥1 hit, {files_without_hits} had none."
    )
    print(
        f"Across {total_cases} cases: {cases_with_hits} had ≥1 hit, {cases_without_hits} had none."
    )

    # 1) Total-stage-3 entries across a set of cases
    def count_stage3_totals(case_set):
        total = 0
        for case in case_set:
            pattern = os.path.join(BASE_DIR, case, "**", "*_stage3.jsonl")
            for fn in glob.glob(pattern, recursive=True):
                with open(fn, encoding="utf-8") as f:
                    total += sum(1 for _ in f)
        return total

    total_pos_stage3 = count_stage3_totals(case_with_hits)
    total_neg_stage3 = count_stage3_totals(case_dirs - case_with_hits)

    print(f"Stage-3 entries in hit-cases    : {total_pos_stage3}")
    print(f"Stage-3 entries in no-hit cases : {total_neg_stage3}")

    # 2) Per-case stage-3 counts for averages, top/bottom
    def count_stage3_by_case(case_set):
        counts = {}
        for case in case_set:
            n = 0
            pattern = os.path.join(BASE_DIR, case, "**", "*_stage3.jsonl")
            for fn in glob.glob(pattern, recursive=True):
                with open(fn, encoding="utf-8") as f:
                    n += sum(1 for _ in f)
            counts[case] = n
        return counts

    pos_counts = count_stage3_by_case(case_with_hits)
    neg_counts = count_stage3_by_case(case_dirs - case_with_hits)

    # compute per-case stage3 counts
    avg_pos = sum(pos_counts.values()) / max(1, len(pos_counts))
    avg_neg = sum(neg_counts.values()) / max(1, len(neg_counts))

    print(f"Average stage-3 records per hit-case   : {avg_pos:.1f}")
    print(f"Average stage-3 records per no-hit case: {avg_neg:.1f}")

    # Show the top 3 and bottom 3 by count in each group:
    for label, counts in [("HIT", pos_counts), ("NO-HIT", neg_counts)]:
        print(f"\nTop 3 {label} cases by #records:")
        for case, n in sorted(counts.items(), key=lambda x: -x[1])[:3]:
            print(f"  {case}: {n}")
        print(f"Bottom 3 {label} cases by #records:")
        for case, n in sorted(counts.items(), key=lambda x: x[1])[:3]:
            print(f"  {case}: {n}")


if __name__ == "__main__":
    main()
