#!/usr/bin/env python3
"""
extract_amounts_stage1.py

Recursively scans `_text_stage1.jsonl` files under the specified directory,
extracts dollar amounts and “million/billion” phrases with surrounding context,
deduplicates via Simhash, groups by case (two levels up), and writes results to
a JSONL file.  At the end, prints file‐level and case‐level hit summaries, plus
counts of `_stage3.jsonl` entries in hit vs. no‐hit cases.
"""

import re
import json
import glob
import os
import hashlib
import argparse
from collections import defaultdict


DEFAULT_INPUT_PATTERN = "data/extracted/**/*_text_stage1.jsonl"
DEFAULT_OUTPUT_FILE = "data/output_stage1_amounts.jsonl"
DEFAULT_CONTEXT_CHARS = 0
DEFAULT_BASE_DIR = "data/extracted"
DEFAULT_MIN_AMOUNT = 1000000

CONTEXT_CHARS = DEFAULT_CONTEXT_CHARS
MIN_AMOUNT = DEFAULT_MIN_AMOUNT

# Regexes
AMOUNT_REGEX = re.compile(
    r"\$[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?|"  # $1,234,567.89
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:million|billion)\b",  # 5 million
    re.IGNORECASE,
)

PROXIMITY_PATTERN = re.compile(
    r"\b(?:settlement|judgment)\b(?:\W+\w+){0,10}\W+", re.IGNORECASE
)


def compute_simhash(text: str) -> int:
    """Inline 64-bit Simhash implementation."""
    v = [0] * 64
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest()[-16:], 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    fingerprint = 0
    for i, weight in enumerate(v):
        if weight > 0:
            fingerprint |= 1 << i
    return fingerprint


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract dollar amounts from stage1 JSONL files"
    )
    p.add_argument(
        "--input-pattern",
        default=DEFAULT_INPUT_PATTERN,
        help="Glob pattern for input `_text_stage1.jsonl` files",
    )
    p.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write extracted JSONL records",
    )
    p.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHARS,
        help="Number of chars of context before/after each match",
    )
    p.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Base directory under which to look for `_stage3.jsonl` files",
    )
    p.add_argument(
        "--min-amount",
        type=float,
        default=DEFAULT_MIN_AMOUNT,
        help="Skip numeric values below this threshold (unless 'million'/'billion')",
    )
    return p.parse_args()


def main(
    input_pattern: str,
    output_file: str,
    context_chars: int,
    base_dir: str,
    min_amount: float,
):
    seen = set()
    total_files = 0
    files_with_hits = 0
    case_dirs = set()
    case_with_hits = set()

    with open(output_file, "w", encoding="utf-8") as out_f:
        for path in glob.glob(input_pattern, recursive=True):
            total_files += 1
            case_id = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(path)))
            )
            case_dirs.add(case_id)
            file_hit = False

            with open(path, encoding="utf-8") as in_f:
                for line in in_f:
                    text = json.loads(line).get("text", "")
                    for m in AMOUNT_REGEX.finditer(text):
                        amt = m.group(0)
                        norm = amt.lower().replace(",", "").replace("$", "").strip()

                        # numeric threshold
                        if "million" not in norm and "billion" not in norm:
                            try:
                                val = float(re.search(r"[0-9.]+", norm)[0])
                                if val < min_amount:
                                    continue
                            except Exception:
                                continue

                        start, end = m.span()
                        pre = text[max(0, start - context_chars) : start]
                        post = text[end : end + context_chars]
                        ctx = (pre + amt + post).replace("\n", " ")

                        if PROXIMITY_PATTERN.search(ctx) is None:
                            continue

                        sim = compute_simhash(ctx)
                        if sim in seen:
                            continue

                        seen.add(sim)
                        file_hit = True
                        case_with_hits.add(case_id)

                        out_f.write(
                            json.dumps(
                                {
                                    "file": os.path.basename(path),
                                    "amount": amt,
                                    "context": ctx,
                                    "simhash": sim,
                                }
                            )
                            + "\n"
                        )

            if file_hit:
                files_with_hits += 1

    # Summaries
    total_cases = len(case_dirs)
    hit_cases = len(case_with_hits)
    print(f"Extraction complete → {output_file}")
    print(
        f"Files scanned: {total_files}, hits: {files_with_hits}, no-hits: {total_files - files_with_hits}"
    )
    print(
        f"Cases total: {total_cases}, hit-cases: {hit_cases}, no-hit-cases: {total_cases - hit_cases}"
    )

    def count_stage3(case_set):
        cnt = 0
        for case in case_set:
            for fn in glob.glob(
                os.path.join(base_dir, case, "**", "*_stage3.jsonl"), recursive=True
            ):
                with open(fn, encoding="utf-8") as f3:
                    cnt += sum(1 for _ in f3)
        return cnt

    print(f"Stage-3 entries in hit-cases    : {count_stage3(case_with_hits)}")
    print(
        f"Stage-3 entries in no-hit-cases : {count_stage3(case_dirs - case_with_hits)}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_pattern,
        args.output_file,
        args.context_chars,
        args.base_dir,
        args.min_amount,
    )
